"""Runner WO + controlador SHAP **por agente** en regimen **MaxFES**.

Pipeline (setup experimental: criterio de parada MaxFES):

1. Inicializa la poblacion (uniforme CEC / repair[+local_search] TMLAP). Las
   evaluaciones de inicializacion de TMLAP se contabilizan como ``fes_init``.
2. Bucle ``while not budget.exhausted()``:
   a. Evalua la poblacion (consciente del presupuesto -> corte exacto en MaxFES).
   b. Actualiza el mejor personal de cada agente y su ``last_improve_fes``.
   c. Calcula las 6 senales del WO con el schedule por FES (``phi = fes/MaxFES``).
   d. Detecta estancamiento POR AGENTE (ventana = 10% de MaxFES) y, sobre los
      agentes mas estancados, aplica SHAP por agente -> accion -> aceptacion.
   e. Aplica el movimiento WO poblacional.
3. Reporta best@MaxFES, gap al optimo y telemetria de FES (search/shap/intervention/init).

Uso:

    python -m runners.run_wo_shap --problem cec2022:F6 --dim 10 --agents 30 \\
        --max-fes 5000,50000,500000,5000000 --runs 51 --profile soft \\
        --output experiments/cec_F6_shap

    python -m runners.run_wo_shap --problem tmlap:1.instancia_simple.txt \\
        --agents 30 --max-fes 50000 --runs 51 --output experiments/tmlap_simple_shap
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from problems.cec2022 import FUNCTION_IDS as CEC_FUNCTION_IDS
from problems.factory import parse_problem_spec
from problems.tmlap import with_exact_optimum
from shap_controller import (
    SHAPFitnessController,
    dispatch_rescue_single,
    improvement_threshold,
)
from wo_core.agent_sim import make_value_function_for_agent
from wo_core.diversity import population_diversity
from wo_core.fes import FESBudget, counting_objective
from wo_core.walrus import (
    apply_wo_movement,
    evaluate_and_update_leaders,
    iteration_signals,
    walrus_role_counts,
)

# Compuertas globales: bloquean a TODOS los agentes -> detienen el escaneo de la
# iteracion. Las demas razones son por-agente y permiten probar el siguiente.
_GLOBAL_BLOCK_REASONS = {
    "max_interventions",
    "shap_budget_exhausted",
    "insufficient_budget_for_shap",
    "guard_window",
    "late_fraction",
    "adaptive_outcome_cooldown",
    "effective_cooldown",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="WO + controlador SHAP por agente en regimen MaxFES."
    )
    parser.add_argument("--problem", required=True)
    parser.add_argument("--dim", type=int, default=10)
    parser.add_argument("--clients", type=int, default=None)
    parser.add_argument("--hubs", type=int, default=None)
    parser.add_argument("--agents", type=int, default=30)
    parser.add_argument(
        "--max-fes",
        type=str,
        default="50000",
        help="Presupuesto(s) MaxFES, separados por coma. Ej: 5000,50000,500000,5000000",
    )
    parser.add_argument("--runs", type=int, default=51)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--init-mode", choices=["local_search", "random"], default="local_search",
        help="Solo TMLAP: 'random' salta local_search (instancias grandes).",
    )
    parser.add_argument(
        "--no-exact-optimum", action="store_true",
        help="No calcular el optimo exacto (instancias TMLAP intratables): gap=NaN.",
    )
    parser.add_argument("--output", required=True)
    # NOTA: el controlador tiene una configuracion UNICA fija (sin perfiles ni
    # knobs de tuning), conforme al setup experimental. Ver shap_controller/profiles.py.
    return parser.parse_args()


def _parse_max_fes(text):
    values = [int(float(token.strip())) for token in str(text).split(",") if token.strip()]
    if not values:
        raise ValueError(f"--max-fes invalido: {text!r}")
    return values


def _make_problems(args):
    if args.problem.startswith("cec2022:"):
        target = args.problem.split(":", 1)[1].strip()
        if target.lower() == "all":
            return [
                (parse_problem_spec(f"cec2022:F{fid}", dim=args.dim), f"F{fid}")
                for fid in CEC_FUNCTION_IDS
            ]
        problem = parse_problem_spec(args.problem, dim=args.dim)
        return [(problem, f"F{problem.function_id}")]
    if args.problem.startswith("tmlap:"):
        problem = parse_problem_spec(args.problem, clients=args.clients, hubs=args.hubs)
        return [(problem, problem.name)]
    raise ValueError(f"Familia desconocida: {args.problem!r}")


def _seed_for(args, max_fes_idx, problem_idx, run_idx):
    # Cada (MaxFES, problema, corrida) arranca de cero con su propia semilla.
    return int(args.seed + max_fes_idx * 100000 + problem_idx * 1000 + run_idx)


def run_one(problem, args, max_fes, max_fes_idx, run_id, problem_idx, problem_label):
    seed = _seed_for(args, max_fes_idx, problem_idx, run_id - 1)
    rng = np.random.default_rng(seed)
    start = time.perf_counter()

    n_agents = args.agents
    dim, lb, ub = problem.dim, problem.lb, problem.ub
    budget = FESBudget(max_fes)
    eval_search = counting_objective(problem.evaluate, budget, "search")
    eval_shap = counting_objective(problem.evaluate, budget, "shap")
    eval_intervention = counting_objective(problem.evaluate, budget, "intervention")

    if getattr(problem, "family", None) == "tmlap":
        positions = problem.initial_population(
            n_agents, rng, init_mode=args.init_mode,
            on_eval=lambda: budget.spend("init", 1),
        )
    else:
        positions = problem.initial_population(n_agents, rng)

    best_pos = np.zeros(dim, dtype=float)
    second_pos = np.zeros(dim, dtype=float)
    best_score = float("inf")
    second_score = float("inf")
    male_count, female_count, child_count = walrus_role_counts(n_agents)
    role_counts = (male_count, female_count, child_count)

    controller = SHAPFitnessController(
        max_fes=max_fes, n_agents=n_agents, random_state=seed,
    )
    window = controller.profile.stagnation_window

    pbest = np.full(n_agents, np.inf, dtype=float)
    last_improve_fes = np.zeros(n_agents, dtype=np.int64)
    diversity_ref = None
    curve = []  # filas (fes, best_fitness)
    initial_fitness = np.nan

    while not budget.exhausted():
        fes_start = budget.total  # FES al inicio de la iteracion -> phi del schedule

        # (a) Evaluacion de poblacion, consciente del presupuesto (corte exacto).
        (
            positions, fitness_values, best_score, best_pos, second_score, second_pos
        ) = evaluate_and_update_leaders(
            positions, lb, ub, eval_search,
            best_score, best_pos, second_score, second_pos, budget=budget,
        )
        if not np.isfinite(second_score):
            second_pos = best_pos.copy()
        if not np.isfinite(initial_fitness):
            initial_fitness = float(best_score)

        # (b) Mejor personal por agente + reloj de su ultima mejora.
        for i in range(n_agents):
            f = fitness_values[i]
            if not np.isfinite(f):
                continue
            if not np.isfinite(pbest[i]):
                pbest[i] = f
                last_improve_fes[i] = budget.total
            elif f < pbest[i] - improvement_threshold(pbest[i]):
                pbest[i] = f
                last_improve_fes[i] = budget.total

        curve.append((int(budget.total), float(best_score)))
        if budget.exhausted():
            break

        # (c) Senales del WO por FES (phi = fes_start / max_fes).
        alpha, beta, A, R, danger_signal, safety_signal = iteration_signals(
            fes_start, max_fes, rng
        )
        signal_state = {
            "alpha": float(alpha), "beta": float(beta), "A": float(A), "R": float(R),
            "danger_signal": float(danger_signal), "safety_signal": float(safety_signal),
        }
        controller.record_state(signal_state, best_score)

        raw_div = float(population_diversity(positions))
        if diversity_ref is None:
            diversity_ref = max(raw_div, 1e-12)
        diversity_norm = raw_div / (diversity_ref + 1e-12)

        # (d) Estancamiento POR AGENTE -> SHAP -> accion. Mas estancados primero.
        fes_since = budget.total - last_improve_fes
        order = np.argsort(-fes_since)
        intervened_this_iter = 0
        top_blocked = None

        for i in order:
            i = int(i)
            if budget.exhausted():
                break
            fsi = int(budget.total - last_improve_fes[i])
            if fsi < window:
                break  # orden descendente: ya no hay candidatos estancados

            state = {
                **signal_state,
                "agent_index": i,
                "agent_fitness": float(fitness_values[i]),
                "fes_since_improve": fsi,
                "diversity_norm": float(diversity_norm),
                "raw_diversity": raw_div,
                "fes": int(budget.total),
            }
            should, reason = controller.should_consider_intervention(state, budget)
            if not should:
                if top_blocked is None:
                    top_blocked = (i, reason, fsi, float(fitness_values[i]))
                if reason in _GLOBAL_BLOCK_REASONS:
                    break
                continue  # razon por-agente: probar el siguiente agente

            # SHAP por agente (gasta ~2^6 * steps FES en el bucket 'shap').
            value_fn = make_value_function_for_agent(
                eval_shap, i, positions, lb, ub, dim, role_counts,
                best_pos, second_pos, signal_state, max_fes, fes_start,
                controller.shapley_steps, rng=rng,
            )
            shap_info = controller.explain_fitness(state, value_function=value_fn)
            decision = controller.decide(state, shap_info)

            old_fitness = float(fitness_values[i])
            # Accion UNICA bifurcada: Rama A (reinit_random) o Rama B (reinit_guided).
            if budget.exhausted():
                break
            candidate = dispatch_rescue_single(
                decision["action"], positions, i, lb, ub,
                signals=signal_state, dominant_feature=decision["dominant_feature"],
                role_counts=role_counts, best_pos=best_pos, second_pos=second_pos,
                amplification_factor=controller.amplification_factor, rng=rng,
            )
            cand_fitness = float(eval_intervention(candidate))
            improved = cand_fitness < old_fitness - improvement_threshold(old_fitness)

            # El reinit se aplica SIEMPRE (sin gate greedy). El GBest se preserva
            # (solo sube). Reseteamos el reloj de estancamiento del agente (ventana
            # fresca del 10%); el action_cooldown (5%) espacia re-intervenciones.
            positions[i, :] = candidate
            fitness_values[i] = cand_fitness
            last_improve_fes[i] = budget.total
            if cand_fitness < pbest[i]:
                pbest[i] = cand_fitness
            if cand_fitness < best_score:
                best_score = cand_fitness
                best_pos = candidate.copy()
            elif cand_fitness < second_score:
                second_score = cand_fitness
                second_pos = candidate.copy()
            intervened_this_iter += 1

            controller.register_decision(
                budget.total, decision, i, old_fitness, cand_fitness,
                raw_div, fsi, accepted=True, improved=improved,
                acceptance_reason="applied_improved" if improved else "applied_worse",
            )

        if intervened_this_iter == 0 and top_blocked is not None:
            ai, areason, afsi, afit = top_blocked
            controller.register_decision(
                budget.total,
                {"intervene": False, "reason": areason, "action": "none",
                 "shap_info": None, "policy_signal": areason},
                ai, afit, afit, raw_div, afsi,
                accepted=False, improved=False, acceptance_reason="blocked",
            )

        if budget.exhausted():
            break

        # (e) Movimiento WO poblacional con las senales por FES.
        gbest_x = np.tile(best_pos, (n_agents, 1))
        positions = apply_wo_movement(
            positions, lb, ub, dim, n_agents,
            male_count, female_count, child_count,
            best_pos, second_pos, gbest_x,
            alpha, beta, R, danger_signal, safety_signal, rng=rng,
        )

    if not curve or curve[-1][0] != budget.total:
        curve.append((int(budget.total), float(best_score)))

    elapsed = time.perf_counter() - start
    declared = getattr(problem, "optimum", np.nan)
    optimum_f = (
        float(declared)
        if declared is not None and np.isfinite(float(declared))
        else np.nan
    )
    gap = float(best_score - optimum_f) if np.isfinite(optimum_f) else np.nan

    row = {
        "run_id": int(run_id),
        "seed": int(seed),
        "algorithm": "WO_shap",
        "controller_profile": controller.profile.name,
        "problem_spec": args.problem,
        "problem_family": getattr(problem, "family", "unknown"),
        "problem": problem_label,
        "dim": int(dim),
        "agents": int(n_agents),
        "max_fes": int(max_fes),
        "initial_fitness": float(initial_fitness),
        "final_fitness": float(best_score),
        "optimum": optimum_f,
        "gap_to_optimum": gap,
        "interventions": int(controller.intervention_count),
        "shap_explanations": int(len(controller.shap_rows)),
        "elapsed_seconds": float(elapsed),
        "best_position": " ".join(f"{v:.12g}" for v in best_pos),
        "init_mode": args.init_mode,
        "shapley_steps": int(controller.shapley_steps),
    }
    _actions = [event["action"] for event in controller.events]
    row["n_reinit_random"] = int(sum(a == "reinit_random" for a in _actions))
    row["n_reinit_guided"] = int(sum(a == "reinit_guided" for a in _actions))
    row.update(budget.as_dict())

    return {
        "row": row,
        "curve": np.asarray(curve, dtype=float),
        "events": controller.events_dataframe(),
        "non_events": controller.non_events_dataframe(),
        "shap_rows": controller.shap_dataframe(),
    }


def build_statistics(summary_df):
    rows = []
    for (problem_label, max_fes), data in summary_df.groupby(["problem", "max_fes"], sort=True):
        rows.append(
            {
                "problem": problem_label,
                "max_fes": int(max_fes),
                "problem_family": data.iloc[0]["problem_family"],
                "runs": int(data["run_id"].nunique()),
                "fitness_best": float(data["final_fitness"].min()),
                "fitness_worst": float(data["final_fitness"].max()),
                "fitness_mean": float(data["final_fitness"].mean()),
                "fitness_median": float(data["final_fitness"].median()),
                "fitness_std": float(data["final_fitness"].std(ddof=1)),
                "gap_mean": float(data["gap_to_optimum"].mean()),
                "gap_std": float(data["gap_to_optimum"].std(ddof=1)),
                "interventions_mean": float(data["interventions"].mean()),
                "shap_explanations_mean": float(data["shap_explanations"].mean()),
                "fes_shap_mean": float(data["fes_shap"].mean()),
                "time_mean_seconds": float(data["elapsed_seconds"].mean()),
            }
        )
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    max_fes_values = _parse_max_fes(args.max_fes)
    problems = _make_problems(args)
    output_dir = Path(args.output)
    values_dir = output_dir / "values"
    curves_dir = output_dir / "curves"
    values_dir.mkdir(parents=True, exist_ok=True)
    curves_dir.mkdir(parents=True, exist_ok=True)

    rows, all_events, all_non_events, all_shap = [], [], [], []
    if not args.no_exact_optimum:
        problems = [(with_exact_optimum(problem), label) for problem, label in problems]

    for max_fes_idx, max_fes in enumerate(max_fes_values):
        for problem_idx, (problem, label) in enumerate(problems):
            print(
                f"\n== {label} ({getattr(problem, 'family', 'unknown')}, "
                f"dim={problem.dim}) | MaxFES={max_fes} ==", flush=True,
            )
            for run_id in range(1, args.runs + 1):
                result = run_one(
                    problem, args, max_fes, max_fes_idx, run_id, problem_idx, label
                )
                rows.append(result["row"])
                curve_path = curves_dir / f"conv_curve_{label}_fes{max_fes}_run{run_id}.csv"
                np.savetxt(curve_path, result["curve"], delimiter=",",
                           header="fes,best_fitness", comments="")
                for bucket, store in (
                    ("events", all_events), ("non_events", all_non_events),
                    ("shap_rows", all_shap),
                ):
                    df = result[bucket]
                    if not df.empty:
                        df = df.copy()
                        df.insert(0, "max_fes", max_fes)
                        df.insert(0, "problem", label)
                        df.insert(0, "run_id", run_id)
                        store.append(df)
                r = result["row"]
                print(
                    f"  run={run_id}/{args.runs} seed={r['seed']} "
                    f"final={r['final_fitness']:.6g} gap={r['gap_to_optimum']:.6g} "
                    f"interv={r['interventions']} fes_shap={r['fes_shap']} "
                    f"fes_total={r['fes_total']}/{r['max_fes']} t={r['elapsed_seconds']:.2f}s",
                    flush=True,
                )

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(values_dir / "summary.csv", index=False)
    build_statistics(summary_df).to_csv(values_dir / "statistics.csv", index=False)
    if all_events:
        pd.concat(all_events, ignore_index=True).to_csv(
            values_dir / "controller_events.csv", index=False)
    if all_non_events:
        pd.concat(all_non_events, ignore_index=True).to_csv(
            values_dir / "controller_non_events.csv", index=False)
    if all_shap:
        pd.concat(all_shap, ignore_index=True).to_csv(
            values_dir / "shap_values.csv", index=False)

    print(f"\nResumen: {values_dir / 'summary.csv'}")
    print(f"Estadisticas: {values_dir / 'statistics.csv'}")


if __name__ == "__main__":
    main()
