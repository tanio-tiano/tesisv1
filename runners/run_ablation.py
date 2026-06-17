"""Runner unificado WO base + WO+SHAP en regimen MaxFES.

Reemplaza a los antiguos `run_wo_base.py` y `run_wo_shap.py`. En una sola
invocacion corre los modos pedidos por `--modes` (subconjunto de `{base, shap}`)
con SEEDS PAREADAS: para un mismo `(max_fes, problem, run_id)`, base y shap
arrancan de la misma semilla y por lo tanto de la misma poblacion inicial. Esto
permite analisis pareado Wilcoxon directo sin re-mapeo.

Estructura de salida (compatible con `analysis/report_html.py`):

    <output>/
    |-- base/                 # solo si 'base' en --modes
    |   |-- values/
    |   |   |-- summary.csv
    |   |   `-- statistics.csv
    |   `-- curves/
    |       `-- conv_curve_<label>_fes<MaxFES>_run<N>.csv
    `-- shap/                 # solo si 'shap' en --modes
        |-- values/
        |   |-- summary.csv
        |   |-- statistics.csv
        |   |-- controller_events.csv
        |   |-- controller_non_events.csv
        |   `-- shap_values.csv
        `-- curves/
            `-- conv_curve_<label>_fes<MaxFES>_run<N>.csv

Telemetria adicional vs runners anteriores:

- `t_init_seconds`: tiempo de wall-clock gastado en la inicializacion de la
  poblacion (relevante en TMLAP con `local_search`).
- `t_shap_seconds` (solo shap): tiempo total dentro de la maquinaria SHAP
  (`make_value_function_for_agent` + `controller.explain_fitness`). NO incluye
  `controller.decide` ni la evaluacion del candidato de intervencion (esos
  viven en `elapsed_seconds`/`fes_intervention`).

Uso:

    python -m runners.run_ablation --problem cec2022:all --dim 10 --agents 30 \\
        --max-fes 5000 --runs 51 --modes base,shap \\
        --output experiments/cec2022_d10_fes5000

    python -m runners.run_ablation --problem tmlap:3.instancia_dura.txt \\
        --agents 30 --max-fes 50000 --runs 51 --modes base,shap \\
        --init-mode random --no-exact-optimum \\
        --output experiments/ablation_b4_dura_51
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
from shap_controller.profiles import STAGNATION_WINDOW_FRACTION
from wo_core.agent_sim import make_value_function_for_agent
from wo_core.diversity import population_diversity
from wo_core.fes import FESBudget, counting_objective
from wo_core.walrus import (
    apply_wo_movement,
    evaluate_and_update_leaders,
    iteration_signals,
    walrus_role_counts,
)

VALID_MODES = ("base", "shap")

# Compuertas globales del controlador SHAP: bloquean a TODOS los agentes, asi
# que cortan el escaneo de la iteracion. Las demas razones son por-agente y
# permiten probar el siguiente.
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
        description="Runner unificado WO base + WO+SHAP en regimen MaxFES.",
    )
    parser.add_argument("--problem", required=True)
    parser.add_argument("--dim", type=int, default=10)
    parser.add_argument("--clients", type=int, default=None)
    parser.add_argument("--hubs", type=int, default=None)
    parser.add_argument("--agents", type=int, default=30)
    parser.add_argument(
        "--max-fes", type=str, default="50000",
        help="Presupuesto(s) MaxFES separados por coma. Ej: 5000,50000,500000,5000000",
    )
    parser.add_argument("--runs", type=int, default=51)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--modes", type=str, default="base,shap",
        help="Modos a correr separados por coma. Validos: base,shap.",
    )
    parser.add_argument(
        "--init-mode", choices=["local_search", "random"], default="local_search",
        help="Solo TMLAP: 'random' salta local_search (instancias grandes).",
    )
    parser.add_argument(
        "--no-exact-optimum", action="store_true",
        help="No calcular el optimo exacto (instancias TMLAP intratables): gap=NaN.",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _parse_max_fes(text):
    values = [int(float(token.strip())) for token in str(text).split(",") if token.strip()]
    if not values:
        raise ValueError(f"--max-fes invalido: {text!r}")
    return values


def _parse_modes(text):
    modes = [token.strip().lower() for token in str(text).split(",") if token.strip()]
    if not modes:
        raise ValueError(f"--modes invalido: {text!r}")
    for mode in modes:
        if mode not in VALID_MODES:
            raise ValueError(
                f"Modo desconocido: {mode!r}. Validos: {VALID_MODES}."
            )
    seen = set()
    deduped = []
    for mode in modes:
        if mode not in seen:
            seen.add(mode)
            deduped.append(mode)
    return deduped


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
    """Semilla del run; INDEPENDIENTE del modo -> base y shap quedan pareados."""
    return int(args.seed + max_fes_idx * 100000 + problem_idx * 1000 + run_idx)


def _initial_population(problem, args, n_agents, rng, budget):
    """Construye la poblacion inicial y mide su wall-clock (`t_init_seconds`)."""
    t0 = time.perf_counter()
    if getattr(problem, "family", None) == "tmlap":
        positions = problem.initial_population(
            n_agents, rng, init_mode=args.init_mode,
            on_eval=lambda: budget.spend("init", 1),
        )
    else:
        positions = problem.initial_population(n_agents, rng)
    return positions, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Modo BASE: WO sin controlador.
# ---------------------------------------------------------------------------
def run_one_base(problem, args, max_fes, max_fes_idx, run_id, problem_idx, problem_label):
    seed = _seed_for(args, max_fes_idx, problem_idx, run_id - 1)
    rng = np.random.default_rng(seed)
    start = time.perf_counter()

    n_agents = args.agents
    dim, lb, ub = problem.dim, problem.lb, problem.ub
    budget = FESBudget(max_fes)
    eval_search = counting_objective(problem.evaluate, budget, "search")

    positions, t_init_seconds = _initial_population(problem, args, n_agents, rng, budget)

    best_pos = np.zeros(dim, dtype=float)
    second_pos = np.zeros(dim, dtype=float)
    best_score = float("inf")
    second_score = float("inf")
    male_count, female_count, child_count = walrus_role_counts(n_agents)
    curve = []
    initial_fitness = np.nan

    # Deteccion de estancamiento por agente (SIN accion: solo telemetria).
    # Ventana fija al 10% MaxFES (igual que el modo shap) para comparacion pareada.
    stagnation_window = max(1, int(round(STAGNATION_WINDOW_FRACTION * max_fes)))
    pbest = np.full(n_agents, np.inf, dtype=float)
    last_improve_fes = np.zeros(n_agents, dtype=np.int64)

    while not budget.exhausted():
        fes_start = budget.total
        (
            positions, fitness_values, best_score, best_pos, second_score, second_pos
        ) = evaluate_and_update_leaders(
            positions, lb, ub, eval_search,
            best_score, best_pos, second_score, second_pos, budget=budget,
        )
        if not np.isfinite(initial_fitness):
            initial_fitness = float(best_score)

        # Actualizacion read-only del pbest/last_improve por agente. NO gasta FES
        # (los fitness ya fueron computados arriba); solo lee budget.total como reloj.
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

        gbest_x = np.tile(best_pos, (n_agents, 1))
        alpha, beta, A, R, danger_signal, safety_signal = iteration_signals(
            fes_start, max_fes, rng
        )
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

    fes_since_final = budget.total - last_improve_fes
    n_stagnant_at_end = int(np.sum(fes_since_final >= stagnation_window))
    mean_fes_since = float(np.mean(fes_since_final))
    max_fes_since = int(np.max(fes_since_final))

    row = {
        "run_id": int(run_id),
        "seed": int(seed),
        "algorithm": "WO_base",
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
        "elapsed_seconds": float(elapsed),
        "t_init_seconds": float(t_init_seconds),
        "t_shap_seconds": float("nan"),
        "stagnation_window": int(stagnation_window),
        "n_stagnant_at_end": int(n_stagnant_at_end),
        "mean_fes_since_improve_at_end": float(mean_fes_since),
        "max_fes_since_improve_at_end": int(max_fes_since),
        "best_position": " ".join(f"{v:.12g}" for v in best_pos),
        "init_mode": args.init_mode,
    }
    row.update(budget.as_dict())
    return {"row": row, "curve": np.asarray(curve, dtype=float)}


# ---------------------------------------------------------------------------
# Modo SHAP: WO + controlador SHAP por agente.
# ---------------------------------------------------------------------------
def run_one_shap(problem, args, max_fes, max_fes_idx, run_id, problem_idx, problem_label):
    seed = _seed_for(args, max_fes_idx, problem_idx, run_id - 1)
    rng = np.random.default_rng(seed)
    start = time.perf_counter()

    n_agents = args.agents
    dim, lb, ub = problem.dim, problem.lb, problem.ub
    budget = FESBudget(max_fes)
    eval_search = counting_objective(problem.evaluate, budget, "search")
    eval_shap = counting_objective(problem.evaluate, budget, "shap")
    eval_intervention = counting_objective(problem.evaluate, budget, "intervention")

    positions, t_init_seconds = _initial_population(problem, args, n_agents, rng, budget)

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
    curve = []
    initial_fitness = np.nan
    t_shap_seconds = 0.0

    while not budget.exhausted():
        fes_start = budget.total

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
                break

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
                continue

            t_shap_start = time.perf_counter()
            value_fn = make_value_function_for_agent(
                eval_shap, i, positions, lb, ub, dim, role_counts,
                best_pos, second_pos, signal_state, max_fes, fes_start,
                controller.shapley_steps, rng=rng,
            )
            shap_info = controller.explain_fitness(state, value_function=value_fn)
            t_shap_seconds += time.perf_counter() - t_shap_start
            decision = controller.decide(state, shap_info)

            old_fitness = float(fitness_values[i])
            if budget.exhausted():
                break
            candidate = dispatch_rescue_single(
                decision["action"], positions, i, lb, ub,
                signals=signal_state, dominant_feature=decision["dominant_feature"],
                role_counts=role_counts, best_pos=best_pos, second_pos=second_pos,
                amplification_factor=controller.amplification_factor, rng=rng,
                dominant_value=decision.get("dominant_value", 0.0),
            )
            cand_fitness = float(eval_intervention(candidate))
            improved = cand_fitness < old_fitness - improvement_threshold(old_fitness)

            positions[i, :] = candidate
            fitness_values[i] = cand_fitness
            # FIX: el reloj de estancamiento se anclaba a "ultima INTERVENCION"
            # (resetear siempre). Ahora se ancla a "ultima MEJORA real" (consistente
            # con Informe_Metodologia.md §4): solo resetear si la intervencion mejoro.
            if improved:
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

    # Telemetria de estancamiento (simetrica con el modo base). El bloque shap
    # ya mantiene pbest/last_improve_fes; aqui solo calculamos snapshot final.
    fes_since_final = budget.total - last_improve_fes
    n_stagnant_at_end = int(np.sum(fes_since_final >= window))
    mean_fes_since = float(np.mean(fes_since_final))
    max_fes_since = int(np.max(fes_since_final))

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
        "t_init_seconds": float(t_init_seconds),
        "t_shap_seconds": float(t_shap_seconds),
        "stagnation_window": int(window),
        "n_stagnant_at_end": int(n_stagnant_at_end),
        "mean_fes_since_improve_at_end": float(mean_fes_since),
        "max_fes_since_improve_at_end": int(max_fes_since),
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


def run_one(mode, problem, args, max_fes, max_fes_idx, run_id, problem_idx, problem_label):
    if mode == "base":
        return run_one_base(
            problem, args, max_fes, max_fes_idx, run_id, problem_idx, problem_label,
        )
    if mode == "shap":
        return run_one_shap(
            problem, args, max_fes, max_fes_idx, run_id, problem_idx, problem_label,
        )
    raise ValueError(f"Modo desconocido: {mode!r}.")


def build_statistics(summary_df, mode):
    rows = []
    has_shap_cols = mode == "shap"
    for (problem_label, max_fes), data in summary_df.groupby(["problem", "max_fes"], sort=True):
        entry = {
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
            "time_mean_seconds": float(data["elapsed_seconds"].mean()),
            "t_init_mean_seconds": float(data["t_init_seconds"].mean()),
            "n_stagnant_at_end_mean": float(data["n_stagnant_at_end"].mean()),
            "max_fes_since_improve_at_end_mean": float(
                data["max_fes_since_improve_at_end"].mean()
            ),
        }
        if has_shap_cols:
            entry["t_shap_mean_seconds"] = float(data["t_shap_seconds"].mean())
            entry["interventions_mean"] = float(data["interventions"].mean())
            entry["shap_explanations_mean"] = float(data["shap_explanations"].mean())
            entry["fes_shap_mean"] = float(data["fes_shap"].mean())
        rows.append(entry)
    return pd.DataFrame(rows)


def _ensure_mode_dirs(output_dir, mode):
    base_dir = output_dir / mode
    values_dir = base_dir / "values"
    curves_dir = base_dir / "curves"
    values_dir.mkdir(parents=True, exist_ok=True)
    curves_dir.mkdir(parents=True, exist_ok=True)
    return values_dir, curves_dir


def main():
    args = parse_args()
    modes = _parse_modes(args.modes)
    max_fes_values = _parse_max_fes(args.max_fes)
    problems = _make_problems(args)
    output_dir = Path(args.output)

    if not args.no_exact_optimum:
        problems = [(with_exact_optimum(problem), label) for problem, label in problems]

    for mode in modes:
        values_dir, curves_dir = _ensure_mode_dirs(output_dir, mode)
        rows, all_events, all_non_events, all_shap = [], [], [], []
        print(f"\n##### MODO {mode.upper()} #####", flush=True)

        for max_fes_idx, max_fes in enumerate(max_fes_values):
            for problem_idx, (problem, label) in enumerate(problems):
                print(
                    f"\n== [{mode}] {label} ({getattr(problem, 'family', 'unknown')}, "
                    f"dim={problem.dim}) | MaxFES={max_fes} ==",
                    flush=True,
                )
                for run_id in range(1, args.runs + 1):
                    result = run_one(
                        mode, problem, args, max_fes, max_fes_idx,
                        run_id, problem_idx, label,
                    )
                    rows.append(result["row"])
                    curve_path = curves_dir / f"conv_curve_{label}_fes{max_fes}_run{run_id}.csv"
                    np.savetxt(
                        curve_path, result["curve"], delimiter=",",
                        header="fes,best_fitness", comments="",
                    )
                    if mode == "shap":
                        for bucket, store in (
                            ("events", all_events),
                            ("non_events", all_non_events),
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
                    if mode == "base":
                        print(
                            f"  run={run_id}/{args.runs} seed={r['seed']} "
                            f"final={r['final_fitness']:.6g} gap={r['gap_to_optimum']:.6g} "
                            f"fes_total={r['fes_total']}/{r['max_fes']} "
                            f"t_init={r['t_init_seconds']:.3f}s t={r['elapsed_seconds']:.2f}s",
                            flush=True,
                        )
                    else:
                        print(
                            f"  run={run_id}/{args.runs} seed={r['seed']} "
                            f"final={r['final_fitness']:.6g} gap={r['gap_to_optimum']:.6g} "
                            f"interv={r['interventions']} fes_shap={r['fes_shap']} "
                            f"t_shap={r['t_shap_seconds']:.3f}s "
                            f"fes_total={r['fes_total']}/{r['max_fes']} "
                            f"t={r['elapsed_seconds']:.2f}s",
                            flush=True,
                        )

        summary_df = pd.DataFrame(rows)
        summary_df.to_csv(values_dir / "summary.csv", index=False)
        build_statistics(summary_df, mode).to_csv(
            values_dir / "statistics.csv", index=False,
        )
        if mode == "shap":
            if all_events:
                pd.concat(all_events, ignore_index=True).to_csv(
                    values_dir / "controller_events.csv", index=False,
                )
            if all_non_events:
                pd.concat(all_non_events, ignore_index=True).to_csv(
                    values_dir / "controller_non_events.csv", index=False,
                )
            if all_shap:
                pd.concat(all_shap, ignore_index=True).to_csv(
                    values_dir / "shap_values.csv", index=False,
                )

        print(f"\n[{mode}] Resumen: {values_dir / 'summary.csv'}")
        print(f"[{mode}] Estadisticas: {values_dir / 'statistics.csv'}")


if __name__ == "__main__":
    main()
