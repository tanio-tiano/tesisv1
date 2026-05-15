"""Runner WO con controlador SHAP on-line, parametrizable por problema.

Uso:

    python -m runners.run_wo_shap \\
        --problem cec2022:F6 \\
        --runs 30 --agents 30 --iterations 500 --seed 1234 --profile soft \\
        --output experiments/cec_F6_shap_30runs

    python -m runners.run_wo_shap \\
        --problem tmlap:1.instancia_simple.txt \\
        --runs 30 --agents 30 --iterations 300 --seed 1234 --profile soft \\
        --output experiments/tmlap_simple_shap_30runs

Logica:

1. Inicializa la poblacion segun el problema (uniforme para CEC,
   repair[+local_search] para TMLAP).
2. En cada iteracion calcula las 6 features de SHAP (alpha, beta, danger,
   safety, diversity, iteration normalizada).
3. Llama al controlador (``SHAPFitnessController.decide``) usando una
   ``value_function`` que el problema provee para Shapley exacto.
4. Si la decision es intervenir, aplica ``partial_restart`` o
   ``random_reinjection`` sobre una copia y acepta con gate
   fitness/diversidad (compuerta Section 4 del informe).
5. Aplica el movimiento WO estandar y avanza.
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
from shap_controller import (
    PROFILE_DEFAULTS,
    SHAPFitnessController,
    dispatch_rescue,
    get_controller_profile,
    improvement_threshold,
)
from wo_core.diversity import (
    domain_diversity_scale,
    normalize_diversity_by_domain,
    population_diversity,
)
from wo_core.walrus import (
    apply_wo_movement,
    evaluate_and_update_leaders,
    iteration_signals,
    walrus_role_counts,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ejecuta WO con controlador SHAP on-line sobre CEC o TMLAP."
    )
    parser.add_argument("--problem", required=True)
    parser.add_argument("--dim", type=int, default=10)
    parser.add_argument("--clients", type=int, default=None)
    parser.add_argument("--hubs", type=int, default=None)
    parser.add_argument("--agents", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--profile", choices=sorted(PROFILE_DEFAULTS), default="soft"
    )
    parser.add_argument(
        "--init-mode",
        choices=["local_search", "random"],
        default="local_search",
        help="Solo TMLAP: 'random' salta local_search en init (instancias grandes).",
    )
    parser.add_argument(
        "--shapley-steps",
        type=int,
        default=3,
        help="Pasos de simulacion WO por cada coalicion Shapley (default 3).",
    )
    parser.add_argument(
        "--acceptance-mode",
        choices=["diversity", "strict"],
        default="diversity",
        help="Gate de aceptacion: 'diversity' = mejorar best O preservar+diversificar.",
    )
    parser.add_argument("--rescue-scale", type=float, default=1.0)
    parser.add_argument(
        "--neutral-cooldown-multiplier", type=float, default=1.5,
        help="Multiplica el cooldown si la intervencion quedo neutral en t+10.",
    )
    parser.add_argument(
        "--rejected-cooldown-multiplier", type=float, default=2.5,
        help="Multiplica el cooldown si la intervencion fue rechazada por el gate.",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


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


def _seed_for(args, problem_idx, run_idx):
    return int(args.seed + problem_idx * 1000 + run_idx)


def _conditional_acceptance(
    current_positions,
    current_fitness,
    candidate_positions,
    candidate_fitness,
    acceptance_mode,
):
    current_best = float(np.min(current_fitness))
    candidate_best = float(np.min(candidate_fitness))
    current_div = population_diversity(current_positions)
    candidate_div = population_diversity(candidate_positions)
    tol = improvement_threshold(current_best)

    if candidate_best < current_best - tol:
        return {
            "accepted": True,
            "reason": "improved_population_best",
            "current_population_best": current_best,
            "candidate_population_best": candidate_best,
            "current_diversity": current_div,
            "candidate_diversity": candidate_div,
        }
    if acceptance_mode == "strict":
        return {
            "accepted": False,
            "reason": "rejected_by_strict",
            "current_population_best": current_best,
            "candidate_population_best": candidate_best,
            "current_diversity": current_div,
            "candidate_diversity": candidate_div,
        }
    if candidate_best <= current_best + tol and candidate_div > current_div + 1e-12:
        return {
            "accepted": True,
            "reason": "preserved_best_and_increased_diversity",
            "current_population_best": current_best,
            "candidate_population_best": candidate_best,
            "current_diversity": current_div,
            "candidate_diversity": candidate_div,
        }
    return {
        "accepted": False,
        "reason": "rejected_by_conditional_acceptance",
        "current_population_best": current_best,
        "candidate_population_best": candidate_best,
        "current_diversity": current_div,
        "candidate_diversity": candidate_div,
    }


def run_one(problem, args, run_id, problem_idx, problem_label):
    seed = _seed_for(args, problem_idx, run_id - 1)
    rng = np.random.default_rng(seed)
    start = time.perf_counter()

    n_agents = args.agents
    dim = problem.dim
    lb = problem.lb
    ub = problem.ub

    if hasattr(problem, "family") and problem.family == "tmlap":
        positions = problem.initial_population(n_agents, rng, init_mode=args.init_mode)
    else:
        positions = problem.initial_population(n_agents, rng)

    best_pos = np.zeros(dim, dtype=float)
    second_pos = np.zeros(dim, dtype=float)
    best_score = float("inf")
    second_score = float("inf")
    male_count, female_count, child_count = walrus_role_counts(n_agents)
    curve = np.zeros(args.iterations, dtype=float)

    profile = get_controller_profile(args.profile)
    controller = SHAPFitnessController(
        max_iter=args.iterations,
        random_state=seed,
        profile=profile,
        rescue_scale=args.rescue_scale,
        neutral_cooldown_multiplier=args.neutral_cooldown_multiplier,
        rejected_cooldown_multiplier=args.rejected_cooldown_multiplier,
    )

    last_improvement_score = float("inf")
    stagnation_length = 0
    best_history = []
    diversity_ref = None

    for iteration in range(args.iterations):
        (
            positions,
            fitness_values,
            best_score,
            best_pos,
            second_score,
            second_pos,
        ) = evaluate_and_update_leaders(
            positions, lb, ub, problem.evaluate,
            best_score, best_pos, second_score, second_pos,
        )
        gbest_x = np.tile(best_pos, (n_agents, 1))
        if not np.isfinite(second_score):
            second_pos = best_pos.copy()

        # Estancamiento.
        if not np.isfinite(last_improvement_score):
            last_improvement_score = best_score
            stagnation_length = 0
        else:
            threshold = improvement_threshold(last_improvement_score)
            if best_score < last_improvement_score - threshold:
                last_improvement_score = best_score
                stagnation_length = 0
            else:
                stagnation_length += 1
        best_history.append(float(best_score))

        # Recent improvement ratio (ventana segun profile).
        lookback = min(profile.recent_improvement_window, max(0, len(best_history) - 1))
        if lookback > 0:
            ref = float(best_history[-lookback - 1])
            recent_improvement = max(0.0, ref - float(best_score))
            recent_improvement_ratio = recent_improvement / max(abs(ref), 1.0)
        else:
            recent_improvement_ratio = 0.0

        # Senales SHAP.
        raw_diversity = population_diversity(positions)
        if diversity_ref is None:
            diversity_ref = max(raw_diversity, 1e-12)
        domain_scale = domain_diversity_scale(lb, ub)
        diversity_norm = raw_diversity / (diversity_ref + 1e-12)
        diversity_normalized_domain = raw_diversity / max(domain_scale, 1e-12)

        alpha, beta, r_signal, danger, safety = iteration_signals(
            iteration, args.iterations, rng
        )
        state = {
            "alpha": float(alpha),
            "beta": float(beta),
            "danger_signal": float(danger),
            "safety_signal": float(safety),
            "diversity": float(diversity_normalized_domain),
            "raw_diversity": float(raw_diversity),
            "diversity_domain_scale": float(domain_scale),
            "diversity_norm": float(diversity_norm),
            "iteration": int(iteration),
            "stagnation_length": int(stagnation_length),
            "recent_improvement_ratio": float(recent_improvement_ratio),
        }
        controller.record_state(state, best_score)
        controller.update_pending_events(iteration, best_score, raw_diversity, stagnation_length)

        # Decision (puede o no llamar a SHAP segun gate previo).
        should_explain, _ = controller.should_consider_intervention(state)
        shap_info = None
        if should_explain:
            value_function = problem.make_value_function_for_shapley(
                state, positions, args.iterations, args.shapley_steps, rng=rng,
            )
            shap_info = controller.explain_fitness(state, value_function=value_function)
        decision = controller.decide(state, shap_info=shap_info)

        accepted = False
        acceptance = {
            "accepted": False,
            "reason": "not_proposed",
            "current_population_best": np.nan,
            "candidate_population_best": np.nan,
            "current_diversity": float(raw_diversity),
            "candidate_diversity": np.nan,
        }
        selected_indices = np.array([], dtype=int)

        if decision["intervene"]:
            candidate_positions, selected_indices = dispatch_rescue(
                action=decision["action"],
                mode=decision["mode"],
                positions=positions,
                fitness_values=fitness_values,
                lb=lb, ub=ub,
                best_pos=best_pos, second_pos=second_pos,
                fraction=decision["fraction"],
                rng=rng,
            )
            candidate_fitness = np.array(
                [problem.evaluate(candidate_positions[i]) for i in range(n_agents)],
                dtype=float,
            )
            acceptance = _conditional_acceptance(
                positions, fitness_values, candidate_positions, candidate_fitness,
                acceptance_mode=args.acceptance_mode,
            )
            accepted = bool(acceptance["accepted"])
            if accepted:
                positions = candidate_positions
                fitness_values = candidate_fitness
                # Reflejar nuevos lideres tras el rescate.
                order = np.argsort(fitness_values)
                if fitness_values[order[0]] < best_score:
                    best_score = float(fitness_values[order[0]])
                    best_pos = positions[order[0], :].copy()
                if len(order) > 1 and fitness_values[order[1]] < second_score:
                    second_score = float(fitness_values[order[1]])
                    second_pos = positions[order[1], :].copy()
                gbest_x = np.tile(best_pos, (n_agents, 1))

        controller.register_decision(
            iteration, decision,
            pre_fitness=best_score, diversity=raw_diversity,
            stagnation_length=stagnation_length,
            indices=selected_indices,
            accepted=accepted,
            acceptance_reason=acceptance["reason"],
            current_population_best=acceptance["current_population_best"],
            candidate_population_best=acceptance["candidate_population_best"],
            candidate_diversity=acceptance["candidate_diversity"],
        )

        # Movimiento WO normal.
        positions = apply_wo_movement(
            positions, lb, ub, dim, n_agents,
            male_count, female_count, child_count,
            best_pos, second_pos, gbest_x,
            alpha, beta, r_signal, danger, safety, rng=rng,
        )
        curve[iteration] = best_score

    elapsed = time.perf_counter() - start
    optimum = getattr(problem, "optimum", np.nan)
    optimum_f = float(optimum) if optimum is not None and np.isfinite(float(optimum)) else np.nan
    gap = float(best_score - optimum_f) if np.isfinite(optimum_f) else np.nan

    return {
        "row": {
            "run_id": int(run_id),
            "seed": int(seed),
            "algorithm": "WO_shap",
            "controller_profile": profile.name,
            "problem_spec": args.problem,
            "problem_family": getattr(problem, "family", "unknown"),
            "problem": problem_label,
            "dim": int(dim),
            "agents": int(n_agents),
            "iterations": int(args.iterations),
            "initial_fitness": float(curve[0]),
            "final_fitness": float(best_score),
            "optimum": optimum_f,
            "gap_to_optimum": gap,
            "interventions": int(controller.intervention_count),
            "shap_explanations": int(len(controller.shap_rows)),
            "elapsed_seconds": float(elapsed),
            "best_position": " ".join(f"{v:.12g}" for v in best_pos),
            "init_mode": args.init_mode,
            "acceptance_mode": args.acceptance_mode,
            "shapley_steps": int(args.shapley_steps),
        },
        "curve": curve,
        "events": controller.events_dataframe(),
        "shap_rows": controller.shap_dataframe(),
    }


def build_statistics(summary_df):
    rows = []
    for problem_label, data in summary_df.groupby("problem", sort=True):
        rows.append(
            {
                "problem": problem_label,
                "problem_family": data.iloc[0]["problem_family"],
                "runs": int(data["run_id"].nunique()),
                "fitness_best": float(data["final_fitness"].min()),
                "fitness_worst": float(data["final_fitness"].max()),
                "fitness_mean": float(data["final_fitness"].mean()),
                "fitness_median": float(data["final_fitness"].median()),
                "fitness_std": float(data["final_fitness"].std(ddof=1)),
                "gap_best": float(data["gap_to_optimum"].min()),
                "gap_worst": float(data["gap_to_optimum"].max()),
                "gap_mean": float(data["gap_to_optimum"].mean()),
                "gap_std": float(data["gap_to_optimum"].std(ddof=1)),
                "interventions_mean": float(data["interventions"].mean()),
                "interventions_total": int(data["interventions"].sum()),
                "time_mean_seconds": float(data["elapsed_seconds"].mean()),
            }
        )
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    problems = _make_problems(args)
    output_dir = Path(args.output)
    values_dir = output_dir / "values"
    curves_dir = output_dir / "curves"
    values_dir.mkdir(parents=True, exist_ok=True)
    curves_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    all_events = []
    all_shap = []
    for problem_idx, (problem, label) in enumerate(problems):
        print(f"\n== {label} ({getattr(problem, 'family', 'unknown')}, dim={problem.dim}) ==",
              flush=True)
        for run_id in range(1, args.runs + 1):
            result = run_one(problem, args, run_id, problem_idx, label)
            rows.append(result["row"])
            curve_path = curves_dir / f"conv_curve_{label}_run{run_id}.csv"
            np.savetxt(curve_path, result["curve"], delimiter=",",
                       header="best_fitness", comments="")
            if not result["events"].empty:
                ev = result["events"].copy()
                ev.insert(0, "problem", label)
                ev.insert(0, "run_id", run_id)
                all_events.append(ev)
            if not result["shap_rows"].empty:
                sr = result["shap_rows"].copy()
                sr.insert(0, "problem", label)
                sr.insert(0, "run_id", run_id)
                all_shap.append(sr)
            row = result["row"]
            print(
                f"  run={run_id}/{args.runs} seed={row['seed']} "
                f"final={row['final_fitness']:.6g} gap={row['gap_to_optimum']:.6g} "
                f"interventions={row['interventions']} time={row['elapsed_seconds']:.2f}s",
                flush=True,
            )

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(values_dir / "summary.csv", index=False)
    stats_df = build_statistics(summary_df)
    stats_df.to_csv(values_dir / "statistics.csv", index=False)

    if all_events:
        events_df = pd.concat(all_events, ignore_index=True)
        events_df.to_csv(values_dir / "controller_events.csv", index=False)
    if all_shap:
        shap_df = pd.concat(all_shap, ignore_index=True)
        shap_df.to_csv(values_dir / "shap_values.csv", index=False)

    print(f"\nResumen: {values_dir / 'summary.csv'}")
    print(f"Estadisticas: {values_dir / 'statistics.csv'}")
    if all_events:
        print(f"Eventos del controlador: {values_dir / 'controller_events.csv'}")
    if all_shap:
        print(f"Valores SHAP: {values_dir / 'shap_values.csv'}")


if __name__ == "__main__":
    main()
