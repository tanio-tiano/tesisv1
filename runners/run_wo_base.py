"""Runner WO base (sin controlador) en regimen **MaxFES**.

Baseline justo para comparar contra ``run_wo_shap``: misma dinamica WO, mismo
schedule por FES (``phi = fes/MaxFES``), mismo criterio de parada (MaxFES) y la
misma contabilidad de FES (incluida la inicializacion de TMLAP).

Uso:

    python -m runners.run_wo_base --problem cec2022:F6 --dim 10 --agents 30 \\
        --max-fes 5000,50000,500000,5000000 --runs 51 --output experiments/cec_F6_base

    python -m runners.run_wo_base --problem tmlap:1.instancia_simple.txt \\
        --agents 30 --max-fes 50000 --runs 51 --output experiments/tmlap_simple_base
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
from wo_core.fes import FESBudget, counting_objective
from wo_core.walrus import (
    apply_wo_movement,
    evaluate_and_update_leaders,
    iteration_signals,
    walrus_role_counts,
)


def parse_args():
    parser = argparse.ArgumentParser(description="WO base en regimen MaxFES.")
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
    return int(args.seed + max_fes_idx * 100000 + problem_idx * 1000 + run_idx)


def run_one(problem, args, max_fes, max_fes_idx, run_id, problem_idx, problem_label):
    seed = _seed_for(args, max_fes_idx, problem_idx, run_id - 1)
    rng = np.random.default_rng(seed)
    start = time.perf_counter()

    n_agents = args.agents
    dim, lb, ub = problem.dim, problem.lb, problem.ub
    budget = FESBudget(max_fes)
    eval_search = counting_objective(problem.evaluate, budget, "search")

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
    curve = []
    initial_fitness = np.nan

    while not budget.exhausted():
        fes_start = budget.total
        (
            positions, _fit, best_score, best_pos, second_score, second_pos
        ) = evaluate_and_update_leaders(
            positions, lb, ub, eval_search,
            best_score, best_pos, second_score, second_pos, budget=budget,
        )
        if not np.isfinite(initial_fitness):
            initial_fitness = float(best_score)
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
        "best_position": " ".join(f"{v:.12g}" for v in best_pos),
        "init_mode": args.init_mode,
    }
    row.update(budget.as_dict())
    return {"row": row, "curve": np.asarray(curve, dtype=float)}


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

    rows = []
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
                r = result["row"]
                print(
                    f"  run={run_id}/{args.runs} seed={r['seed']} "
                    f"final={r['final_fitness']:.6g} gap={r['gap_to_optimum']:.6g} "
                    f"fes_total={r['fes_total']}/{r['max_fes']} t={r['elapsed_seconds']:.2f}s",
                    flush=True,
                )

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(values_dir / "summary.csv", index=False)
    build_statistics(summary_df).to_csv(values_dir / "statistics.csv", index=False)
    print(f"\nResumen: {values_dir / 'summary.csv'}")
    print(f"Estadisticas: {values_dir / 'statistics.csv'}")


if __name__ == "__main__":
    main()
