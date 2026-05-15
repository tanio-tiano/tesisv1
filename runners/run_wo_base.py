"""Runner WO base (sin controlador), parametrizable por problema.

Uso:

    python -m runners.run_wo_base \\
        --problem cec2022:F6 \\
        --runs 30 --agents 30 --iterations 500 --seed 1234 \\
        --output experiments/cec_F6_base_30runs

    python -m runners.run_wo_base \\
        --problem tmlap:1.instancia_simple.txt \\
        --runs 30 --agents 30 --iterations 300 --seed 1234 \\
        --output experiments/tmlap_simple_base_30runs

El runner usa la misma dinamica WO definida en ``wo_core.walrus`` y deja la
parte especifica del problema al adaptador en ``problems/``.

Para CEC se itera sobre F1..F12 si se omite el sufijo ``:Fk``, replicando el
comportamiento de un benchmark.
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

from problems.cec2022 import FUNCTION_IDS as CEC_FUNCTION_IDS, get_function_metadata
from problems.factory import parse_problem_spec
from wo_core.walrus import (
    apply_wo_movement,
    evaluate_and_update_leaders,
    iteration_signals,
    walrus_role_counts,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ejecuta WO base sobre un problema CEC o TMLAP."
    )
    parser.add_argument(
        "--problem",
        required=True,
        help=(
            "Spec del problema: 'cec2022:F6', 'cec2022:all' (F1..F12), "
            "'tmlap:1.instancia_simple.txt', etc."
        ),
    )
    parser.add_argument("--dim", type=int, default=10, help="Dimension para CEC.")
    parser.add_argument(
        "--clients",
        type=int,
        default=None,
        help="Override de n_clientes para subinstancias TMLAP.",
    )
    parser.add_argument(
        "--hubs",
        type=int,
        default=None,
        help="Override de n_hubs para subinstancias TMLAP.",
    )
    parser.add_argument("--agents", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--init-mode",
        choices=["local_search", "random"],
        default="local_search",
        help=(
            "Solo aplica a TMLAP: 'random' salta local_search en init "
            "(necesario para instancias grandes). CEC ignora este flag."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Carpeta de salida. Se crean subcarpetas values/ y curves/.",
    )
    return parser.parse_args()


def _make_problems(args):
    """Resuelve la spec a una lista de (problem, identifier) que el runner iterara.

    Para 'cec2022:all' devuelve F1..F12; para 'cec2022:F6' solo F6; para TMLAP
    siempre 1 problema.
    """
    if args.problem.startswith("cec2022:"):
        target = args.problem.split(":", 1)[1].strip()
        if target.lower() == "all":
            return [
                (
                    parse_problem_spec(f"cec2022:F{fid}", dim=args.dim),
                    f"F{fid}",
                )
                for fid in CEC_FUNCTION_IDS
            ]
        problem = parse_problem_spec(args.problem, dim=args.dim)
        return [(problem, f"F{problem.function_id}")]
    if args.problem.startswith("tmlap:"):
        problem = parse_problem_spec(
            args.problem, clients=args.clients, hubs=args.hubs
        )
        return [(problem, problem.name)]
    raise ValueError(f"Familia desconocida: {args.problem!r}")


def _seed_for(args, problem_idx, run_idx):
    """seed reproducible que separa problemas y corridas."""
    return int(args.seed + problem_idx * 1000 + run_idx)


def run_one(problem, args, run_id, problem_idx, problem_label):
    """Ejecuta una corrida del WO base sobre ``problem``. Devuelve dict + curva."""
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

    for iteration in range(args.iterations):
        (
            positions,
            _fit,
            best_score,
            best_pos,
            second_score,
            second_pos,
        ) = evaluate_and_update_leaders(
            positions, lb, ub, problem.evaluate,
            best_score, best_pos, second_score, second_pos,
        )
        gbest_x = np.tile(best_pos, (n_agents, 1))

        alpha, beta, r_signal, danger, safety = iteration_signals(
            iteration, args.iterations, rng
        )
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
            "algorithm": "WO_base",
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
            "elapsed_seconds": float(elapsed),
            "best_position": " ".join(f"{v:.12g}" for v in best_pos),
            "init_mode": args.init_mode,
        },
        "curve": curve,
    }


def build_statistics(summary_df):
    """Agrega min/max/mean/median/std por problema."""
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
    for problem_idx, (problem, label) in enumerate(problems):
        print(f"\n== {label} ({getattr(problem, 'family', 'unknown')}, dim={problem.dim}) ==",
              flush=True)
        for run_id in range(1, args.runs + 1):
            result = run_one(problem, args, run_id, problem_idx, label)
            rows.append(result["row"])
            curve_path = curves_dir / f"conv_curve_{label}_run{run_id}.csv"
            np.savetxt(curve_path, result["curve"], delimiter=",",
                       header="best_fitness", comments="")
            row = result["row"]
            print(
                f"  run={run_id}/{args.runs} seed={row['seed']} "
                f"final={row['final_fitness']:.6g} gap={row['gap_to_optimum']:.6g} "
                f"time={row['elapsed_seconds']:.2f}s",
                flush=True,
            )

    summary_df = pd.DataFrame(rows)
    summary_path = values_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)

    stats_df = build_statistics(summary_df)
    stats_path = values_dir / "statistics.csv"
    stats_df.to_csv(stats_path, index=False)

    print(f"\nResumen: {summary_path}")
    print(f"Estadisticas: {stats_path}")


if __name__ == "__main__":
    main()
