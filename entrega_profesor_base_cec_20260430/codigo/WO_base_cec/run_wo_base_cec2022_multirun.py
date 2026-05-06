import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from halton import hal
from initialization import initialization
from levy_flight import levy_flight
from opfunu_cec_adapter import (
    OpfunuCECProblem,
    benchmark_function_ids,
    get_function_metadata,
    normalize_benchmark,
    parse_function_id as parse_opfunu_function_id,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ejecuta WO base multirun sobre benchmarks CEC usando opfunu."
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=["cec2022"],
        default="cec2022",
        help="Benchmark opfunu a ejecutar. Actualmente solo CEC 2022.",
    )
    parser.add_argument("--dim", type=int, default=10, help="Dimensionalidad CEC.")
    parser.add_argument("--agents", type=int, default=30, help="Tamano de poblacion.")
    parser.add_argument("--iterations", type=int, default=500, help="Iteraciones.")
    parser.add_argument("--runs", type=int, default=30, help="Corridas independientes.")
    parser.add_argument("--seed", type=int, default=1234, help="Semilla base reproducible.")
    parser.add_argument(
        "--functions",
        type=str,
        default=None,
        help="Lista de funciones separadas por coma, por ejemplo F1,F6,F12. Default: F1-F12.",
    )
    parser.add_argument(
        "--gbest-mode",
        type=str,
        choices=["updated", "legacy"],
        default="legacy",
        help="legacy reproduce el WO.m de MathWorks con GBestX inicial en cero; updated usa best_pos como guia global.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="all_functions_outputs_30runs",
        help="Carpeta de salida.",
    )
    return parser.parse_args()


def parse_function_id(label):
    return parse_opfunu_function_id(label, benchmark="cec2022")


def selected_functions(args):
    benchmark = normalize_benchmark(args.benchmark)
    if args.functions is None:
        return benchmark_function_ids(benchmark)
    return [
        parse_opfunu_function_id(item, benchmark=benchmark)
        for item in args.functions.split(",")
        if item.strip()
    ]


def evaluate_and_update_leaders(
    positions,
    lb,
    ub,
    objective,
    best_score,
    best_pos,
    second_score,
    second_pos,
):
    positions = np.clip(positions, lb, ub)
    fitness_values = np.zeros(positions.shape[0], dtype=float)
    for i in range(positions.shape[0]):
        fitness = float(objective(positions[i, :]))
        fitness_values[i] = fitness
        if fitness < best_score:
            best_score = fitness
            best_pos = positions[i, :].copy()
        if fitness > best_score and fitness < second_score:
            second_score = fitness
            second_pos = positions[i, :].copy()
    return positions, fitness_values, best_score, best_pos, second_score, second_pos


def run_wo_base(search_agents_no, max_iter, lb, ub, dim, objective, gbest_mode="legacy"):
    best_pos = np.zeros(dim)
    second_pos = np.zeros(dim)
    best_score = float("inf")
    second_score = float("inf")
    gbest_x = np.tile(best_pos, (search_agents_no, 1))

    positions = initialization(search_agents_no, dim, ub, lb)
    convergence_curve = np.zeros(max_iter)

    ratio = 0.4
    female_count = round(search_agents_no * ratio)
    male_count = female_count
    child_count = search_agents_no - female_count - male_count

    for iteration in range(max_iter):
        (
            positions,
            _fitness_values,
            best_score,
            best_pos,
            second_score,
            second_pos,
        ) = evaluate_and_update_leaders(
            positions,
            lb,
            ub,
            objective,
            best_score,
            best_pos,
            second_score,
            second_pos,
        )
        if gbest_mode == "updated":
            gbest_x = np.tile(best_pos, (search_agents_no, 1))

        alpha = 1 - iteration / max_iter
        beta = 1 - 1 / (1 + np.exp((0.5 * max_iter - iteration) / max_iter * 10))
        r_signal = 2 * np.random.rand() - 1
        danger_signal = 2 * alpha * r_signal
        safety_signal = np.random.rand()

        if abs(danger_signal) >= 1:
            r3 = np.random.rand()
            p1 = np.random.permutation(search_agents_no)
            p2 = np.random.permutation(search_agents_no)
            positions = positions + (beta * r3**2) * (positions[p1, :] - positions[p2, :])
        else:
            if safety_signal >= 0.5:
                for i in range(male_count):
                    positions[i, :] = lb + hal(i + 1, 7) * (ub - lb)

                last_male_index = male_count - 1
                for j in range(male_count, male_count + female_count):
                    positions[j, :] = positions[j, :] + alpha * (
                        positions[last_male_index, :] - positions[j, :]
                    ) + (1 - alpha) * (gbest_x[j, :] - positions[j, :])

                for i in range(search_agents_no - child_count, search_agents_no):
                    p = np.random.rand()
                    o = gbest_x[i, :] + positions[i, :] * levy_flight(dim)
                    positions[i, :] = p * (o - positions[i, :])

            if safety_signal < 0.5 and abs(danger_signal) >= 0.5:
                for i in range(search_agents_no):
                    r4 = np.random.rand()
                    positions[i, :] = positions[i, :] * r_signal - np.abs(
                        gbest_x[i, :] - positions[i, :]
                    ) * r4**2

            if safety_signal < 0.5 and abs(danger_signal) < 0.5:
                for i in range(search_agents_no):
                    for j_dim in range(dim):
                        theta1 = np.random.rand()
                        a1 = beta * np.random.rand() - beta
                        b1 = np.tan(theta1 * np.pi)
                        x1 = best_pos[j_dim] - a1 * b1 * abs(
                            best_pos[j_dim] - positions[i, j_dim]
                        )

                        theta2 = np.random.rand()
                        a2 = beta * np.random.rand() - beta
                        b2 = np.tan(theta2 * np.pi)
                        x2 = second_pos[j_dim] - a2 * b2 * abs(
                            second_pos[j_dim] - positions[i, j_dim]
                        )
                        positions[i, j_dim] = (x1 + x2) / 2

        convergence_curve[iteration] = best_score

    return best_score, best_pos, convergence_curve


def build_statistics(summary_df):
    rows = []
    for function_id, data in summary_df.groupby("function_id", sort=True):
        first = data.iloc[0]
        rows.append(
            {
                "function": f"F{int(function_id)}",
                "function_id": int(function_id),
                "function_name": first["function_name"],
                "function_family": first["function_family"],
                "runs": int(data["run_id"].nunique()),
                "fitness_best": float(data["final_fitness"].min()),
                "fitness_worst": float(data["final_fitness"].max()),
                "fitness_mean": float(data["final_fitness"].mean()),
                "fitness_median": float(data["final_fitness"].median()),
                "fitness_std": float(data["final_fitness"].std(ddof=1)),
                "gap_best": float(data["gap_to_optimum"].min()),
                "gap_worst": float(data["gap_to_optimum"].max()),
                "gap_mean": float(data["gap_to_optimum"].mean()),
                "gap_median": float(data["gap_to_optimum"].median()),
                "gap_std": float(data["gap_to_optimum"].std(ddof=1)),
            }
        )
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    args.benchmark = normalize_benchmark(args.benchmark)
    function_ids = selected_functions(args)
    output_dir = Path(args.output)
    values_dir = output_dir / "values"
    values_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for run_id in range(1, args.runs + 1):
        print(f"\nRUN {run_id}/{args.runs}")
        for function_id in function_ids:
            seed = int(args.seed + function_id - 1 + (run_id - 1) * 1000)
            np.random.seed(seed)
            problem = OpfunuCECProblem(args.benchmark, function_id, args.dim)
            metadata = get_function_metadata(args.benchmark, function_id, dim=args.dim)

            best_score, best_pos, curve = run_wo_base(
                args.agents,
                args.iterations,
                problem.lb,
                problem.ub,
                problem.dim,
                problem.evaluate,
                gbest_mode=args.gbest_mode,
            )

            curve_path = values_dir / f"conv_curve_F{function_id}_run{run_id}.csv"
            np.savetxt(curve_path, curve, delimiter=",", header="best_fitness", comments="")
            optimum = float(metadata["optimum"])
            row = {
                "run_id": run_id,
                "benchmark": args.benchmark,
                "benchmark_source": "opfunu",
                "algorithm": "WO_base",
                "gbest_mode": args.gbest_mode,
                "function": f"F{function_id}",
                "function_id": function_id,
                "function_name": metadata["name"],
                "function_family": metadata["family"],
                "dim": problem.dim,
                "agents": args.agents,
                "iterations": args.iterations,
                "seed": seed,
                "initial_fitness": float(curve[0]),
                "final_fitness": float(best_score),
                "optimum": optimum,
                "gap_to_optimum": float(best_score - optimum),
                "best_position": " ".join(f"{value:.12g}" for value in best_pos),
                "curve_csv": str(curve_path),
                "status": "completed",
            }
            rows.append(row)
            pd.DataFrame([row]).to_csv(
                values_dir / f"result_wo_base_F{function_id}_run{run_id}.csv",
                index=False,
            )
            print(
                f"  F{function_id}: final={best_score:.12f}, "
                f"gap={best_score - optimum:.12f}"
            )

    summary_df = pd.DataFrame(rows)
    summary_path = values_dir / (
        f"summary_wo_base_{args.benchmark}_{args.agents}x{args.iterations}_"
        f"{args.runs}runs_{args.gbest_mode}.csv"
    )
    summary_df.to_csv(summary_path, index=False)

    stats_df = build_statistics(summary_df)
    stats_path = values_dir / (
        f"statistics_wo_base_{args.benchmark}_{args.agents}x{args.iterations}_"
        f"{args.runs}runs_{args.gbest_mode}.csv"
    )
    stats_df.to_csv(stats_path, index=False)

    print(f"\nResumen: {summary_path}")
    print(f"Estadisticas: {stats_path}")


if __name__ == "__main__":
    main()
