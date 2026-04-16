import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from base_visualizer import create_base_monitor
from cec2021_adapter import CEC2021Problem, FUNCTION_BIASES as CEC2021_BIASES
from cec_adapter import CECProblem as CEC2022Problem
from wo_paper_benchmark.paper_benchmark_23 import FUNCTION_SPECS as WO_PAPER_SPECS
from wo_paper_benchmark import WOPaper23Problem
from wo import run_wo


BENCHMARKS = {
    "cec2022": {
        "problem_class": CEC2022Problem,
        "biases": {
            1: 300.0,
            2: 400.0,
            3: 600.0,
            4: 800.0,
            5: 900.0,
            6: 1800.0,
            7: 2000.0,
            8: 2200.0,
            9: 2300.0,
            10: 2400.0,
            11: 2600.0,
            12: 2700.0,
        },
        "curve_name": lambda function_id: f"conv_curve_F{function_id}.csv",
        "curve_glob": "conv_curve_F*.csv",
        "summary_name": lambda agents, iterations: f"summary_wo_base_cec2022_{agents}x{iterations}_all_functions.csv",
        "comparison_name": lambda agents, iterations: f"comparison_wo_base_vs_cec2022_optimum_{agents}x{iterations}.csv",
        "monitor_filename": "interactive_monitor_base_all_functions.html",
        "page_title": "Monitor interactivo WO Base - CEC 2022",
        "chart_title": "Curva de convergencia WO base - CEC 2022",
        "hint_text": "Curva generada desde los CSV de `WO_base_cec/all_functions_outputs_final/values` para CEC 2022.",
    },
    "cec2021": {
        "problem_class": CEC2021Problem,
        "biases": CEC2021_BIASES,
        "curve_name": lambda function_id: f"conv_curve_cec2021_F{function_id}.csv",
        "curve_glob": "conv_curve_cec2021_F*.csv",
        "summary_name": lambda agents, iterations: f"summary_wo_base_cec2021_{agents}x{iterations}_all_functions.csv",
        "comparison_name": lambda agents, iterations: f"comparison_wo_base_vs_cec2021_optimum_{agents}x{iterations}.csv",
        "monitor_filename": "interactive_monitor_base_cec2021_all_functions.html",
        "page_title": "Monitor interactivo WO Base - CEC 2021",
        "chart_title": "Curva de convergencia WO base - CEC 2021",
        "hint_text": "Curva generada desde los CSV de `WO_base_cec/all_functions_outputs_final/values` para CEC 2021.",
    },
    "wo_paper_23": {
        "problem_class": WOPaper23Problem,
        "biases": {fid: spec["optimum"] for fid, spec in WO_PAPER_SPECS.items()},
        "curve_name": lambda function_id: f"conv_curve_wo_paper_23_F{function_id}.csv",
        "curve_glob": "conv_curve_wo_paper_23_F*.csv",
        "summary_name": lambda agents, iterations: f"summary_wo_base_wo_paper_23_{agents}x{iterations}_all_functions.csv",
        "comparison_name": lambda agents, iterations: f"comparison_wo_base_vs_wo_paper_23_optimum_{agents}x{iterations}.csv",
        "monitor_filename": "interactive_monitor_base_wo_paper_23_all_functions.html",
        "page_title": "Monitor interactivo WO Base - Paper WO (23 funciones)",
        "chart_title": "Curva de convergencia WO base - Benchmark del paper WO",
        "hint_text": "Curva generada desde los CSV de `WO_base_cec/all_functions_outputs_final/values` para las 23 funciones del paper de Walrus Optimizer.",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Ejecuta WO base sobre todas las funciones de un benchmark CEC.")
    parser.add_argument(
        "--benchmark",
        type=str,
        default="cec2021",
        choices=sorted(BENCHMARKS),
        help="Benchmark a ejecutar: cec2021, cec2022 o wo_paper_23.",
    )
    parser.add_argument("--dim", type=int, default=10, help="Dimensionalidad. Se ignora en wo_paper_23 porque cada funcion usa su dimension fija.")
    parser.add_argument("--agents", type=int, default=30, help="Tamano de poblacion.")
    parser.add_argument("--iterations", type=int, default=500, help="Iteraciones.")
    parser.add_argument("--seed", type=int, default=1234, help="Semilla base reproducible.")
    parser.add_argument(
        "--output",
        type=str,
        default="all_functions_outputs_final",
        help="Carpeta donde se guardan curvas, resumenes y panel.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    benchmark_key = args.benchmark.lower()
    benchmark = BENCHMARKS[benchmark_key]
    output_dir = Path(args.output)
    values_dir = output_dir / "values"
    graphs_dir = output_dir / "graphs"
    values_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for function_id in sorted(benchmark["biases"]):
        np.random.seed(args.seed + function_id - 1)
        if benchmark_key == "wo_paper_23":
            problem = benchmark["problem_class"](function_id)
        else:
            problem = benchmark["problem_class"](function_id, args.dim)

        optimum = benchmark["biases"][function_id]
        best_score, best_pos, convergence_curve = run_wo(
            args.agents,
            args.iterations,
            problem.lb,
            problem.ub,
            problem.dim,
            problem.evaluate,
        )

        curve_path = values_dir / benchmark["curve_name"](function_id)
        np.savetxt(curve_path, convergence_curve, delimiter=",", header="best_fitness", comments="")

        rows.append(
            {
                "benchmark": benchmark_key,
                "function": f"F{function_id}",
                "function_id": function_id,
                "dim": problem.dim,
                "agents": args.agents,
                "iterations": args.iterations,
                "seed": args.seed + function_id - 1,
                "initial_fitness": float(convergence_curve[0]),
                "final_fitness": float(best_score),
                "optimum": optimum,
                "gap_to_optimum": float(best_score - optimum) if not pd.isna(optimum) else np.nan,
                "best_position": " ".join(f"{value:.12g}" for value in best_pos),
            }
        )

        optimum_text = f"{optimum:.12f}" if not pd.isna(optimum) else "nan"
        gap_text = f"{best_score - optimum:.12f}" if not pd.isna(optimum) else "nan"
        print(
            f"{benchmark_key.upper()} F{function_id}: final={best_score:.12f}, "
            f"optimum={optimum_text}, "
            f"gap={gap_text}"
        )

    summary_df = pd.DataFrame(rows)
    summary_path = values_dir / benchmark["summary_name"](args.agents, args.iterations)
    summary_df.to_csv(summary_path, index=False)

    comparison_df = summary_df[["function", "final_fitness", "optimum", "gap_to_optimum"]].copy()
    comparison_path = values_dir / benchmark["comparison_name"](args.agents, args.iterations)
    comparison_df.to_csv(comparison_path, index=False)

    monitor_path = create_base_monitor(
        values_dir,
        graphs_dir,
        curve_glob=benchmark["curve_glob"],
        monitor_filename=benchmark["monitor_filename"],
        page_title=benchmark["page_title"],
        chart_title=benchmark["chart_title"],
        hint_text=benchmark["hint_text"],
    )

    print(f"Resumen: {summary_path}")
    print(f"Comparacion: {comparison_path}")
    if monitor_path is not None:
        print(f"Panel interactivo: {monitor_path}")


if __name__ == "__main__":
    main()
