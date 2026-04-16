import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from base_visualizer import create_base_monitor
from cec2021_adapter import CEC2021Problem
from cec_adapter import CECProblem as CEC2022Problem
from wo_paper_benchmark import WOPaper23Problem
from wo import run_wo

SELECTED_FUNCTION = "F2"
BENCHMARKS = {
    "cec2022": {
        "problem_class": CEC2022Problem,
        "max_function": 12,
        "curve_name": lambda function_id: f"conv_curve_F{function_id}.csv",
        "curve_glob": "conv_curve_F*.csv",
        "monitor_filename": "interactive_monitor_base_all_functions.html",
        "page_title": "Monitor interactivo WO Base - CEC 2022",
        "chart_title": "Curva de convergencia WO base - CEC 2022",
        "hint_text": "Curva generada desde los CSV de `WO_base_cec/all_functions_outputs_final/values` para CEC 2022.",
    },
    "cec2021": {
        "problem_class": CEC2021Problem,
        "max_function": 20,
        "curve_name": lambda function_id: f"conv_curve_cec2021_F{function_id}.csv",
        "curve_glob": "conv_curve_cec2021_F*.csv",
        "monitor_filename": "interactive_monitor_base_cec2021_all_functions.html",
        "page_title": "Monitor interactivo WO Base - CEC 2021",
        "chart_title": "Curva de convergencia WO base - CEC 2021",
        "hint_text": "Curva generada desde los CSV de `WO_base_cec/all_functions_outputs_final/values` para CEC 2021.",
    },
    "wo_paper_23": {
        "problem_class": WOPaper23Problem,
        "max_function": 23,
        "curve_name": lambda function_id: f"conv_curve_wo_paper_23_F{function_id}.csv",
        "curve_glob": "conv_curve_wo_paper_23_F*.csv",
        "monitor_filename": "interactive_monitor_base_wo_paper_23_all_functions.html",
        "page_title": "Monitor interactivo WO Base - Paper WO (23 funciones)",
        "chart_title": "Curva de convergencia WO base - Benchmark del paper WO",
        "hint_text": "Curva generada desde los CSV de `WO_base_cec/all_functions_outputs_final/values` para las 23 funciones del paper de Walrus Optimizer.",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ejecuta Walrus Optimizer sobre una funcion CEC."
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="cec2022",
        choices=sorted(BENCHMARKS),
        help="Benchmark a ejecutar: cec2021, cec2022 o wo_paper_23.",
    )
    parser.add_argument(
        "--function",
        type=str,
        default=SELECTED_FUNCTION,
        help="Funcion CEC a ejecutar, por ejemplo F1 o F12.",
    )
    parser.add_argument("--dim", type=int, default=10, help="Dimensionalidad.")
    parser.add_argument("--agents", type=int, default=30, help="Tamano de poblacion.")
    parser.add_argument("--iterations", type=int, default=500, help="Iteraciones.")
    parser.add_argument("--seed", type=int, default=1234, help="Semilla aleatoria.")
    parser.add_argument(
        "--output",
        type=str,
        default="all_functions_outputs_final",
        help="Carpeta donde se guardan curva y panel interactivo.",
    )
    return parser.parse_args()


def select_function(function_id):
    if not isinstance(function_id, str):
        raise ValueError("SELECTED_FUNCTION debe ser un texto como 'F1'.")

    normalized = function_id.strip().upper()
    if not normalized.startswith("F"):
        raise ValueError("SELECTED_FUNCTION debe tener formato 'F1', 'F2', etc.")

    return int(normalized[1:])


def save_outputs(output_dir, function_id, convergence_curve, benchmark_key):
    benchmark_info = BENCHMARKS[benchmark_key]
    output_dir.mkdir(parents=True, exist_ok=True)
    values_dir = output_dir / "values"
    graphs_dir = output_dir / "graphs"
    values_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir.mkdir(parents=True, exist_ok=True)

    curve_path = values_dir / benchmark_info["curve_name"](function_id)
    np.savetxt(curve_path, convergence_curve, delimiter=",", header="best_fitness", comments="")

    monitor_path = create_base_monitor(
        values_dir,
        graphs_dir,
        curve_glob=benchmark_info["curve_glob"],
        monitor_filename=benchmark_info["monitor_filename"],
        page_title=benchmark_info["page_title"],
        chart_title=benchmark_info["chart_title"],
        hint_text=benchmark_info["hint_text"],
    )

    return curve_path, monitor_path


def problem_optimum(problem):
    return float(getattr(problem, "f_global", np.nan))


def save_result_summary(output_dir, benchmark_key, function_id, problem, agents, iterations, seed, best_score, best_pos):
    values_dir = output_dir / "values"
    values_dir.mkdir(parents=True, exist_ok=True)
    optimum = problem_optimum(problem)
    summary_path = values_dir / f"result_wo_base_{benchmark_key}_F{function_id}.csv"
    pd.DataFrame(
        [
            {
                "benchmark": benchmark_key,
                "function": f"F{function_id}",
                "function_id": int(function_id),
                "dim": int(problem.dim),
                "agents": int(agents),
                "iterations": int(iterations),
                "seed": int(seed),
                "final_fitness": float(best_score),
                "optimum": optimum,
                "gap_to_optimum": float(best_score - optimum) if not pd.isna(optimum) else np.nan,
                "best_position": " ".join(f"{value:.12g}" for value in best_pos),
            }
        ]
    ).to_csv(summary_path, index=False)
    return summary_path, optimum


def main():
    args = parse_args()
    benchmark_key = args.benchmark.lower()
    benchmark_info = BENCHMARKS[benchmark_key]
    function_id = select_function(args.function)
    if function_id < 1 or function_id > benchmark_info["max_function"]:
        raise ValueError(
            f"La funcion para {benchmark_key.upper()} debe estar entre 1 y {benchmark_info['max_function']}."
        )
    output_dir = Path(args.output)

    print(f"Ejecutando WO sobre {benchmark_key.upper()}, funcion F{function_id}...")

    np.random.seed(args.seed)
    if benchmark_key == "wo_paper_23":
        problem = benchmark_info["problem_class"](function_id)
    else:
        problem = benchmark_info["problem_class"](function_id, args.dim)

    problem_dim = int(problem.dim)
    best_score, best_pos, convergence_curve = run_wo(
        args.agents,
        args.iterations,
        problem.lb,
        problem.ub,
        problem_dim,
        problem.evaluate,
    )

    curve_path, monitor_path = save_outputs(output_dir, function_id, convergence_curve, benchmark_key)
    summary_path, optimum = save_result_summary(
        output_dir,
        benchmark_key,
        function_id,
        problem,
        args.agents,
        args.iterations,
        args.seed,
        best_score,
        best_pos,
    )

    print("Ejecucion finalizada.")
    print(f"Benchmark evaluado: {benchmark_key.upper()}")
    print(f"Funcion evaluada: F{function_id}")
    print(f"Mejor fitness encontrado: {best_score}")
    if not pd.isna(optimum):
        print(f"Optimo de referencia: {optimum}")
        print(f"Gap al optimo: {best_score - optimum}")
    print(f"Mejor posicion encontrada: {best_pos}")
    print(f"CSV de convergencia: {curve_path}")
    print(f"CSV de resumen: {summary_path}")
    if monitor_path is not None:
        print(f"Panel interactivo WO base: {monitor_path}")


if __name__ == "__main__":
    main()
