import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cec_adapter import CECProblem
from multi_function_visualizer import create_all_functions_monitor
from online_controller import OnlineXAIController
from wo_controlled import run_wo_controlled


FUNCTION_METADATA = {
    1: {"name": "Zakharov", "family": "basic"},
    2: {"name": "Rosenbrock", "family": "basic"},
    3: {"name": "Schaffer F7", "family": "basic"},
    4: {"name": "Step Rastrigin", "family": "basic"},
    5: {"name": "Levy", "family": "basic"},
    6: {"name": "Hybrid Function 02", "family": "hybrid"},
    7: {"name": "Hybrid Function 10", "family": "hybrid"},
    8: {"name": "Hybrid Function 06", "family": "hybrid"},
    9: {"name": "Composition Function 01", "family": "composition"},
    10: {"name": "Composition Function 02", "family": "composition"},
    11: {"name": "Composition Function 06", "family": "composition"},
    12: {"name": "Composition Function 07", "family": "composition"},
}

DIFFICULTY_PRESETS = {
    "soft": {
        "function": "F1",
        "rationale": "Funcion basica de CEC 2022, adecuada como caso suave del benchmark.",
    },
    "medium": {
        "function": "F7",
        "rationale": "Funcion hibrida representativa, con dificultad intermedia para WO_shap.",
    },
    "hard": {
        "function": "F12",
        "rationale": "Funcion de composicion, adecuada como caso dificil del benchmark.",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Ejecuta una suite de pruebas WO_shap sobre CEC 2022 con tres casos "
            "representativos: soft, medium y hard."
        )
    )
    parser.add_argument(
        "--soft-function",
        type=str,
        default=DIFFICULTY_PRESETS["soft"]["function"],
        help="Funcion para el caso soft, por ejemplo F1.",
    )
    parser.add_argument(
        "--medium-function",
        type=str,
        default=DIFFICULTY_PRESETS["medium"]["function"],
        help="Funcion para el caso medium, por ejemplo F7.",
    )
    parser.add_argument(
        "--hard-function",
        type=str,
        default=DIFFICULTY_PRESETS["hard"]["function"],
        help="Funcion para el caso hard, por ejemplo F12.",
    )
    parser.add_argument("--dim", type=int, default=10, help="Dimensionalidad CEC 2022.")
    parser.add_argument("--agents", type=int, default=30, help="Tamano de poblacion.")
    parser.add_argument("--iterations", type=int, default=500, help="Iteraciones.")
    parser.add_argument(
        "--delta-window",
        type=int,
        default=50,
        help="Ventana para detectar estancamiento.",
    )
    parser.add_argument(
        "--max-shap-episodes",
        type=int,
        default=None,
        help="Numero maximo de episodios severos donde calcular SHAP exacto.",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Semilla base reproducible.")
    parser.add_argument(
        "--output",
        type=str,
        default="test_suite_outputs_cec2022",
        help="Carpeta donde se guardan resultados de la suite.",
    )
    return parser.parse_args()


def parse_function_id(function_label):
    if not isinstance(function_label, str):
        raise ValueError("La funcion debe ser un texto como 'F1'.")

    normalized = function_label.strip().upper()
    if not normalized.startswith("F"):
        raise ValueError("La funcion debe tener formato 'F1', 'F2', etc.")

    try:
        function_id = int(normalized[1:])
    except ValueError as exc:
        raise ValueError(f"Funcion invalida: {function_label}") from exc

    if function_id not in FUNCTION_METADATA:
        raise ValueError("La funcion CEC 2022 debe estar entre F1 y F12.")

    return function_id


def validate_configuration(cases, dim):
    if dim not in (2, 10, 20):
        raise ValueError("CEC 2022 solo define D=2, D=10 y D=20.")

    function_ids = [case["function_id"] for case in cases]
    if len(set(function_ids)) != len(function_ids):
        raise ValueError("La suite requiere tres funciones distintas para soft, medium y hard.")

    if dim == 2:
        invalid_for_dim2 = [fid for fid in function_ids if fid in (6, 7, 8)]
        if invalid_for_dim2:
            invalid_text = ", ".join(f"F{fid}" for fid in invalid_for_dim2)
            raise ValueError(
                f"CEC 2022 no define {invalid_text} para D=2. Usa D=10/D=20 o cambia esas funciones."
            )


def save_curve(values_dir, function_id, convergence_curve):
    curve_path = values_dir / f"conv_curve_shap_F{function_id}.csv"
    np.savetxt(curve_path, convergence_curve, delimiter=",", header="best_fitness", comments="")
    return curve_path


def save_case_result(values_dir, case, problem, args, seed, best_score, best_pos, controller):
    optimum = float(getattr(problem, "f_global", np.nan))
    event_summary = controller.event_summary()
    metadata = FUNCTION_METADATA[case["function_id"]]
    result_path = values_dir / f"result_wo_shap_{case['difficulty']}_F{case['function_id']}.csv"
    row = {
        "benchmark": "cec2022",
        "difficulty": case["difficulty"],
        "difficulty_rationale": case["rationale"],
        "function": f"F{case['function_id']}",
        "function_id": int(case["function_id"]),
        "function_name": metadata["name"],
        "function_family": metadata["family"],
        "dim": int(problem.dim),
        "agents": int(args.agents),
        "iterations": int(args.iterations),
        "delta_window": int(args.delta_window),
        "seed": int(seed),
        "final_fitness": float(best_score),
        "optimum": optimum,
        "gap_to_optimum": float(best_score - optimum) if not pd.isna(optimum) else np.nan,
        "event_count": int(event_summary["event_count"]),
        "episode_count": int(event_summary["episode_count"]),
        "best_position": " ".join(f"{value:.12g}" for value in best_pos),
        "status": "completed",
    }
    pd.DataFrame([row]).to_csv(result_path, index=False)
    row["result_csv"] = str(result_path)
    return row


def run_case(case, args, values_dir):
    seed = int(args.seed + case["function_id"] - 1)
    np.random.seed(seed)
    problem = CECProblem(case["function_id"], args.dim)
    controller = OnlineXAIController(
        delta_window=args.delta_window,
        max_shap_episodes=args.max_shap_episodes,
    )
    best_score, best_pos, convergence_curve = run_wo_controlled(
        args.agents,
        args.iterations,
        problem.lb,
        problem.ub,
        problem.dim,
        problem.evaluate,
        controller,
    )
    curve_path = save_curve(values_dir, case["function_id"], convergence_curve)
    controller.save_logs(values_dir, case["function_id"])
    row = save_case_result(values_dir, case, problem, args, seed, best_score, best_pos, controller)
    row["curve_csv"] = str(curve_path)
    return row


def build_cases(args):
    return [
        {
            "difficulty": "soft",
            "function_id": parse_function_id(args.soft_function),
            "rationale": DIFFICULTY_PRESETS["soft"]["rationale"],
        },
        {
            "difficulty": "medium",
            "function_id": parse_function_id(args.medium_function),
            "rationale": DIFFICULTY_PRESETS["medium"]["rationale"],
        },
        {
            "difficulty": "hard",
            "function_id": parse_function_id(args.hard_function),
            "rationale": DIFFICULTY_PRESETS["hard"]["rationale"],
        },
    ]


def main():
    args = parse_args()
    cases = build_cases(args)
    validate_configuration(cases, args.dim)

    output_dir = Path(args.output)
    values_dir = output_dir / "values"
    graphs_dir = output_dir / "graphs"
    values_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for case in cases:
        metadata = FUNCTION_METADATA[case["function_id"]]
        print(
            f"Ejecutando caso {case['difficulty']} -> F{case['function_id']} "
            f"({metadata['name']}, {metadata['family']})"
        )
        row = run_case(case, args, values_dir)
        rows.append(row)
        print(
            f"  final={row['final_fitness']:.12f}, optimum={row['optimum']:.12f}, "
            f"gap={row['gap_to_optimum']:.12f}, eventos={row['event_count']}, episodios={row['episode_count']}"
        )

    summary_df = pd.DataFrame(rows)
    summary_path = values_dir / "summary_wo_shap_cec2022_difficulty_suite.csv"
    summary_df.to_csv(summary_path, index=False)

    comparison_df = summary_df[
        [
            "difficulty",
            "function",
            "function_name",
            "function_family",
            "final_fitness",
            "optimum",
            "gap_to_optimum",
            "event_count",
            "episode_count",
            "status",
        ]
    ].copy()
    comparison_path = values_dir / "comparison_wo_shap_cec2022_difficulty_suite.csv"
    comparison_df.to_csv(comparison_path, index=False)

    monitor_path = create_all_functions_monitor(values_dir, graphs_dir)

    print(f"Resumen de la suite: {summary_path}")
    print(f"Comparacion de la suite: {comparison_path}")
    if monitor_path is not None:
        print(f"Panel interactivo: {monitor_path}")


if __name__ == "__main__":
    main()
