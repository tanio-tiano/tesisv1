import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cec_adapter import CECProblem
from online_controller import OnlineXAIController
from wo_controlled import run_wo_controlled

SELECTED_FUNCTION = "F2"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ejecuta Walrus Optimizer con controlador SHAP sobre una funcion CEC 2022."
    )
    parser.add_argument("--dim", type=int, default=10, help="Dimensionalidad.")
    parser.add_argument("--agents", type=int, default=50, help="Tamano de poblacion.")
    parser.add_argument("--iterations", type=int, default=500, help="Iteraciones.")
    parser.add_argument("--seed", type=int, default=1234, help="Semilla aleatoria.")
    parser.add_argument(
        "--output",
        type=str,
        default="test_outputs",
        help="Carpeta donde se guardan resultados.",
    )
    return parser.parse_args()


def select_function(function_id):
    if not isinstance(function_id, str):
        raise ValueError("SELECTED_FUNCTION debe ser un texto como 'F1'.")

    normalized = function_id.strip().upper()
    if not normalized.startswith("F"):
        raise ValueError("SELECTED_FUNCTION debe tener formato 'F1', 'F2', etc.")

    return int(normalized[1:])


def validate_function(function_id):
    if function_id < 1 or function_id > 12:
        raise ValueError("La funcion debe estar entre 1 y 12.")


def save_outputs(output_dir, function_id, convergence_curve, controller):
    output_dir.mkdir(parents=True, exist_ok=True)

    curve_path = output_dir / f"conv_curve_shap_F{function_id}.csv"
    np.savetxt(curve_path, convergence_curve, delimiter=",", header="best_fitness", comments="")

    figure_path = output_dir / f"conv_curve_shap_F{function_id}.png"
    plt.figure(figsize=(9, 5))
    plt.plot(range(1, len(convergence_curve) + 1), convergence_curve, linewidth=2)
    plt.xlabel("Iteraciones")
    plt.ylabel("Mejor fitness")
    plt.title(f"Curva de convergencia WO + SHAP - CEC 2022 F{function_id}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=150)
    plt.close()

    controller.save_logs(output_dir, function_id)
    return curve_path, figure_path


def main():
    args = parse_args()
    function_id = select_function(SELECTED_FUNCTION)
    validate_function(function_id)
    output_dir = Path(args.output)

    print(f"Ejecutando WO con SHAP, funcion F{function_id}...")

    np.random.seed(args.seed)
    problem = CECProblem(function_id, args.dim)
    controller = OnlineXAIController()

    best_score, best_pos, convergence_curve = run_wo_controlled(
        args.agents,
        args.iterations,
        problem.lb,
        problem.ub,
        args.dim,
        problem.evaluate,
        controller,
    )

    curve_path, figure_path = save_outputs(
        output_dir, function_id, convergence_curve, controller
    )

    print("Ejecucion finalizada.")
    print(f"Funcion evaluada: F{function_id}")
    print(f"Mejor fitness encontrado: {best_score}")
    print(f"Mejor posicion encontrada: {best_pos}")
    print(f"CSV de convergencia: {curve_path}")
    print(f"PNG de convergencia: {figure_path}")
    print(
        f"CSV de estados del controlador: {output_dir / f'controller_state_F{function_id}.csv'}"
    )
    print(f"CSV de valores SHAP: {output_dir / f'shap_values_F{function_id}.csv'}")


if __name__ == "__main__":
    main()
