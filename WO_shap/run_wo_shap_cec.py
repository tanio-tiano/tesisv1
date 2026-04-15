import argparse
from pathlib import Path
import webbrowser

import numpy as np

from cec_adapter import CECProblem
from online_controller import OnlineXAIController
from multi_function_visualizer import create_all_functions_monitor
from wo_controlled import run_wo_controlled

SELECTED_FUNCTION = "F2"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ejecuta Walrus Optimizer con controlador SHAP sobre una funcion CEC 2022."
    )
    parser.add_argument(
        "--function",
        type=str,
        default=SELECTED_FUNCTION,
        help="Funcion CEC a ejecutar, por ejemplo F1 o F12.",
    )
    parser.add_argument("--dim", type=int, default=10, help="Dimensionalidad.")
    parser.add_argument("--agents", type=int, default=50, help="Tamano de poblacion.")
    parser.add_argument("--iterations", type=int, default=500, help="Iteraciones.")
    parser.add_argument(
        "--delta-window",
        type=int,
        default=50,
        help="Ventana para calcular delta fitness y detectar estancamiento.",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Semilla aleatoria.")
    parser.add_argument(
        "--output",
        type=str,
        default="all_functions_outputs_final",
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
    values_dir = output_dir / "values"
    graphs_dir = output_dir / "graphs"
    values_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir.mkdir(parents=True, exist_ok=True)

    curve_path = values_dir / f"conv_curve_shap_F{function_id}.csv"
    np.savetxt(curve_path, convergence_curve, delimiter=",", header="best_fitness", comments="")

    controller.save_logs(values_dir, function_id)
    monitor_path = create_all_functions_monitor(values_dir, graphs_dir)
    return curve_path, monitor_path


def main():
    args = parse_args()
    function_id = select_function(args.function)
    validate_function(function_id)
    output_dir = Path(args.output)

    print(f"Ejecutando WO con SHAP, funcion F{function_id}...")

    np.random.seed(args.seed)
    problem = CECProblem(function_id, args.dim)
    controller = OnlineXAIController(delta_window=args.delta_window)

    best_score, best_pos, convergence_curve = run_wo_controlled(
        args.agents,
        args.iterations,
        problem.lb,
        problem.ub,
        args.dim,
        problem.evaluate,
        controller,
    )

    curve_path, monitor_path = save_outputs(
        output_dir, function_id, convergence_curve, controller
    )
    event_summary = controller.event_summary()

    print("Ejecucion finalizada.")
    print(f"Funcion evaluada: F{function_id}")
    print(f"Mejor fitness encontrado: {best_score}")
    print(f"Mejor posicion encontrada: {best_pos}")
    print(f"CSV de convergencia: {curve_path}")
    print(
        f"CSV de estados del controlador: {output_dir / 'values' / f'controller_state_F{function_id}.csv'}"
    )
    print(f"CSV de valores SHAP: {output_dir / 'values' / f'shap_values_F{function_id}.csv'}")
    print(f"CSV de eventos del controlador: {output_dir / 'values' / f'controller_events_F{function_id}.csv'}")
    if monitor_path is not None:
        print(f"Panel interactivo: {monitor_path}")
        try:
            webbrowser.open(monitor_path.resolve().as_uri())
        except Exception:
            pass
    print(f"Activaciones del controlador: {event_summary['event_count']}")
    if event_summary["actions"]:
        print(f"Resumen por accion: {event_summary['actions']}")
    if event_summary["stagnation_states"]:
        print(f"Resumen por estado de estancamiento: {event_summary['stagnation_states']}")


if __name__ == "__main__":
    main()
