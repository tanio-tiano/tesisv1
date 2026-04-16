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

CONTROLLER_SENSITIVITY_PRESETS = {
    "soft": {
        "delta_window": 75,
        "action_cooldown": 90,
        "effective_action_cooldown": 120,
        "post_effective_guard_window": 180,
        "late_intervention_fraction": 0.84,
        "max_interventions": 6,
        "minimum_stagnation_scale": 1.20,
        "default_runs": None,
        "max_shap_episodes": None,
        "description": "Controlador relajado: ventana grande, intervenciones menos frecuentes.",
    },
    "medium": {
        "delta_window": 50,
        "action_cooldown": 32,
        "effective_action_cooldown": 60,
        "post_effective_guard_window": 130,
        "late_intervention_fraction": 0.92,
        "max_interventions": 8,
        "minimum_stagnation_scale": 0.95,
        "default_runs": 10,
        "max_shap_episodes": None,
        "description": "Controlador estándar: valores default.",
    },
    "hard": {
        "delta_window": 30,
        "action_cooldown": 20,
        "effective_action_cooldown": 40,
        "post_effective_guard_window": 90,
        "late_intervention_fraction": 0.95,
        "max_interventions": 10,
        "minimum_stagnation_scale": 0.70,
        "default_runs": 10,
        "max_shap_episodes": None,
        "description": "Controlador agresivo: ventana pequeña, intervenciones más frecuentes.",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Ejecuta suite de pruebas WO_shap sobre CEC 2022 (12 funciones). "
            "Soft/medium/hard depende del ajuste del controlador."
        )
    )
    parser.add_argument(
        "--controller-sensitivity",
        type=str,
        choices=["soft", "medium", "hard"],
        default=None,
        help="Ejecuta una unica instancia de sensibilidad del controlador.",
    )
    parser.add_argument(
        "--controller-sensitivities",
        type=str,
        default="soft,medium,hard",
        help="Lista separada por comas de instancias a ejecutar. Default: soft,medium,hard.",
    )
    parser.add_argument("--dim", type=int, default=10, help="Dimensionalidad CEC 2022.")
    parser.add_argument("--agents", type=int, default=30, help="Tamano de poblacion.")
    parser.add_argument("--iterations", type=int, default=500, help="Iteraciones.")
    parser.add_argument(
        "--delta-window",
        type=int,
        default=None,
        help="Ventana para detectar estancamiento (sobreescribe preset si se especifica).",
    )
    parser.add_argument(
        "--max-shap-episodes",
        type=int,
        default=None,
        help="Numero maximo de episodios severos donde calcular SHAP exacto.",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Semilla base reproducible.")
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Numero de corridas independientes a ejecutar (default: 1).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="test_suite_outputs_cec2022",
        help="Carpeta donde se guardan resultados de la suite.",
    )
    return parser.parse_args()


def parse_controller_sensitivities(args):
    if args.controller_sensitivity is not None:
        return [str(args.controller_sensitivity).strip().lower()]

    raw = str(args.controller_sensitivities).strip()
    if not raw:
        return ["soft", "medium", "hard"]

    values = [item.strip().lower() for item in raw.split(",") if item.strip()]
    allowed = {"soft", "medium", "hard"}
    invalid = [item for item in values if item not in allowed]
    if invalid:
        raise ValueError(
            f"Instancias de sensibilidad no soportadas: {', '.join(invalid)}. "
            "Usa soft, medium y/o hard."
        )

    ordered = []
    seen = set()
    for value in values:
        if value not in seen:
            ordered.append(value)
            seen.add(value)
    return ordered


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
        raise ValueError("La suite requiere funciones distintas.")

    if dim == 2:
        invalid_for_dim2 = [fid for fid in function_ids if fid in (6, 7, 8)]
        if invalid_for_dim2:
            invalid_text = ", ".join(f"F{fid}" for fid in invalid_for_dim2)
            raise ValueError(
                f"CEC 2022 no define {invalid_text} para D=2. Usa D=10/D=20 o cambia esas funciones."
            )


def save_curve(values_dir, function_id, convergence_curve, run_id=None):
    if run_id is not None:
        curve_path = values_dir / f"conv_curve_shap_F{function_id}_run{run_id}.csv"
    else:
        curve_path = values_dir / f"conv_curve_shap_F{function_id}.csv"
    np.savetxt(curve_path, convergence_curve, delimiter=",", header="best_fitness", comments="")
    return curve_path


def resolve_controller_sensitivity(args):
    sensitivity = getattr(args, "controller_sensitivity", "medium")
    preset = dict(CONTROLLER_SENSITIVITY_PRESETS[sensitivity])
    if args.delta_window is not None:
        preset["delta_window"] = int(args.delta_window)
    if args.max_shap_episodes is not None:
        preset["max_shap_episodes"] = int(args.max_shap_episodes)
    preset["controller_sensitivity"] = sensitivity
    return preset


def resolve_instance_runs(args, sensitivity):
    preset_runs = CONTROLLER_SENSITIVITY_PRESETS[sensitivity].get("default_runs")
    if preset_runs is None:
        return int(args.runs)
    return int(preset_runs)


def build_controller(args):
    controller_config = resolve_controller_sensitivity(args)
    controller = OnlineXAIController(
        delta_window=int(controller_config["delta_window"]),
        max_shap_episodes=controller_config["max_shap_episodes"],
        sensitivity_profile="medium",
        max_interventions=int(controller_config["max_interventions"]),
    )

    controller.sensitivity_profile = str(controller_config["controller_sensitivity"])
    controller.delta_window = int(controller_config["delta_window"])
    controller.action_cooldown = int(controller_config["action_cooldown"])
    controller.effective_action_cooldown = int(controller_config["effective_action_cooldown"])
    controller.post_effective_guard_window = int(controller_config["post_effective_guard_window"])
    controller.late_intervention_fraction = float(controller_config["late_intervention_fraction"])
    controller.max_interventions = int(controller_config["max_interventions"])
    controller.minimum_stagnation_scale = float(controller_config["minimum_stagnation_scale"])
    controller.max_shap_episodes = controller_config["max_shap_episodes"]
    return controller, controller_config


def build_statistics(summary_df):
    stats_rows = []
    for function_id in range(1, 13):
        case_data = summary_df[summary_df["function_id"] == function_id]
        if len(case_data) == 0:
            continue

        fitness_std = float(case_data["final_fitness"].std(ddof=0))
        gap_std = float(case_data["gap_to_optimum"].std(ddof=0))
        event_std = float(case_data["event_count"].std(ddof=0))
        episode_std = float(case_data["episode_count"].std(ddof=0))

        stats_rows.append(
            {
                "controller_sensitivity": str(case_data["controller_sensitivity"].iloc[0]),
                "function": f"F{function_id}",
                "function_id": function_id,
                "function_name": FUNCTION_METADATA[function_id]["name"],
                "function_family": FUNCTION_METADATA[function_id]["family"],
                "runs": int(len(case_data)),
                "fitness_best": float(case_data["final_fitness"].min()),
                "fitness_worst": float(case_data["final_fitness"].max()),
                "fitness_mean": float(case_data["final_fitness"].mean()),
                "fitness_average": float(case_data["final_fitness"].mean()),
                "fitness_std": fitness_std,
                "gap_best": float(case_data["gap_to_optimum"].min()),
                "gap_worst": float(case_data["gap_to_optimum"].max()),
                "gap_mean": float(case_data["gap_to_optimum"].mean()),
                "gap_average": float(case_data["gap_to_optimum"].mean()),
                "gap_std": gap_std,
                "event_count_mean": float(case_data["event_count"].mean()),
                "event_count_std": event_std,
                "episode_count_mean": float(case_data["episode_count"].mean()),
                "episode_count_std": episode_std,
            }
        )
    return pd.DataFrame(stats_rows)


def write_suite_index_html(output_dir, suite_rows):
    cards = []
    for row in suite_rows:
        summary_rel = Path(row["summary_path"]).relative_to(output_dir).as_posix()
        stats_rel = Path(row["stats_path"]).relative_to(output_dir).as_posix()
        links = []
        if row.get("monitor_path") is not None:
            monitor_rel = Path(row["monitor_path"]).relative_to(output_dir).as_posix()
            links.append(f'<li><a href="{monitor_rel}">Visualizador HTML</a></li>')
        links.append(f'<li><a href="{summary_rel}">Resumen CSV por corrida</a></li>')
        links.append(f'<li><a href="{stats_rel}">Estadisticas CSV por funcion</a></li>')
        cards.append(
            f"""
            <div class="card">
              <h2>{row['sensitivity']}</h2>
              <p>{row['description']}</p>
              <ul>
                {''.join(links)}
              </ul>
            </div>
            """
        )

    html = f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>WO SHAP CEC2022 - Instancias de Sensibilidad</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; background: #f4f4f4; color: #222; }}
    .wrap {{ max-width: 1100px; margin: 0 auto; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 18px; }}
    .card {{ background: white; border-radius: 14px; padding: 20px; box-shadow: 0 8px 24px rgba(0,0,0,0.08); }}
    h1 {{ margin-top: 0; }}
    a {{ color: #0f4c81; text-decoration: none; font-weight: 600; }}
    a:hover {{ text-decoration: underline; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>WO SHAP - CEC 2022</h1>
    <p>Indice de resultados para las instancias soft, medium y hard del controlador.</p>
    <div class="grid">
      {''.join(cards)}
    </div>
  </div>
</body>
</html>
"""
    index_path = output_dir / "index_instances_cec2022.html"
    index_path.write_text(html, encoding="utf-8")
    return index_path


def save_case_result(values_dir, case, problem, args, seed, best_score, best_pos, controller, run_id=None):
    optimum = float(getattr(problem, "f_global", np.nan))
    event_summary = controller.event_summary()
    metadata = FUNCTION_METADATA[case["function_id"]]
    if run_id is not None:
        result_path = values_dir / f"result_wo_shap_F{case['function_id']}_run{run_id}.csv"
    else:
        result_path = values_dir / f"result_wo_shap_F{case['function_id']}.csv"
    
    row = {
        "run_id": int(run_id) if run_id is not None else 1,
        "benchmark": "cec2022",
        "controller_sensitivity": str(controller.sensitivity_profile),
        "function": f"F{case['function_id']}",
        "function_id": int(case["function_id"]),
        "function_name": metadata["name"],
        "function_family": metadata["family"],
        "dim": int(problem.dim),
        "agents": int(args.agents),
        "iterations": int(args.iterations),
        "delta_window": int(controller.delta_window),
        "action_cooldown": int(controller.action_cooldown),
        "effective_action_cooldown": int(controller.effective_action_cooldown),
        "post_effective_guard_window": int(controller.post_effective_guard_window),
        "late_intervention_fraction": float(controller.late_intervention_fraction),
        "max_interventions": int(controller.max_interventions),
        "minimum_stagnation_scale": float(controller.minimum_stagnation_scale),
        "max_shap_episodes": controller.max_shap_episodes,
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


def run_case(case, args, values_dir, run_id=None):
    seed = int(args.seed + case["function_id"] - 1 + (run_id - 1) * 1000 if run_id else args.seed + case["function_id"] - 1)
    np.random.seed(seed)
    problem = CECProblem(case["function_id"], args.dim)

    controller, controller_config = build_controller(args)
    best_score, best_pos, convergence_curve = run_wo_controlled(
        args.agents,
        args.iterations,
        problem.lb,
        problem.ub,
        problem.dim,
        problem.evaluate,
        controller,
    )
    curve_path = save_curve(values_dir, case["function_id"], convergence_curve, run_id)
    
    # Crear carpeta de logs por corrida si hay múltiples ejecuciones
    if run_id is not None and run_id > 1:
        logs_dir = values_dir / f"run{run_id}_logs"
        logs_dir.mkdir(exist_ok=True)
        controller.save_logs(logs_dir, case["function_id"])
    else:
        controller.save_logs(values_dir, case["function_id"])
    
    row = save_case_result(values_dir, case, problem, args, seed, best_score, best_pos, controller, run_id)
    row["controller_description"] = str(controller_config["description"])
    row["curve_csv"] = str(curve_path)
    return row


def consolidate_logs(values_dir, num_runs, cases):
    """Consolida los logs de cada corrida en su propio archivo único."""
    
    consolidated_files = {}
    
    for run_id in range(1, num_runs + 1):
        run_consolidated = []
        
        for case in cases:
            fid = case["function_id"]
            
            # Determinar carpeta de logs
            if run_id == 1:
                logs_dir = values_dir
            else:
                logs_dir = values_dir / f"run{run_id}_logs"
            
            # Cargar state
            state_file = logs_dir / f"controller_state_F{fid}.csv"
            if not state_file.exists():
                continue
            
            state_df = pd.read_csv(state_file)
            state_df["run_id"] = run_id
            state_df["function_id"] = fid
            
            # Cargar events
            events_file = logs_dir / f"controller_events_F{fid}.csv"
            events_df = pd.read_csv(events_file) if events_file.exists() else pd.DataFrame()
            if not events_df.empty:
                events_df["run_id"] = run_id
                events_df["function_id"] = fid
            
            # Cargar shap_values
            shap_file = logs_dir / f"shap_values_F{fid}.csv"
            shap_df = pd.read_csv(shap_file) if shap_file.exists() else pd.DataFrame()
            if not shap_df.empty:
                shap_df["run_id"] = run_id
                shap_df["function_id"] = fid
            
            # Cargar episodes
            episodes_file = logs_dir / f"controller_episodes_F{fid}.csv"
            episodes_df = pd.read_csv(episodes_file) if episodes_file.exists() else pd.DataFrame()
            if not episodes_df.empty:
                episodes_df["run_id"] = run_id
                episodes_df["function_id"] = fid
            
            # Cargar convergence curves
            curve_file = values_dir / f"conv_curve_shap_F{fid}_run{run_id}.csv"
            if curve_file.exists():
                curve_data = np.loadtxt(curve_file, delimiter=",", skiprows=1)
                if curve_data.ndim == 1:
                    curve_data = curve_data.reshape(-1, 1)
                curves_dict = {
                    int(iteration): float(fitness[0])
                    for iteration, fitness in enumerate(curve_data)
                }
                state_df["best_fitness_curve"] = state_df["iteration"].map(curves_dict)
            
            # Mergear events a state (left merge por event_id)
            if not events_df.empty:
                # Seleccionar columnas relevantes de events
                events_cols = [col for col in events_df.columns 
                             if col not in ["run_id", "function_id"]]
                state_df = state_df.merge(
                    events_df[events_cols],
                    on="event_id",
                    how="left",
                    suffixes=("", "_event")
                )
            
            # Mergear shap a state (left merge por event_id y phase si existe)
            if not shap_df.empty:
                shap_cols = [col for col in shap_df.columns 
                           if col not in ["run_id", "function_id"] and col.startswith("SHAP_")]
                if "phase" in shap_df.columns:
                    # Para cada row, tomar pre-shap
                    shap_pre = shap_df[shap_df["phase"] == "pre"].copy()
                    shap_pre_cols = [col for col in shap_pre.columns 
                                   if col.startswith("SHAP_")]
                    for col in shap_pre_cols:
                        shap_pre = shap_pre.rename(columns={col: f"{col}_pre"})
                    state_df = state_df.merge(
                        shap_pre[["event_id", "run_id", "function_id"] + 
                                [f"{col}_pre" for col in shap_pre_cols]],
                        on=["event_id", "run_id", "function_id"],
                        how="left"
                    )
                else:
                    state_df = state_df.merge(
                        shap_df[["event_id", "run_id", "function_id"] + shap_cols],
                        on=["event_id", "run_id", "function_id"],
                        how="left"
                    )
            
            # Mergear episodes a state (left merge por episode_id si existe)
            if not episodes_df.empty and "episode_id" in state_df.columns:
                episode_cols = [col for col in episodes_df.columns 
                              if col not in ["run_id", "function_id"]]
                state_df = state_df.merge(
                    episodes_df[episode_cols],
                    on="episode_id",
                    how="left"
                )
            
            run_consolidated.append(state_df)
        
        # Guardar archivo para esta corrida
        if run_consolidated:
            run_df = pd.concat(run_consolidated, ignore_index=True)
            
            # Reorganizar columnas para mejor legibilidad
            core_cols = ["run_id", "function_id", "iteration", "event_id", "episode_id"]
            state_cols = ["alpha", "beta", "danger_signal", "safety_signal", 
                         "diversity", "diversity_norm", "stagnation", "stagnation_length"]
            event_cols = [col for col in run_df.columns 
                         if col.startswith("event_") or col in ["action_taken", "action_kind", 
                                                                 "rescue_fraction", "rescue_mode"]]
            shap_cols = [col for col in run_df.columns if col.startswith("SHAP_")]
            episode_cols = [col for col in run_df.columns 
                           if col.startswith("episode_") or col in ["start_iteration", "end_iteration"]]
            curve_cols = ["best_fitness_curve"]
            other_cols = [col for col in run_df.columns 
                         if col not in (core_cols + state_cols + event_cols + 
                                       shap_cols + episode_cols + curve_cols)]
            
            # Reordenar
            final_cols = (core_cols + state_cols + curve_cols + event_cols + 
                         shap_cols + episode_cols + other_cols)
            final_cols = [col for col in final_cols if col in run_df.columns]
            
            run_df = run_df[final_cols]
            
            consolidated_path = values_dir / f"consolidated_run{run_id}_all_functions.csv"
            run_df.to_csv(consolidated_path, index=False)
            consolidated_files[f"run{run_id}"] = consolidated_path
    
    return consolidated_files


def consolidate_all_runs(values_dir, num_runs, cases, sensitivity="medium"):
    """Consolida todas las corridas de una instancia en un único archivo."""
    all_dfs = []
    
    for run_id in range(1, num_runs + 1):
        consolidated_path = values_dir / f"consolidated_run{run_id}_all_functions.csv"
        if consolidated_path.exists():
            df = pd.read_csv(consolidated_path)
            all_dfs.append(df)
    
    if all_dfs:
        full_df = pd.concat(all_dfs, ignore_index=True)
        full_df["controller_sensitivity"] = sensitivity
        
        # Reordenar para que sensitivity sea una de las primeras columnas
        cols = list(full_df.columns)
        if "controller_sensitivity" in cols:
            cols.remove("controller_sensitivity")
        cols = ["controller_sensitivity"] + cols
        full_df = full_df[cols]
        
        global_path = values_dir / f"consolidated_all_runs_{sensitivity}_instance.csv"
        full_df.to_csv(global_path, index=False)
        return global_path
    
    return None


def build_cases(args):
    """Construye casos para todas las 12 funciones de CEC 2022."""
    cases = []
    for function_id in range(1, 13):
        cases.append({
            "function_id": function_id,
            "function_name": FUNCTION_METADATA[function_id]["name"],
            "function_family": FUNCTION_METADATA[function_id]["family"],
        })
    return cases


def main():
    args = parse_args()
    sensitivities = parse_controller_sensitivities(args)
    cases = build_cases(args)
    validate_configuration(cases, args.dim)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    suite_rows = []
    global_rows = []

    for sensitivity in sensitivities:
        args.controller_sensitivity = sensitivity
        instance_runs = resolve_instance_runs(args, sensitivity)
        instance_dir = output_dir / sensitivity
        values_dir = instance_dir / "values"
        graphs_dir = instance_dir / "graphs"
        values_dir.mkdir(parents=True, exist_ok=True)
        graphs_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*80}")
        print(f"INSTANCIA {sensitivity.upper()}")
        print(f"{'='*80}")
        print(CONTROLLER_SENSITIVITY_PRESETS[sensitivity]["description"])
        print(f"Corridas configuradas para esta instancia: {instance_runs}")

        all_rows = []
        for run_id in range(1, instance_runs + 1):
            print(f"\n{'-'*80}")
            print(f"CORRIDA {run_id}/{instance_runs}")
            print(f"{'-'*80}\n")

            rows = []
            for case in cases:
                metadata = FUNCTION_METADATA[case["function_id"]]
                print(
                    f"Ejecutando F{case['function_id']} ({metadata['name']}, {metadata['family']})"
                )
                row = run_case(case, args, values_dir, run_id)
                rows.append(row)
                print(
                    f"  final={row['final_fitness']:.12f}, optimum={row['optimum']:.12f}, "
                    f"gap={row['gap_to_optimum']:.12f}, eventos={row['event_count']}, episodios={row['episode_count']}"
                )

            all_rows.extend(rows)

        summary_df = pd.DataFrame(all_rows)
        summary_path = values_dir / f"summary_wo_shap_cec2022_{sensitivity}_all_runs.csv"
        summary_df.to_csv(summary_path, index=False)

        stats_df = build_statistics(summary_df)
        stats_path = values_dir / f"statistics_wo_shap_cec2022_{sensitivity}_aggregated.csv"
        stats_df.to_csv(stats_path, index=False)

        comparison_df = summary_df[
            [
                "run_id",
                "controller_sensitivity",
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
        comparison_path = values_dir / f"comparison_wo_shap_cec2022_{sensitivity}_all_runs.csv"
        comparison_df.to_csv(comparison_path, index=False)

        consolidated_files = consolidate_logs(values_dir, instance_runs, cases)
        global_consolidated_path = consolidate_all_runs(
            values_dir,
            instance_runs,
            cases,
            sensitivity=sensitivity,
        )
        if global_consolidated_path:
            print(f"\nConsolidado global de instancia: {global_consolidated_path}")

        monitor_path = create_all_functions_monitor(values_dir, graphs_dir)

        print(f"\n{'='*80}")
        print(f"RESUMEN FINAL - INSTANCIA {sensitivity.upper()}")
        print(f"{'='*80}")
        print(f"Resumen completo: {summary_path}")
        print(f"Comparacion: {comparison_path}")
        print(f"Estadísticas agregadas: {stats_path}")
        print(f"\nARCHIVOS CONSOLIDADOS:")
        for log_type, filepath in consolidated_files.items():
            print(f"  - {log_type}: {filepath}")
        if monitor_path is not None:
            print(f"\nPanel interactivo: {monitor_path}")

        suite_rows.append(
            {
                "sensitivity": sensitivity,
                "description": CONTROLLER_SENSITIVITY_PRESETS[sensitivity]["description"],
                "summary_path": summary_path,
                "stats_path": stats_path,
                "comparison_path": comparison_path,
                "monitor_path": monitor_path,
            }
        )
        global_rows.append(summary_df)

    if global_rows:
        global_summary_df = pd.concat(global_rows, ignore_index=True)
        global_summary_path = output_dir / "summary_wo_shap_cec2022_all_instances.csv"
        global_summary_df.to_csv(global_summary_path, index=False)

        global_stats_df = (
            global_summary_df.groupby(
                ["controller_sensitivity", "function", "function_id", "function_name", "function_family"],
                as_index=False,
            )
            .agg(
                runs=("run_id", "count"),
                fitness_best=("final_fitness", "min"),
                fitness_worst=("final_fitness", "max"),
                fitness_mean=("final_fitness", "mean"),
                fitness_average=("final_fitness", "mean"),
                fitness_std=("final_fitness", lambda s: float(s.std(ddof=0))),
                gap_best=("gap_to_optimum", "min"),
                gap_worst=("gap_to_optimum", "max"),
                gap_mean=("gap_to_optimum", "mean"),
                gap_average=("gap_to_optimum", "mean"),
                gap_std=("gap_to_optimum", lambda s: float(s.std(ddof=0))),
                event_count_mean=("event_count", "mean"),
                event_count_std=("event_count", lambda s: float(s.std(ddof=0))),
                episode_count_mean=("episode_count", "mean"),
                episode_count_std=("episode_count", lambda s: float(s.std(ddof=0))),
            )
        )
        global_stats_path = output_dir / "statistics_wo_shap_cec2022_all_instances.csv"
        global_stats_df.to_csv(global_stats_path, index=False)

        index_path = write_suite_index_html(output_dir, suite_rows)

        print(f"\n{'='*80}")
        print("RESUMEN GLOBAL DE INSTANCIAS")
        print(f"{'='*80}")
        print(f"Resumen global: {global_summary_path}")
        print(f"Estadísticas globales: {global_stats_path}")
        print(f"Índice HTML: {index_path}")


if __name__ == "__main__":
    main()
