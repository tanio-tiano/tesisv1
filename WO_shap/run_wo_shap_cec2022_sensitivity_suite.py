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

PROFILE_ORDER = ("soft", "medium", "hard")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Ejecuta WO+SHAP sobre CEC 2022 con perfiles de sensibilidad "
            "soft, medium y hard. Pensado como runner simple para WSL."
        )
    )
    parser.add_argument(
        "--functions",
        type=str,
        default="all",
        help="Lista separada por comas (ej: F1,F7,F12) o 'all' para F1-F12.",
    )
    parser.add_argument(
        "--profiles",
        type=str,
        default="soft,medium,hard",
        help="Perfiles a ejecutar, separados por comas.",
    )
    parser.add_argument("--dim", type=int, default=10, help="Dimensionalidad CEC 2022.")
    parser.add_argument("--agents", type=int, default=30, help="Tamano de poblacion.")
    parser.add_argument("--iterations", type=int, default=500, help="Iteraciones.")
    parser.add_argument(
        "--delta-window",
        type=int,
        default=50,
        help="Ventana base de estancamiento para el perfil medium.",
    )
    parser.add_argument(
        "--max-shap-episodes",
        type=int,
        default=None,
        help="Numero maximo de episodios seleccionados para SHAP exacto.",
    )
    parser.add_argument(
        "--max-interventions",
        type=int,
        default=None,
        help="Tope duro opcional de intervenciones por corrida.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Numero de corridas por funcion y perfil. Default=1 para prueba rapida.",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Semilla base reproducible.")
    parser.add_argument(
        "--output",
        type=str,
        default="sensitivity_suite_cec2022_outputs",
        help="Carpeta de salida raiz.",
    )
    return parser.parse_args()


def parse_profiles(text):
    profiles = [item.strip().lower() for item in text.split(",") if item.strip()]
    if not profiles:
        raise ValueError("Debes indicar al menos un perfil.")

    invalid = [profile for profile in profiles if profile not in PROFILE_ORDER]
    if invalid:
        raise ValueError(
            f"Perfiles no soportados: {', '.join(invalid)}. Usa: {', '.join(PROFILE_ORDER)}."
        )

    seen = set()
    ordered = []
    for profile in profiles:
        if profile not in seen:
            ordered.append(profile)
            seen.add(profile)
    return ordered


def parse_functions(text):
    if text.strip().lower() == "all":
        return list(range(1, 13))

    function_ids = []
    for token in text.split(","):
        normalized = token.strip().upper()
        if not normalized:
            continue
        if not normalized.startswith("F"):
            raise ValueError(f"Funcion invalida: {token}. Usa formato F1, F2, ..., F12.")
        try:
            function_id = int(normalized[1:])
        except ValueError as exc:
            raise ValueError(f"Funcion invalida: {token}.") from exc
        if function_id not in FUNCTION_METADATA:
            raise ValueError(f"Funcion fuera de rango: {normalized}. Usa F1-F12.")
        function_ids.append(function_id)

    if not function_ids:
        raise ValueError("Debes indicar al menos una funcion.")
    return sorted(set(function_ids))


def save_curve(values_dir, function_id, convergence_curve):
    curve_path = values_dir / f"conv_curve_shap_F{function_id}.csv"
    np.savetxt(curve_path, convergence_curve, delimiter=",", header="best_fitness", comments="")
    return curve_path


def write_index_html(output_dir, rows):
    profile_paths = {}
    for row in rows:
        monitor_html = row.get("monitor_html")
        if monitor_html:
            profile_paths[row["profile"]] = monitor_html

    links = []
    for profile in PROFILE_ORDER:
        if profile in profile_paths:
            rel = Path(profile_paths[profile]).relative_to(output_dir)
            links.append(
                f'<li><a href="{rel.as_posix()}">{profile}</a></li>'
            )

    html = f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>WO SHAP Sensitivity Suite</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; background:#f7f7f7; color:#222; }}
    .card {{ max-width: 900px; background:white; border-radius:16px; padding:24px; box-shadow:0 8px 24px rgba(0,0,0,0.08); }}
    h1 {{ margin-top:0; }}
    a {{ color:#0f4c81; text-decoration:none; font-weight:600; }}
    a:hover {{ text-decoration:underline; }}
    code {{ background:#f0f0f0; padding:2px 6px; border-radius:6px; }}
  </style>
</head>
<body>
  <div class="card">
    <h1>WO SHAP Sensitivity Suite - CEC 2022</h1>
    <p>Indice de visualizadores HTML generados para los perfiles de sensibilidad.</p>
    <ul>
      {''.join(links)}
    </ul>
    <p>Resumen consolidado: <code>values/summary_wo_shap_cec2022_sensitivity_suite.csv</code></p>
  </div>
</body>
</html>
"""
    index_path = output_dir / "index_sensitivity_suite.html"
    index_path.write_text(html, encoding="utf-8")
    return index_path


def build_compact_comparison(aggregate_df):
    id_columns = ["function", "function_name", "function_family"]
    metrics = [
        "mean_final_fitness",
        "mean_gap_to_optimum",
        "mean_event_count",
        "mean_episode_count",
    ]

    result = None
    for metric in metrics:
        pivot = aggregate_df.pivot(
            index=id_columns,
            columns="profile",
            values=metric,
        ).reset_index()
        pivot.columns = [
            column if isinstance(column, str) else "_".join(str(part) for part in column if part)
            for column in pivot.columns.to_flat_index()
        ]
        rename_map = {}
        for profile in PROFILE_ORDER:
            if profile in pivot.columns:
                rename_map[profile] = f"{metric}_{profile}"
        pivot = pivot.rename(columns=rename_map)

        if result is None:
            result = pivot
        else:
            result = result.merge(pivot, on=id_columns, how="outer")

    ordered_columns = id_columns[:]
    for metric in metrics:
        for profile in PROFILE_ORDER:
            column_name = f"{metric}_{profile}"
            if column_name in result.columns:
                ordered_columns.append(column_name)

    return result[ordered_columns]


def run_single_case(function_id, profile, run_id, args, values_dir):
    seed = int(args.seed + (function_id - 1) * 100 + run_id)
    np.random.seed(seed)

    problem = CECProblem(function_id, args.dim)
    controller = OnlineXAIController(
        delta_window=args.delta_window,
        max_shap_episodes=args.max_shap_episodes,
        sensitivity_profile=profile,
        max_interventions=args.max_interventions,
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

    curve_path = save_curve(values_dir, function_id, convergence_curve)
    controller.save_logs(values_dir, function_id)
    summary = controller.event_summary()
    optimum = float(getattr(problem, "f_global", np.nan))
    meta = FUNCTION_METADATA[function_id]

    return {
        "benchmark": "cec2022",
        "profile": profile,
        "function": f"F{function_id}",
        "function_id": int(function_id),
        "function_name": meta["name"],
        "function_family": meta["family"],
        "run_id": int(run_id),
        "dim": int(problem.dim),
        "agents": int(args.agents),
        "iterations": int(args.iterations),
        "delta_window_base": int(args.delta_window),
        "seed": int(seed),
        "final_fitness": float(best_score),
        "optimum": optimum,
        "gap_to_optimum": float(best_score - optimum) if not pd.isna(optimum) else np.nan,
        "event_count": int(summary["event_count"]),
        "episode_count": int(summary["episode_count"]),
        "actions": str(summary["actions"]),
        "effects": str(summary["effects"]),
        "curve_csv": str(curve_path),
        "best_position": " ".join(f"{value:.12g}" for value in best_pos),
    }


def main():
    args = parse_args()
    if int(args.runs) != 1:
        raise ValueError(
            "Este runner genera un monitor HTML por perfil y actualmente esta preparado para runs=1."
        )
    function_ids = parse_functions(args.functions)
    profiles = parse_profiles(args.profiles)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    root_values_dir = output_dir / "values"
    root_values_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for profile in profiles:
        profile_dir = output_dir / profile
        values_dir = profile_dir / "values"
        graphs_dir = profile_dir / "graphs"
        values_dir.mkdir(parents=True, exist_ok=True)
        graphs_dir.mkdir(parents=True, exist_ok=True)

        print(f"Perfil {profile}: ejecutando funciones {', '.join(f'F{i}' for i in function_ids)}")

        profile_rows = []
        for function_id in function_ids:
            for run_id in range(1, args.runs + 1):
                row = run_single_case(function_id, profile, run_id, args, values_dir)
                profile_rows.append(row)
                rows.append(dict(row))
                print(
                    f"  {profile} F{function_id} run={run_id}: "
                    f"final={row['final_fitness']:.12f}, gap={row['gap_to_optimum']:.12f}, "
                    f"eventos={row['event_count']}, episodios={row['episode_count']}"
                )

        profile_summary_path = values_dir / f"summary_{profile}_wo_shap_cec2022.csv"
        pd.DataFrame(profile_rows).to_csv(profile_summary_path, index=False)

        monitor_path = create_all_functions_monitor(values_dir, graphs_dir)
        if monitor_path is not None:
            for row in rows:
                if row["profile"] == profile:
                    row["monitor_html"] = str(monitor_path)
        print(f"  resumen perfil: {profile_summary_path}")
        if monitor_path is not None:
            print(f"  monitor perfil: {monitor_path}")

    summary_df = pd.DataFrame(rows)
    summary_path = root_values_dir / "summary_wo_shap_cec2022_sensitivity_suite.csv"
    summary_df.to_csv(summary_path, index=False)

    aggregate_df = (
        summary_df.groupby(["profile", "function", "function_name", "function_family"], as_index=False)
        .agg(
            runs=("run_id", "count"),
            mean_final_fitness=("final_fitness", "mean"),
            mean_gap_to_optimum=("gap_to_optimum", "mean"),
            mean_event_count=("event_count", "mean"),
            mean_episode_count=("episode_count", "mean"),
        )
    )
    aggregate_path = root_values_dir / "aggregate_wo_shap_cec2022_sensitivity_suite.csv"
    aggregate_df.to_csv(aggregate_path, index=False)

    compact_df = build_compact_comparison(aggregate_df)
    compact_path = root_values_dir / "compact_profile_comparison_wo_shap_cec2022.csv"
    compact_df.to_csv(compact_path, index=False)

    index_path = write_index_html(output_dir, rows)

    print(f"Resumen consolidado: {summary_path}")
    print(f"Agregado consolidado: {aggregate_path}")
    print(f"Comparacion compacta: {compact_path}")
    print(f"Indice HTML: {index_path}")


if __name__ == "__main__":
    main()
