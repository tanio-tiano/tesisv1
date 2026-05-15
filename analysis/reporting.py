from __future__ import annotations

import base64
import json
import math
from html import escape
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

try:
    import opfunu
except ImportError:  # pragma: no cover - only used in broken environments
    opfunu = None


FEATURE_COLUMNS = [
    "alpha",
    "beta",
    "danger_signal",
    "safety_signal",
    "diversity",
    "iteration",
]


def _numeric_sort_key(value):
    text = str(value)
    digits = "".join(ch for ch in text if ch.isdigit())
    return int(digits) if digits else math.inf


def _html_table(df, max_rows=None, float_format="{:.6g}"):
    if df is None or df.empty:
        return "<p class=\"muted\">No hay datos disponibles.</p>"
    display_df = df.copy()
    if max_rows is not None:
        display_df = display_df.head(max_rows)
    return display_df.to_html(
        index=False,
        escape=True,
        border=0,
        classes="data-table",
        float_format=lambda value: float_format.format(value),
    )


def _save_figure(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _image_tag(path, alt):
    if path is None or not Path(path).exists():
        return ""
    data = Path(path).read_bytes()
    encoded = base64.b64encode(data).decode("ascii")
    return f'<img src="data:image/png;base64,{encoded}" alt="{escape(alt)}">'


def _resolve_csv_path(path_value, output_dir, summary_path):
    if not isinstance(path_value, str) or not path_value.strip():
        return None
    candidate = Path(path_value)
    candidates = []
    if candidate.is_absolute():
        candidates.append(candidate)
    else:
        candidates.extend(
            [
                Path.cwd() / candidate,
                output_dir / candidate,
                output_dir.parent / candidate,
                summary_path.parent / candidate,
                summary_path.parent.parent / candidate,
            ]
        )
    for item in candidates:
        if item.exists():
            return item
    return None


def _read_curve(path):
    try:
        values = pd.read_csv(path)["best_fitness"].astype(float).to_numpy()
    except Exception:
        values = np.loadtxt(path, delimiter=",", skiprows=1)
    return np.asarray(values, dtype=float)


def build_paper_statistics(summary_df):
    rows = []
    for function_id, data in summary_df.groupby("function_id", sort=True):
        function = f"F{int(function_id)}"
        elapsed = (
            data["elapsed_seconds"].astype(float)
            if "elapsed_seconds" in data.columns
            else pd.Series(dtype=float)
        )
        rows.append(
            {
                "function": function,
                "function_id": int(function_id),
                "best": float(data["final_fitness"].min()),
                "worst": float(data["final_fitness"].max()),
                "avg_mean": float(data["final_fitness"].mean()),
                "std": float(data["final_fitness"].std(ddof=1)),
                "time_mean_seconds": float(elapsed.mean()) if not elapsed.empty else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("function_id")


def build_shapiro_normality(summary_df, value_column="gap_to_optimum"):
    rows = []
    for function_id, data in summary_df.groupby("function_id", sort=True):
        values = (
            pd.to_numeric(data[value_column], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        statistic = np.nan
        p_value = np.nan
        if len(values) >= 3 and values.nunique() > 1:
            statistic, p_value = stats.shapiro(values)
        elif len(values) >= 3:
            p_value = 1.0
        rows.append(
            {
                "function": f"F{int(function_id)}",
                "function_id": int(function_id),
                "n": int(len(values)),
                "mean": float(values.mean()) if len(values) else np.nan,
                "std": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                "shapiro_statistic": float(statistic) if pd.notna(statistic) else np.nan,
                "shapiro_p": float(p_value) if pd.notna(p_value) else np.nan,
                "normal_alpha_0_05": bool(p_value > 0.05) if pd.notna(p_value) else False,
            }
        )
    return pd.DataFrame(rows).sort_values("function_id")


def _find_base_summary(current_summary, base_results_root):
    if base_results_root is None:
        return None
    root = Path(base_results_root)
    if not root.exists():
        return None

    candidates = sorted(
        root.rglob("summary_wo_base*.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return None

    current_first = current_summary.iloc[0]
    current_functions = set(current_summary["function_id"].astype(int))
    for candidate in candidates:
        try:
            base_df = pd.read_csv(candidate)
        except Exception:
            continue
        if base_df.empty or "function_id" not in base_df.columns:
            continue
        base_first = base_df.iloc[0]
        base_functions = set(base_df["function_id"].astype(int))
        same_config = (
            str(base_first.get("benchmark", "")) == str(current_first.get("benchmark", ""))
            and int(base_first.get("dim", -1)) == int(current_first.get("dim", -2))
            and int(base_first.get("agents", -1)) == int(current_first.get("agents", -2))
            and int(base_first.get("iterations", -1))
            == int(current_first.get("iterations", -2))
            and current_functions.issubset(base_functions)
        )
        if same_config:
            return candidate
    return None


def build_base_comparison(summary_df, base_summary_path):
    if base_summary_path is None:
        return pd.DataFrame(), None
    base_df = pd.read_csv(base_summary_path)
    rows = []
    for function_id, shap_data in summary_df.groupby("function_id", sort=True):
        base_data = base_df[base_df["function_id"].astype(int) == int(function_id)]
        if base_data.empty:
            continue

        shap_values = pd.to_numeric(
            shap_data["gap_to_optimum"], errors="coerce"
        ).dropna()
        base_values = pd.to_numeric(
            base_data["gap_to_optimum"], errors="coerce"
        ).dropna()
        if shap_values.empty or base_values.empty:
            continue

        mann_stat = np.nan
        mann_p = np.nan
        if len(shap_values) >= 2 and len(base_values) >= 2:
            mann_stat, mann_p = stats.mannwhitneyu(
                shap_values,
                base_values,
                alternative="two-sided",
            )

        paired = pd.merge(
            shap_data[["run_id", "gap_to_optimum"]],
            base_data[["run_id", "gap_to_optimum"]],
            on="run_id",
            suffixes=("_shap", "_base"),
        ).dropna()
        wilcoxon_stat = np.nan
        wilcoxon_p = np.nan
        if len(paired) >= 2:
            differences = (
                pd.to_numeric(paired["gap_to_optimum_shap"], errors="coerce")
                - pd.to_numeric(paired["gap_to_optimum_base"], errors="coerce")
            ).dropna()
            if len(differences) >= 2 and not np.allclose(differences, 0):
                wilcoxon_stat, wilcoxon_p = stats.wilcoxon(
                    differences,
                    alternative="two-sided",
                    zero_method="wilcox",
                )
            elif len(differences) >= 2:
                wilcoxon_p = 1.0

        mean_gap_shap = float(shap_values.mean())
        mean_gap_base = float(base_values.mean())
        delta = mean_gap_shap - mean_gap_base
        p_for_outcome = wilcoxon_p if pd.notna(wilcoxon_p) else mann_p
        if pd.notna(p_for_outcome) and p_for_outcome <= 0.05:
            outcome = "SHAP_better" if delta < 0 else "BASE_better"
        else:
            outcome = "Tie_or_not_significant"

        rows.append(
            {
                "function": f"F{int(function_id)}",
                "function_id": int(function_id),
                "n_shap": int(len(shap_values)),
                "n_base": int(len(base_values)),
                "paired_n": int(len(paired)),
                "mean_gap_shap": mean_gap_shap,
                "mean_gap_base": mean_gap_base,
                "delta_gap_shap_minus_base": delta,
                "wilcoxon_signed_rank_statistic": float(wilcoxon_stat)
                if pd.notna(wilcoxon_stat)
                else np.nan,
                "wilcoxon_signed_rank_p": float(wilcoxon_p)
                if pd.notna(wilcoxon_p)
                else np.nan,
                "mann_whitney_u_statistic": float(mann_stat)
                if pd.notna(mann_stat)
                else np.nan,
                "mann_whitney_u_p": float(mann_p) if pd.notna(mann_p) else np.nan,
                "significant_alpha_0_05": bool(
                    pd.notna(p_for_outcome) and p_for_outcome <= 0.05
                ),
                "outcome": outcome,
            }
        )

    comparison_df = pd.DataFrame(rows).sort_values("function_id")
    if comparison_df.empty:
        return comparison_df, None

    summary = {
        "base_summary_path": str(base_summary_path),
        "shap_better": int((comparison_df["outcome"] == "SHAP_better").sum()),
        "base_better": int((comparison_df["outcome"] == "BASE_better").sum()),
        "ties_or_not_significant": int(
            (comparison_df["outcome"] == "Tie_or_not_significant").sum()
        ),
    }
    return comparison_df, summary


def _plot_gap_boxplot(summary_df, plots_dir):
    functions = sorted(summary_df["function"].unique(), key=_numeric_sort_key)
    values = [
        pd.to_numeric(
            summary_df.loc[summary_df["function"] == function, "gap_to_optimum"],
            errors="coerce",
        )
        .dropna()
        .to_numpy()
        for function in functions
    ]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.boxplot(values, labels=functions, showfliers=True)
    ax.set_title("Boxplots de gap al optimo por funcion")
    ax.set_xlabel("Funcion")
    ax.set_ylabel("Gap to optimum")
    positive_values = np.concatenate([v[v > 0] for v in values if len(v)])
    if len(positive_values) and np.nanmax(positive_values) > 100 * max(
        np.nanmin(positive_values), 1e-9
    ):
        ax.set_yscale("symlog", linthresh=1e-6)
        ax.set_ylabel("Gap to optimum (escala symlog)")
    ax.grid(axis="y", alpha=0.25)
    return _save_figure(fig, plots_dir / "boxplot_gap_to_optimum.png")


def _plot_mean_gap(summary_df, plots_dir):
    data = (
        summary_df.groupby(["function_id", "function"], as_index=False)["gap_to_optimum"]
        .mean()
        .sort_values("function_id")
    )
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(data["function"], data["gap_to_optimum"], color="#4b74b9")
    ax.set_title("Gap promedio al optimo")
    ax.set_xlabel("Funcion")
    ax.set_ylabel("Mean gap to optimum")
    ax.grid(axis="y", alpha=0.25)
    return _save_figure(fig, plots_dir / "mean_gap_to_optimum.png")


def _plot_time(summary_df, plots_dir):
    if "elapsed_seconds" not in summary_df.columns:
        return None
    data = (
        summary_df.groupby(["function_id", "function"], as_index=False)["elapsed_seconds"]
        .mean()
        .sort_values("function_id")
    )
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(data["function"], data["elapsed_seconds"], color="#587d65")
    ax.set_title("Tiempo promedio por funcion")
    ax.set_xlabel("Funcion")
    ax.set_ylabel("Segundos")
    ax.grid(axis="y", alpha=0.25)
    return _save_figure(fig, plots_dir / "time_mean_by_function.png")


def _plot_convergence_by_function(summary_df, output_dir, summary_path, plots_dir):
    paths = []
    for function_id, data in summary_df.groupby("function_id", sort=True):
        curves = []
        for path_value in data.get("curve_csv", []):
            path = _resolve_csv_path(path_value, output_dir, summary_path)
            if path is None:
                continue
            try:
                curves.append(_read_curve(path))
            except Exception:
                continue
        if not curves:
            continue
        min_len = min(len(curve) for curve in curves)
        matrix = np.vstack([curve[:min_len] for curve in curves])
        mean_curve = np.mean(matrix, axis=0)

        function = f"F{int(function_id)}"
        fig, ax = plt.subplots(figsize=(9, 4.8))
        for curve in matrix:
            ax.plot(curve, color="#9aa9c2", alpha=0.22, linewidth=0.8)
        ax.plot(mean_curve, color="#1f4e8c", linewidth=2.2, label="Promedio")
        ax.set_title(f"Curva de convergencia {function}")
        ax.set_xlabel("Iteracion")
        ax.set_ylabel("Best fitness")
        positive_values = matrix[matrix > 0]
        if len(positive_values) and np.nanmax(positive_values) > 100 * max(
            np.nanmin(positive_values), 1e-9
        ):
            ax.set_yscale("symlog", linthresh=1e-6)
            ax.set_ylabel("Best fitness (escala symlog)")
        ax.grid(alpha=0.25)
        ax.legend()
        paths.append(_save_figure(fig, plots_dir / f"convergence_{function}.png"))
    return paths


def _plot_controller_events(summary_df, plots_dir):
    required = {
        "control_events",
        "accepted_control_events",
        "rejected_control_events",
        "effective_control_events",
    }
    if not required.issubset(summary_df.columns):
        return None
    data = (
        summary_df.groupby(["function_id", "function"], as_index=False)[list(required)]
        .mean()
        .sort_values("function_id")
    )
    x = np.arange(len(data))
    width = 0.2
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - 1.5 * width, data["control_events"], width, label="Propuestas")
    ax.bar(x - 0.5 * width, data["accepted_control_events"], width, label="Aceptadas")
    ax.bar(x + 0.5 * width, data["rejected_control_events"], width, label="Rechazadas")
    ax.bar(x + 1.5 * width, data["effective_control_events"], width, label="Efectivas")
    ax.set_xticks(x, data["function"])
    ax.set_title("Eventos del controlador por funcion")
    ax.set_xlabel("Funcion")
    ax.set_ylabel("Promedio por corrida")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    return _save_figure(fig, plots_dir / "controller_events_by_function.png")


def _load_shap_rows(output_dir):
    shap_dir = output_dir / "shap"
    if not shap_dir.exists():
        return pd.DataFrame()
    frames = []
    for path in shap_dir.glob("*.csv"):
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _summarize_shap_importance(shap_df, context):
    if shap_df is None or shap_df.empty:
        return pd.DataFrame()
    rows = []
    for feature in FEATURE_COLUMNS:
        column = f"shap_{feature}"
        if column not in shap_df.columns:
            continue
        values = pd.to_numeric(shap_df[column], errors="coerce").dropna()
        rows.append(
            {
                "context": context,
                "feature": feature,
                "n": int(len(values)),
                "mean_shap": float(values.mean()) if len(values) else np.nan,
                "mean_abs_shap": float(values.abs().mean()) if len(values) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False)


def _build_shap_importance(output_dir, context="global"):
    shap_df = _load_shap_rows(output_dir)
    if shap_df.empty:
        return pd.DataFrame()

    filtered = shap_df
    if context == "interventions":
        if "intervened" in filtered.columns:
            filtered = filtered[filtered["intervened"].astype(str).str.lower() == "true"]
    elif context == "accepted_interventions":
        if "accepted_intervention" in filtered.columns:
            filtered = filtered[
                filtered["accepted_intervention"].astype(str).str.lower() == "true"
            ]
    elif context == "stagnation":
        if "pre_stagnation_length" in filtered.columns:
            stagnation = pd.to_numeric(
                filtered["pre_stagnation_length"], errors="coerce"
            )
            threshold = max(1.0, float(stagnation.quantile(0.75)))
            filtered = filtered[stagnation >= threshold]
        elif "policy_signal" in filtered.columns:
            filtered = filtered[
                filtered["policy_signal"].astype(str).str.contains(
                    "stagnation|diversity", case=False, regex=True
                )
            ]

    return _summarize_shap_importance(filtered, context)


def _plot_shap_importance(shap_importance_df, plots_dir, filename, title):
    if shap_importance_df is None or shap_importance_df.empty:
        return None
    data = shap_importance_df.sort_values("mean_abs_shap", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(data["feature"], data["mean_abs_shap"], color="#8f5d46")
    ax.set_title(title)
    ax.set_xlabel("Mean |SHAP|")
    ax.grid(axis="x", alpha=0.25)
    return _save_figure(fig, plots_dir / filename)


def generate_standard_report(
    output_dir,
    summary_path,
    statistics_path=None,
    base_results_root=None,
    config=None,
):
    output_dir = Path(output_dir)
    summary_path = Path(summary_path)
    values_dir = output_dir / "values"
    plots_dir = output_dir / "plots"
    values_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.read_csv(summary_path)
    paper_stats = build_paper_statistics(summary_df)
    paper_stats_path = values_dir / "paper_statistics.csv"
    paper_stats.to_csv(paper_stats_path, index=False)

    normality_df = build_shapiro_normality(summary_df, "gap_to_optimum")
    normality_path = values_dir / "normality_shapiro_wilk.csv"
    normality_df.to_csv(normality_path, index=False)

    base_summary_path = _find_base_summary(summary_df, base_results_root)
    comparison_df, comparison_summary = build_base_comparison(summary_df, base_summary_path)
    comparison_path = values_dir / "comparison_wilcoxon_mann_whitney.csv"
    comparison_df.to_csv(comparison_path, index=False)
    if comparison_summary is not None:
        (values_dir / "comparison_wtl_summary.json").write_text(
            json.dumps(comparison_summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    shap_importance_df = _build_shap_importance(output_dir, context="global")
    shap_importance_path = values_dir / "shap_importance.csv"
    shap_importance_df.to_csv(shap_importance_path, index=False)
    shap_intervention_importance_df = _build_shap_importance(
        output_dir, context="accepted_interventions"
    )
    shap_intervention_importance_path = values_dir / "shap_importance_interventions.csv"
    shap_intervention_importance_df.to_csv(
        shap_intervention_importance_path, index=False
    )
    shap_stagnation_importance_df = _build_shap_importance(
        output_dir, context="stagnation"
    )
    shap_stagnation_importance_path = values_dir / "shap_importance_stagnation.csv"
    shap_stagnation_importance_df.to_csv(shap_stagnation_importance_path, index=False)

    plot_paths = {
        "gap_boxplot": _plot_gap_boxplot(summary_df, plots_dir),
        "mean_gap": _plot_mean_gap(summary_df, plots_dir),
        "time": _plot_time(summary_df, plots_dir),
        "controller_events": _plot_controller_events(summary_df, plots_dir),
        "shap_importance": _plot_shap_importance(
            shap_importance_df,
            plots_dir,
            "shap_mean_abs_importance.png",
            "Importancia SHAP global promedio absoluta",
        ),
        "shap_interventions": _plot_shap_importance(
            shap_intervention_importance_df,
            plots_dir,
            "shap_mean_abs_importance_interventions.png",
            "Importancia SHAP en intervenciones aceptadas",
        ),
        "shap_stagnation": _plot_shap_importance(
            shap_stagnation_importance_df,
            plots_dir,
            "shap_mean_abs_importance_stagnation.png",
            "Importancia SHAP en ventanas de estancamiento",
        ),
    }
    convergence_paths = _plot_convergence_by_function(
        summary_df, output_dir, summary_path, plots_dir
    )

    validation = {
        "rows": int(len(summary_df)),
        "functions": int(summary_df["function_id"].nunique()),
        "runs": int(summary_df["run_id"].nunique()),
        "negative_gaps": int((summary_df["gap_to_optimum"] < -1e-9).sum()),
        "status_not_completed": int((summary_df.get("status", "") != "completed").sum())
        if "status" in summary_df.columns
        else 0,
        "opfunu_version": getattr(opfunu, "__version__", "not_available"),
    }
    if "elapsed_seconds" in summary_df.columns:
        validation["total_elapsed_seconds"] = float(summary_df["elapsed_seconds"].sum())

    config_payload = dict(config or {})
    first = summary_df.iloc[0]
    config_payload.update(
        {
            "algorithm": str(first.get("algorithm", "")),
            "benchmark": str(first.get("benchmark", "")),
            "dim": int(first.get("dim", 0)),
            "agents": int(first.get("agents", 0)),
            "iterations": int(first.get("iterations", 0)),
            "runs": int(summary_df["run_id"].nunique()),
            "functions": ", ".join(
                sorted(summary_df["function"].unique(), key=_numeric_sort_key)
            ),
        }
    )
    if "controller_profile" in summary_df.columns:
        config_payload["controller_profile"] = str(first.get("controller_profile", ""))

    config_path = values_dir / "report_config.json"
    config_path.write_text(
        json.dumps(config_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    css = """
    body { font-family: Arial, sans-serif; margin: 28px; color: #1f2933; }
    h1, h2, h3 { color: #16324f; }
    h1 { margin-bottom: 4px; }
    .muted { color: #667085; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 14px; }
    .metric { border: 1px solid #d7dde8; border-radius: 6px; padding: 12px; background: #f8fafc; }
    .metric strong { display: block; font-size: 22px; margin-top: 5px; }
    .data-table { border-collapse: collapse; width: 100%; margin: 10px 0 22px; font-size: 13px; }
    .data-table th, .data-table td { border: 1px solid #d7dde8; padding: 6px 8px; text-align: right; }
    .data-table th { background: #eef3f8; color: #16324f; }
    .data-table td:first-child, .data-table th:first-child { text-align: left; }
    img { max-width: 100%; border: 1px solid #d7dde8; border-radius: 6px; background: white; }
    .plot-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 16px; }
    code { background: #eef3f8; padding: 1px 4px; border-radius: 4px; }
    """

    comparison_note = ""
    if comparison_summary is None:
        comparison_note = (
            "<p class=\"muted\">No se encontro un resumen base compatible para "
            "comparar Wilcoxon/Mann-Whitney.</p>"
        )
    else:
        comparison_note = (
            "<div class=\"grid\">"
            f"<div class=\"metric\">SHAP mejor<strong>{comparison_summary['shap_better']}</strong></div>"
            f"<div class=\"metric\">Base mejor<strong>{comparison_summary['base_better']}</strong></div>"
            f"<div class=\"metric\">Empate/no significativo<strong>{comparison_summary['ties_or_not_significant']}</strong></div>"
            "</div>"
        )

    convergence_html = "\n".join(
        f"<section><h3>{escape(path.stem.replace('convergence_', ''))}</h3>"
        f"{_image_tag(path, path.stem)}</section>"
        for path in convergence_paths
    )

    main_plots_html = "\n".join(
        _image_tag(path, name)
        for name, path in plot_paths.items()
        if path is not None and Path(path).exists()
    )

    html = f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <title>Reporte estandarizado WO</title>
  <style>{css}</style>
</head>
<body>
  <h1>Reporte estandarizado WO</h1>
  <p class="muted">Generado desde <code>{escape(str(summary_path))}</code></p>

  <h2>Configuracion</h2>
  {_html_table(pd.DataFrame([config_payload]))}

  <h2>Validacion</h2>
  {_html_table(pd.DataFrame([validation]))}

  <h2>Metricas estilo paper</h2>
  <p class="muted">Best, Worst, Avg/Mean, Std y Time por funcion.</p>
  {_html_table(paper_stats)}

  <h2>Normalidad Shapiro-Wilk</h2>
  <p class="muted">Aplicado sobre <code>gap_to_optimum</code>. Si p &gt; 0.05, no se rechaza normalidad.</p>
  {_html_table(normality_df)}

  <h2>Wilcoxon / Mann-Whitney contra WO base</h2>
  {comparison_note}
  {_html_table(comparison_df)}

  <h2>Graficos principales</h2>
  <div class="plot-grid">{main_plots_html}</div>

  <h2>Curvas de convergencia por funcion</h2>
  <div class="plot-grid">{convergence_html}</div>

  <h2>Importancia SHAP</h2>
  <h3>Global</h3>
  {_html_table(shap_importance_df)}
  <h3>Intervenciones aceptadas</h3>
  <p class="muted">Filtra explicaciones asociadas a intervenciones aceptadas por la compuerta de aceptacion.</p>
  {_html_table(shap_intervention_importance_df)}
  <h3>Ventanas de estancamiento</h3>
  <p class="muted">Filtra explicaciones con longitud de estancamiento en el cuartil superior disponible.</p>
  {_html_table(shap_stagnation_importance_df)}
</body>
</html>
"""

    report_path = output_dir / "report.html"
    report_path.write_text(html, encoding="utf-8")

    visual_report_path = None
    try:
        from visual_dashboard import generate_visual_dashboard

        visual_report_path = generate_visual_dashboard(output_dir)
    except Exception as exc:  # pragma: no cover - reporting should not hide run results
        (values_dir / "visual_report_error.txt").write_text(str(exc), encoding="utf-8")

    return {
        "report_path": report_path,
        "visual_report_path": visual_report_path,
        "paper_statistics_path": paper_stats_path,
        "normality_path": normality_path,
        "comparison_path": comparison_path,
        "shap_importance_path": shap_importance_path,
        "shap_intervention_importance_path": shap_intervention_importance_path,
        "shap_stagnation_importance_path": shap_stagnation_importance_path,
        "base_summary_path": base_summary_path,
        "validation": validation,
    }
