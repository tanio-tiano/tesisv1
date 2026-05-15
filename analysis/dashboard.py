from __future__ import annotations

import argparse
import base64
import json
import math
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError


def _read_csv(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def _find_summary(values_dir):
    candidates = sorted(
        values_dir.glob("summary_*.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No se encontro summary_*.csv en {values_dir}")
    return candidates[0]


def _function_key(label):
    digits = "".join(ch for ch in str(label) if ch.isdigit())
    return int(digits) if digits else 9999


def _fmt(value, digits=4):
    if value is None:
        return "NA"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return escape(str(value))
    if not math.isfinite(value):
        return "NA"
    abs_value = abs(value)
    if abs_value == 0:
        return "0"
    if abs_value >= 100000 or abs_value < 0.001:
        return f"{value:.3e}"
    if abs_value >= 100:
        return f"{value:.2f}"
    return f"{value:.{digits}f}"


def _image_data_uri(path):
    path = Path(path)
    if not path.exists():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _badge(text, kind="neutral"):
    return f'<span class="badge {kind}">{escape(str(text))}</span>'


def _outcome_badge(outcome):
    mapping = {
        "SHAP_better": ("SHAP mejor", "good"),
        "BASE_better": ("Base mejor", "bad"),
        "Tie_or_not_significant": ("No significativo", "neutral"),
    }
    text, kind = mapping.get(str(outcome), (str(outcome), "neutral"))
    return _badge(text, kind)


def _normal_badge(value):
    if bool(value):
        return _badge("Normal", "good")
    return _badge("No normal", "warn")


def _html_table(df, columns, classes=None):
    if df.empty:
        return '<p class="empty">No hay datos disponibles.</p>'
    classes = classes or {}
    rows = []
    for _, row in df.iterrows():
        cells = []
        for header, key, formatter in columns:
            value = row.get(key, "")
            rendered = formatter(value, row) if formatter else escape(str(value))
            css = classes.get(key, "")
            cells.append(f'<td class="{css}">{rendered}</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    headers = "".join(f"<th>{escape(header)}</th>" for header, _, _ in columns)
    return (
        '<div class="table-wrap"><table class="data-table">'
        f"<thead><tr>{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def _main_plot_card(title, image_uri, note=""):
    if not image_uri:
        body = '<p class="empty">Grafico no disponible.</p>'
    else:
        body = f'<img class="plot" src="{image_uri}" alt="{escape(title)}">'
    return (
        '<section class="panel">'
        f"<h3>{escape(title)}</h3>"
        f"{body}"
        f'<p class="caption">{escape(note)}</p>' if note else ""
        "</section>"
    )


def _build_matrix(summary_df, paper_df, normal_df, comparison_df):
    events = pd.DataFrame()
    if {"function", "function_id"}.issubset(summary_df.columns):
        event_cols = [
            col
            for col in [
                "control_events",
                "accepted_control_events",
                "rejected_control_events",
                "effective_control_events",
            ]
            if col in summary_df.columns
        ]
        if event_cols:
            events = (
                summary_df.groupby(["function_id", "function"], as_index=False)[event_cols]
                .mean()
                .rename(
                    columns={
                        "control_events": "events_mean",
                        "accepted_control_events": "accepted_mean",
                        "rejected_control_events": "rejected_mean",
                        "effective_control_events": "effective_mean",
                    }
                )
            )

    matrix = paper_df.copy()
    if not normal_df.empty:
        matrix = matrix.merge(
            normal_df[
                [
                    "function_id",
                    "shapiro_p",
                    "normal_alpha_0_05",
                    "mean",
                    "std",
                ]
            ].rename(columns={"mean": "gap_mean", "std": "gap_std"}),
            on="function_id",
            how="left",
        )
    if not comparison_df.empty:
        matrix = matrix.merge(
            comparison_df[
                [
                    "function_id",
                    "mean_gap_base",
                    "delta_gap_shap_minus_base",
                    "wilcoxon_signed_rank_p",
                    "mann_whitney_u_p",
                    "outcome",
                ]
            ],
            on="function_id",
            how="left",
        )
    if not events.empty:
        matrix = matrix.merge(events.drop(columns=["function"]), on="function_id", how="left")
    return matrix.sort_values("function_id")


def generate_visual_dashboard(output_dir):
    output_dir = Path(output_dir)
    values_dir = output_dir / "values"
    plots_dir = output_dir / "plots"

    summary_path = _find_summary(values_dir)
    summary_df = _read_csv(summary_path)
    paper_df = _read_csv(values_dir / "paper_statistics.csv")
    normal_df = _read_csv(values_dir / "normality_shapiro_wilk.csv")
    comparison_df = _read_csv(values_dir / "comparison_wilcoxon_mann_whitney.csv")
    shap_df = _read_csv(values_dir / "shap_importance.csv")
    shap_intervention_df = _read_csv(values_dir / "shap_importance_interventions.csv")
    shap_stagnation_df = _read_csv(values_dir / "shap_importance_stagnation.csv")

    if paper_df.empty:
        raise FileNotFoundError("No se encontro values/paper_statistics.csv")

    matrix_df = _build_matrix(summary_df, paper_df, normal_df, comparison_df)

    first = summary_df.iloc[0] if not summary_df.empty else pd.Series(dtype=object)
    algorithm = str(first.get("algorithm", "WO_SHAP"))
    profile = str(first.get("controller_profile", ""))
    benchmark = str(first.get("benchmark", ""))
    runs = int(summary_df["run_id"].nunique()) if "run_id" in summary_df else 0
    functions = int(summary_df["function_id"].nunique()) if "function_id" in summary_df else 0
    rows = int(len(summary_df))
    negative_gaps = (
        int((pd.to_numeric(summary_df["gap_to_optimum"], errors="coerce") < -1e-9).sum())
        if "gap_to_optimum" in summary_df
        else 0
    )

    normal_count = (
        int(normal_df["normal_alpha_0_05"].astype(bool).sum())
        if "normal_alpha_0_05" in normal_df
        else 0
    )
    shap_better = (
        int((comparison_df["outcome"] == "SHAP_better").sum())
        if "outcome" in comparison_df
        else 0
    )
    base_better = (
        int((comparison_df["outcome"] == "BASE_better").sum())
        if "outcome" in comparison_df
        else 0
    )
    ties = (
        int((comparison_df["outcome"] == "Tie_or_not_significant").sum())
        if "outcome" in comparison_df
        else 0
    )
    total_events = int(summary_df.get("control_events", pd.Series(dtype=float)).sum())
    accepted_events = int(
        summary_df.get("accepted_control_events", pd.Series(dtype=float)).sum()
    )
    rejected_events = int(
        summary_df.get("rejected_control_events", pd.Series(dtype=float)).sum()
    )
    effective_events = int(
        summary_df.get("effective_control_events", pd.Series(dtype=float)).sum()
    )

    normal_functions = (
        ", ".join(normal_df.loc[normal_df["normal_alpha_0_05"].astype(bool), "function"])
        if "normal_alpha_0_05" in normal_df
        else ""
    )
    shap_better_functions = (
        ", ".join(comparison_df.loc[comparison_df["outcome"] == "SHAP_better", "function"])
        if "outcome" in comparison_df
        else ""
    )
    base_better_functions = (
        ", ".join(comparison_df.loc[comparison_df["outcome"] == "BASE_better", "function"])
        if "outcome" in comparison_df
        else ""
    )

    main_images = {
        "mean_gap": _image_data_uri(plots_dir / "mean_gap_to_optimum.png"),
        "boxplot": _image_data_uri(plots_dir / "boxplot_gap_to_optimum.png"),
        "events": _image_data_uri(plots_dir / "controller_events_by_function.png"),
        "time": _image_data_uri(plots_dir / "time_mean_by_function.png"),
        "shap": _image_data_uri(plots_dir / "shap_mean_abs_importance.png"),
        "shap_interventions": _image_data_uri(
            plots_dir / "shap_mean_abs_importance_interventions.png"
        ),
        "shap_stagnation": _image_data_uri(
            plots_dir / "shap_mean_abs_importance_stagnation.png"
        ),
    }

    convergence_images = {}
    for path in sorted(plots_dir.glob("convergence_F*.png"), key=lambda p: _function_key(p.stem)):
        function = path.stem.replace("convergence_", "")
        convergence_images[function] = _image_data_uri(path)

    cards = [
        ("Corridas", str(runs), f"{functions} funciones, {rows} filas"),
        ("Gaps negativos", str(negative_gaps), "Debe ser 0"),
        ("Normalidad", f"{normal_count}/{functions}", normal_functions or "Sin funciones normales"),
        ("Comparacion", f"{shap_better}-{ties}-{base_better}", "SHAP / empate / base"),
        ("Eventos SHAP", str(total_events), f"{accepted_events} aceptados, {rejected_events} rechazados"),
        ("Efectivos", str(effective_events), "Mejoras post-intervencion"),
    ]
    cards_html = "".join(
        '<section class="metric">'
        f"<span>{escape(title)}</span><strong>{escape(value)}</strong>"
        f"<small>{escape(note)}</small></section>"
        for title, value, note in cards
    )

    quick_html = (
        '<section class="insight good"><h3>Funciones con normalidad</h3>'
        f"<p>{escape(normal_functions or 'Ninguna')}</p></section>"
        '<section class="insight neutral"><h3>SHAP mejor significativo</h3>'
        f"<p>{escape(shap_better_functions or 'Ninguna')}</p></section>"
        '<section class="insight bad"><h3>Base mejor significativo</h3>'
        f"<p>{escape(base_better_functions or 'Ninguna')}</p></section>"
    )

    matrix_columns = [
        ("Funcion", "function", lambda v, r: f"<strong>{escape(str(v))}</strong>"),
        ("Best", "best", lambda v, r: _fmt(v)),
        ("Worst", "worst", lambda v, r: _fmt(v)),
        ("Avg/Mean", "avg_mean", lambda v, r: _fmt(v)),
        ("Std", "std", lambda v, r: _fmt(v)),
        ("Time s", "time_mean_seconds", lambda v, r: _fmt(v, 3)),
        ("Gap medio", "gap_mean", lambda v, r: _fmt(v)),
        ("Shapiro", "normal_alpha_0_05", lambda v, r: _normal_badge(v)),
        ("Shapiro p", "shapiro_p", lambda v, r: _fmt(v, 3)),
        ("Vs base", "outcome", lambda v, r: _outcome_badge(v)),
        ("Wilcoxon p", "wilcoxon_signed_rank_p", lambda v, r: _fmt(v, 3)),
        ("MWU p", "mann_whitney_u_p", lambda v, r: _fmt(v, 3)),
        ("Eventos", "events_mean", lambda v, r: _fmt(v, 2)),
        ("Aceptados", "accepted_mean", lambda v, r: _fmt(v, 2)),
    ]

    paper_columns = [
        ("Funcion", "function", lambda v, r: f"<strong>{escape(str(v))}</strong>"),
        ("Best", "best", lambda v, r: _fmt(v)),
        ("Worst", "worst", lambda v, r: _fmt(v)),
        ("Avg/Mean", "avg_mean", lambda v, r: _fmt(v)),
        ("Std", "std", lambda v, r: _fmt(v)),
        ("Time s", "time_mean_seconds", lambda v, r: _fmt(v, 3)),
    ]

    normal_columns = [
        ("Funcion", "function", lambda v, r: f"<strong>{escape(str(v))}</strong>"),
        ("n", "n", lambda v, r: _fmt(v, 0)),
        ("Mean gap", "mean", lambda v, r: _fmt(v)),
        ("Std gap", "std", lambda v, r: _fmt(v)),
        ("Shapiro stat", "shapiro_statistic", lambda v, r: _fmt(v, 4)),
        ("Shapiro p", "shapiro_p", lambda v, r: _fmt(v, 3)),
        ("Decision", "normal_alpha_0_05", lambda v, r: _normal_badge(v)),
    ]

    comparison_columns = [
        ("Funcion", "function", lambda v, r: f"<strong>{escape(str(v))}</strong>"),
        ("Mean gap SHAP", "mean_gap_shap", lambda v, r: _fmt(v)),
        ("Mean gap base", "mean_gap_base", lambda v, r: _fmt(v)),
        ("Delta SHAP-base", "delta_gap_shap_minus_base", lambda v, r: _fmt(v)),
        ("Wilcoxon p", "wilcoxon_signed_rank_p", lambda v, r: _fmt(v, 3)),
        ("Mann-Whitney p", "mann_whitney_u_p", lambda v, r: _fmt(v, 3)),
        ("Resultado", "outcome", lambda v, r: _outcome_badge(v)),
    ]

    shap_columns = [
        ("Variable", "feature", lambda v, r: f"<strong>{escape(str(v))}</strong>"),
        ("n", "n", lambda v, r: _fmt(v, 0)),
        ("Mean SHAP", "mean_shap", lambda v, r: _fmt(v)),
        ("Mean |SHAP|", "mean_abs_shap", lambda v, r: _fmt(v)),
    ]
    paper_display = paper_df.sort_values("function_id") if "function_id" in paper_df else paper_df
    normal_display = normal_df.sort_values("function_id") if "function_id" in normal_df else normal_df
    comparison_display = (
        comparison_df.sort_values("function_id")
        if "function_id" in comparison_df
        else comparison_df
    )

    conv_options = "".join(
        f'<option value="{escape(function)}">{escape(function)}</option>'
        for function in sorted(convergence_images, key=_function_key)
    )
    conv_json = json.dumps(convergence_images)

    css = """
    :root {
      --ink: #172033;
      --muted: #657083;
      --line: #d9e0ea;
      --bg: #f4f6f9;
      --panel: #ffffff;
      --blue: #2f5f9f;
      --green: #1f7a4d;
      --red: #a4453f;
      --amber: #9a6a18;
      --slate: #46556a;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Arial, Helvetica, sans-serif;
      font-size: 14px;
    }
    header {
      background: #ffffff;
      border-bottom: 1px solid var(--line);
      padding: 18px 26px 14px;
      position: sticky;
      top: 0;
      z-index: 10;
    }
    h1 { margin: 0 0 6px; font-size: 24px; letter-spacing: 0; }
    h2 { margin: 0 0 14px; font-size: 19px; letter-spacing: 0; }
    h3 { margin: 0 0 10px; font-size: 15px; letter-spacing: 0; }
    p { line-height: 1.45; }
    .subtitle { color: var(--muted); margin: 0; }
    .nav {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 14px;
    }
    .nav button {
      border: 1px solid var(--line);
      background: #ffffff;
      color: var(--ink);
      padding: 7px 10px;
      border-radius: 6px;
      cursor: pointer;
      font-size: 13px;
    }
    .nav button.active {
      background: var(--blue);
      border-color: var(--blue);
      color: white;
    }
    main { padding: 22px 26px 34px; max-width: 1480px; margin: 0 auto; }
    .section { display: none; }
    .section.active { display: block; }
    .metrics {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin-bottom: 16px;
    }
    .metric, .panel, .insight {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 14px;
    }
    .metric span, .metric small { display: block; color: var(--muted); }
    .metric strong {
      display: block;
      margin: 6px 0 4px;
      font-size: 26px;
      letter-spacing: 0;
    }
    .insights {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 12px;
      margin-bottom: 16px;
    }
    .insight.good { border-left: 4px solid var(--green); }
    .insight.bad { border-left: 4px solid var(--red); }
    .insight.neutral { border-left: 4px solid var(--slate); }
    .insight p { margin: 0; color: var(--ink); }
    .plot-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(430px, 1fr));
      gap: 14px;
    }
    .plot {
      display: block;
      width: 100%;
      max-height: 520px;
      object-fit: contain;
      background: white;
      border: 1px solid var(--line);
      border-radius: 6px;
    }
    .caption { margin: 8px 0 0; color: var(--muted); font-size: 12px; }
    .table-wrap {
      overflow-x: auto;
      background: white;
      border: 1px solid var(--line);
      border-radius: 6px;
    }
    .data-table {
      border-collapse: collapse;
      width: 100%;
      min-width: 980px;
      font-size: 13px;
    }
    .data-table th, .data-table td {
      border-bottom: 1px solid #e7ecf2;
      padding: 8px 9px;
      text-align: right;
      white-space: nowrap;
    }
    .data-table th {
      background: #eef2f6;
      color: #253348;
      position: sticky;
      top: 0;
      z-index: 1;
    }
    .data-table th:first-child, .data-table td:first-child { text-align: left; }
    .badge {
      display: inline-block;
      padding: 3px 8px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
    }
    .badge.good { color: #125b39; background: #dff3e8; }
    .badge.bad { color: #822b28; background: #f8dddd; }
    .badge.warn { color: #7a520f; background: #f7ead1; }
    .badge.neutral { color: #435166; background: #e8edf3; }
    .controls {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 10px;
      margin-bottom: 12px;
    }
    select {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 7px 10px;
      background: white;
      color: var(--ink);
      font-size: 14px;
    }
    .convergence-img {
      width: 100%;
      max-height: 720px;
      object-fit: contain;
      background: white;
      border: 1px solid var(--line);
      border-radius: 6px;
    }
    .file-links {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(290px, 1fr));
      gap: 10px;
    }
    .file-links a {
      display: block;
      background: white;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
      color: var(--blue);
      text-decoration: none;
      overflow-wrap: anywhere;
    }
    .empty { color: var(--muted); margin: 0; }
    code {
      background: #e9eef5;
      border-radius: 4px;
      padding: 1px 4px;
    }
    @media (max-width: 720px) {
      header { padding: 14px 16px; position: static; }
      main { padding: 16px; }
      .plot-grid { grid-template-columns: 1fr; }
      .metric strong { font-size: 22px; }
    }
    """

    html = f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Reporte visual WO SHAP</title>
  <style>{css}</style>
</head>
<body>
  <header>
    <h1>Reporte visual WO + SHAP</h1>
    <p class="subtitle">{escape(algorithm)} {escape(profile)} - {escape(benchmark)} - {runs} corridas - {functions} funciones</p>
    <nav class="nav">
      <button class="active" data-section="overview">Resumen</button>
      <button data-section="matrix">Matriz por funcion</button>
      <button data-section="paper">Metricas paper</button>
      <button data-section="normality">Shapiro-Wilk</button>
      <button data-section="comparison">Wilcoxon / Mann-Whitney</button>
      <button data-section="convergence">Convergencia</button>
      <button data-section="shap">SHAP</button>
      <button data-section="files">Archivos</button>
    </nav>
  </header>
  <main>
    <section id="overview" class="section active">
      <div class="metrics">{cards_html}</div>
      <div class="insights">{quick_html}</div>
      <div class="plot-grid">
        {_main_plot_card("Gap promedio al optimo", main_images["mean_gap"], "Lectura rapida del error medio por funcion.")}
        {_main_plot_card("Boxplots de gap al optimo", main_images["boxplot"], "Dispersion de las 30 corridas por funcion.")}
        {_main_plot_card("Eventos del controlador", main_images["events"], "Promedio de propuestas, aceptaciones, rechazos y eventos efectivos.")}
        {_main_plot_card("Tiempo promedio por funcion", main_images["time"], "Costo computacional medio por funcion.")}
      </div>
    </section>

    <section id="matrix" class="section">
      <h2>Matriz por funcion</h2>
      {_html_table(matrix_df, matrix_columns)}
    </section>

    <section id="paper" class="section">
      <h2>Metricas estilo paper</h2>
      {_html_table(paper_display, paper_columns)}
    </section>

    <section id="normality" class="section">
      <h2>Test de normalidad Shapiro-Wilk</h2>
      {_html_table(normal_display, normal_columns)}
    </section>

    <section id="comparison" class="section">
      <h2>Wilcoxon signed-rank y Mann-Whitney U</h2>
      {_html_table(comparison_display, comparison_columns)}
    </section>

    <section id="convergence" class="section">
      <h2>Curvas de convergencia por funcion</h2>
      <div class="panel">
        <div class="controls">
          <label for="functionSelect">Funcion</label>
          <select id="functionSelect">{conv_options}</select>
        </div>
        <img id="convergenceImage" class="convergence-img" alt="Curva de convergencia">
      </div>
    </section>

    <section id="shap" class="section">
      <h2>SHAP</h2>
      <div class="plot-grid">
        {_main_plot_card("Importancia SHAP global", main_images["shap"], "Variables internas que mas influyeron en el surrogate del fitness.")}
        {_main_plot_card("SHAP en intervenciones aceptadas", main_images["shap_interventions"], "Variables mas influyentes cuando una accion paso la compuerta de aceptacion.")}
        {_main_plot_card("SHAP en estancamiento", main_images["shap_stagnation"], "Variables mas influyentes en el cuartil superior de estancamiento observado.")}
      </div>
      <h3>Global</h3>
      {_html_table(shap_df, shap_columns)}
      <h3>Intervenciones aceptadas</h3>
      {_html_table(shap_intervention_df, shap_columns)}
      <h3>Ventanas de estancamiento</h3>
      {_html_table(shap_stagnation_df, shap_columns)}
    </section>

    <section id="files" class="section">
      <h2>Archivos fuente</h2>
      <div class="file-links">
        <a href="values/{escape(summary_path.name)}">summary completo</a>
        <a href="values/paper_statistics.csv">paper_statistics.csv</a>
        <a href="values/normality_shapiro_wilk.csv">normality_shapiro_wilk.csv</a>
        <a href="values/comparison_wilcoxon_mann_whitney.csv">comparison_wilcoxon_mann_whitney.csv</a>
        <a href="values/shap_importance.csv">shap_importance.csv</a>
        <a href="values/shap_importance_interventions.csv">shap_importance_interventions.csv</a>
        <a href="values/shap_importance_stagnation.csv">shap_importance_stagnation.csv</a>
        <a href="report.html">report.html original</a>
      </div>
    </section>
  </main>
  <script>
    const convergenceImages = {conv_json};
    const buttons = document.querySelectorAll('.nav button');
    const sections = document.querySelectorAll('.section');
    buttons.forEach((button) => {{
      button.addEventListener('click', () => {{
        buttons.forEach((item) => item.classList.remove('active'));
        sections.forEach((item) => item.classList.remove('active'));
        button.classList.add('active');
        document.getElementById(button.dataset.section).classList.add('active');
      }});
    }});

    const select = document.getElementById('functionSelect');
    const image = document.getElementById('convergenceImage');
    function updateConvergence() {{
      const key = select.value;
      image.src = convergenceImages[key] || '';
      image.alt = 'Curva de convergencia ' + key;
    }}
    select.addEventListener('change', updateConvergence);
    updateConvergence();
  </script>
</body>
</html>
"""

    dashboard_path = output_dir / "reporte_visual.html"
    dashboard_path.write_text(html, encoding="utf-8")
    return dashboard_path


def main():
    parser = argparse.ArgumentParser(description="Genera un dashboard HTML visual.")
    parser.add_argument(
        "--output",
        default="outputs_shap_soft_30runs_standard_report",
        help="Carpeta de resultados del experimento.",
    )
    args = parser.parse_args()
    path = generate_visual_dashboard(args.output)
    print(path)


if __name__ == "__main__":
    main()
