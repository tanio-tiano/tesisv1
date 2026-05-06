import argparse
import html
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analiza normalidad de resultados WO base CEC 2022."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=(
            "all_functions_outputs_30runs_legacy_20260429/values/"
            "summary_wo_base_cec2022_30x500_30runs_legacy.csv"
        ),
        help="CSV resumen con una fila por corrida y funcion.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="final_fitness",
        help="Columna numerica a evaluar. Recomendado: final_fitness.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Nivel de significancia para rechazar normalidad.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="normality_analysis_legacy_20260429",
        help="Directorio de salida.",
    )
    return parser.parse_args()


def _safe_test(func, values, min_n):
    if values.size < min_n:
        return np.nan, np.nan
    if np.isclose(np.std(values, ddof=1), 0.0):
        return np.nan, np.nan
    try:
        result = func(values)
    except Exception:
        return np.nan, np.nan
    if hasattr(result, "statistic") and hasattr(result, "pvalue"):
        return float(result.statistic), float(result.pvalue)
    return float(result[0]), float(result[1])


def anderson_normal(values, alpha):
    if values.size < 3 or np.isclose(np.std(values, ddof=1), 0.0):
        return np.nan, np.nan, np.nan, False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        result = stats.anderson(values, dist="norm")
    significance = np.asarray(result.significance_level, dtype=float)
    critical = np.asarray(result.critical_values, dtype=float)
    target = alpha * 100.0
    index = int(np.argmin(np.abs(significance - target)))
    critical_value = float(critical[index])
    level = float(significance[index])
    statistic = float(result.statistic)
    return statistic, critical_value, level, bool(statistic < critical_value)


def decision_from_tests(row, alpha):
    shapiro_ok = bool(row["shapiro_p"] >= alpha) if pd.notna(row["shapiro_p"]) else None
    anderson_ok = (
        bool(row["anderson_normal_at_alpha"])
        if pd.notna(row["anderson_statistic"])
        else None
    )
    dagostino_ok = (
        bool(row["dagostino_p"] >= alpha) if pd.notna(row["dagostino_p"]) else None
    )

    available = [value for value in (shapiro_ok, anderson_ok, dagostino_ok) if value is not None]
    if not available:
        return "SIN_DECISION"
    if all(available):
        return "COMPATIBLE_NORMAL"
    if not shapiro_ok or not anderson_ok:
        return "RECHAZA_NORMALIDAD"
    return "MIXTO"


def analyze(summary_df, metric, alpha):
    rows = []
    for function_id, group in summary_df.groupby("function_id", sort=True):
        values = group[metric].astype(float).to_numpy()
        first = group.iloc[0]

        shapiro_w, shapiro_p = _safe_test(stats.shapiro, values, min_n=3)
        dagostino_k2, dagostino_p = _safe_test(stats.normaltest, values, min_n=8)
        jarque_bera_stat, jarque_bera_p = _safe_test(stats.jarque_bera, values, min_n=3)
        anderson_stat, anderson_crit, anderson_level, anderson_ok = anderson_normal(
            values, alpha
        )

        row = {
            "function": first["function"],
            "function_id": int(function_id),
            "function_name": first.get("function_name", ""),
            "family": first.get("function_family", ""),
            "metric": metric,
            "n": int(values.size),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values, ddof=1)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "skewness": float(stats.skew(values, bias=False)),
            "excess_kurtosis": float(stats.kurtosis(values, fisher=True, bias=False)),
            "shapiro_w": shapiro_w,
            "shapiro_p": shapiro_p,
            "shapiro_reject_normal": bool(shapiro_p < alpha)
            if pd.notna(shapiro_p)
            else np.nan,
            "dagostino_k2": dagostino_k2,
            "dagostino_p": dagostino_p,
            "dagostino_reject_normal": bool(dagostino_p < alpha)
            if pd.notna(dagostino_p)
            else np.nan,
            "jarque_bera_statistic": jarque_bera_stat,
            "jarque_bera_p": jarque_bera_p,
            "jarque_bera_reject_normal": bool(jarque_bera_p < alpha)
            if pd.notna(jarque_bera_p)
            else np.nan,
            "anderson_statistic": anderson_stat,
            "anderson_critical_value": anderson_crit,
            "anderson_significance_level_pct": anderson_level,
            "anderson_normal_at_alpha": anderson_ok
            if pd.notna(anderson_stat)
            else np.nan,
        }
        row["decision"] = decision_from_tests(row, alpha)
        rows.append(row)
    return pd.DataFrame(rows)


def format_float(value):
    if pd.isna(value):
        return ""
    return f"{float(value):.6g}"


def html_table(df):
    display = df.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(format_float)
    return display.to_html(index=False, escape=True, classes="results")


def write_html_report(report_path, tests_df, summary_df, args, input_path):
    reject_count = int((tests_df["decision"] == "RECHAZA_NORMALIDAD").sum())
    compatible_count = int((tests_df["decision"] == "COMPATIBLE_NORMAL").sum())
    mixed_count = int((tests_df["decision"] == "MIXTO").sum())
    total = int(len(tests_df))

    recommendation = (
        "No asumir normalidad global; usar pruebas no parametricas para comparaciones "
        "generales, o validar normalidad de las diferencias pareadas si luego comparas "
        "WO base contra otra variante."
        if reject_count > 0
        else "Los datos son compatibles con normalidad en todas las funciones bajo estos tests."
    )

    selected_cols = [
        "function",
        "function_name",
        "family",
        "n",
        "mean",
        "std",
        "skewness",
        "excess_kurtosis",
        "shapiro_p",
        "dagostino_p",
        "jarque_bera_p",
        "anderson_statistic",
        "anderson_critical_value",
        "decision",
    ]
    table_html = html_table(tests_df[selected_cols])

    source = html.escape(str(input_path.resolve()))
    metric = html.escape(args.metric)
    alpha = html.escape(str(args.alpha))

    html_text = f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <title>Normalidad WO base CEC 2022</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #1f2933; }}
    h1 {{ margin-bottom: 4px; }}
    .muted {{ color: #64748b; }}
    .cards {{ display: flex; gap: 12px; flex-wrap: wrap; margin: 18px 0; }}
    .card {{ border: 1px solid #d8dee9; border-radius: 6px; padding: 12px 14px; min-width: 150px; }}
    .value {{ font-size: 24px; font-weight: 700; }}
    table.results {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    table.results th, table.results td {{ border: 1px solid #d8dee9; padding: 7px; text-align: left; }}
    table.results th {{ background: #f1f5f9; }}
    code {{ background: #f1f5f9; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>Analisis de normalidad - WO base CEC 2022</h1>
  <p class="muted">Fuente: <code>{source}</code></p>
  <p>Variable analizada: <code>{metric}</code>. Nivel de significancia: <code>{alpha}</code>.</p>

  <div class="cards">
    <div class="card"><div class="value">{total}</div><div>Funciones evaluadas</div></div>
    <div class="card"><div class="value">{compatible_count}</div><div>Compatibles con normalidad</div></div>
    <div class="card"><div class="value">{reject_count}</div><div>Rechazan normalidad</div></div>
    <div class="card"><div class="value">{mixed_count}</div><div>Resultado mixto</div></div>
  </div>

  <h2>Conclusion</h2>
  <p>{html.escape(recommendation)}</p>
  <p>Para cada funcion hay 30 corridas. Shapiro-Wilk se toma como referencia principal por el tamano muestral; Anderson-Darling se usa como contraste complementario.</p>

  <h2>Resultados por funcion</h2>
  {table_html}

  <h2>Notas</h2>
  <p>El <code>gap_to_optimum</code> es una traslacion de <code>final_fitness</code> dentro de cada funcion, por lo que conserva la misma normalidad. Se reporta <code>final_fitness</code> porque es la metrica directa del algoritmo.</p>
</body>
</html>
"""
    report_path.write_text(html_text, encoding="utf-8")


def main():
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = Path.cwd() / input_path
    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.read_csv(input_path)
    if args.metric not in summary_df.columns:
        raise ValueError(f"No existe la columna {args.metric} en {input_path}")

    tests_df = analyze(summary_df, args.metric, args.alpha)
    tests_path = output_dir / "normality_tests_wo_base_cec2022_30runs_legacy.csv"
    report_path = output_dir / "normality_report_wo_base_cec2022_30runs_legacy.html"
    tests_df.to_csv(tests_path, index=False)
    write_html_report(report_path, tests_df, summary_df, args, input_path)

    print(f"Tests: {tests_path}")
    print(f"HTML: {report_path}")
    print(tests_df[["function", "shapiro_p", "anderson_normal_at_alpha", "decision"]])


if __name__ == "__main__":
    main()
