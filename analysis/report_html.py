"""Reporte HTML comparativo WO base vs WO+SHAP (regimen MaxFES).

Lee las carpetas de salida de los runners (``values/summary.csv`` + ``curves/``,
y ``values/controller_events.csv`` para el SHAP) y produce un HTML autocontenido:

- Tabla comparativa por problema (media +/- std, mejor, gap; ganador pareado).
- Actividad del controlador (intervenciones, costo SHAP en FES, split de ramas
  reinit_random / reinit_guided, feature dominante).
- Graficos de convergencia (media sobre corridas) base vs SHAP por problema.

Asume el arbol que crea el script de pruebas:

    <input>/cec/{base,shap}
    <input>/tmlap_simple/{base,shap}
    <input>/tmlap_mediana/{base,shap}
    <input>/tmlap_dura/{base,shap}

Uso:
    python -m analysis.report_html --input experiments/test_report \\
        --output experiments/test_report/report.html
"""

from __future__ import annotations

import argparse
import base64
import io
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


GROUPS = [
    ("CEC2022", "cec"),
    ("TMLAP simple", "tmlap_simple"),
    ("TMLAP mediana", "tmlap_mediana"),
    ("TMLAP dura", "tmlap_dura"),
]


def _read_csv(path):
    path = Path(path)
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _problem_key(label):
    m = re.match(r"F(\d+)", str(label))
    return (0, int(m.group(1))) if m else (1, str(label))


def _fmt(value, sig=4):
    if value is None:
        return "&mdash;"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "&mdash;"
    if not np.isfinite(value):
        return "&mdash;"
    return f"{value:.{sig}g}"


def _load_curves(curves_dir, label):
    out = []
    cdir = Path(curves_dir)
    if not cdir.exists():
        return out
    for f in sorted(cdir.glob(f"conv_curve_{label}_fes*_run*.csv")):
        df = pd.read_csv(f)
        if "fes" in df.columns and "best_fitness" in df.columns and len(df) >= 2:
            out.append((df["fes"].to_numpy(float), df["best_fitness"].to_numpy(float)))
    return out


def _mean_curve(curves, max_fes, n=250):
    grid = np.linspace(0, max_fes, n)
    ys = [np.interp(grid, fes, best, left=best[0], right=best[-1]) for fes, best in curves]
    return (grid, np.mean(ys, axis=0)) if ys else (None, None)


def _paired_counts(base_df, shap_df, problem, tol=1e-8):
    b = base_df[base_df["problem"] == problem].set_index("run_id")["final_fitness"]
    s = shap_df[shap_df["problem"] == problem].set_index("run_id")["final_fitness"]
    common = b.index.intersection(s.index)
    if len(common) == 0:
        return 0, 0, 0
    b, s = b.loc[common], s.loc[common]
    wins = int((s < b - tol).sum())     # SHAP mejor (minimizacion)
    losses = int((s > b + tol).sum())   # SHAP peor
    return wins, len(common) - wins - losses, losses


def _stats(df, problem, column):
    sub = df[df["problem"] == problem][column].astype(float)
    sub = sub[np.isfinite(sub)]
    if sub.empty:
        return {"mean": None, "std": None, "best": None, "median": None}
    return {
        "mean": float(sub.mean()),
        "std": float(sub.std(ddof=1)) if len(sub) > 1 else 0.0,
        "best": float(sub.min()),
        "median": float(sub.median()),
    }


def _png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=96, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _convergence_figure(title, problems, base_dir, shap_dir, base_df, shap_df, max_fes):
    n = len(problems)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.0 * rows), squeeze=False)
    for idx, prob in enumerate(problems):
        ax = axes[idx // cols][idx % cols]
        opt = np.nan
        if not shap_df.empty and prob in set(shap_df["problem"]):
            opt = float(shap_df[shap_df["problem"] == prob]["optimum"].iloc[0])
        for dirpath, df, color, name in (
            (base_dir, base_df, "#1f77b4", "base"),
            (shap_dir, shap_df, "#ff7f0e", "SHAP"),
        ):
            curves = _load_curves(Path(dirpath) / "curves", prob)
            grid, mean = _mean_curve(curves, max_fes)
            if grid is None:
                continue
            y = mean - opt if np.isfinite(opt) else mean
            if np.isfinite(opt) and np.all(y > 0):
                ax.semilogy(grid, np.maximum(y, 1e-12), color=color, label=name)
            else:
                ax.plot(grid, y, color=color, label=name)
        ax.set_title(str(prob), fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=8)
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle(f"Convergencia (media sobre corridas) — {title}", fontsize=12)
    fig.text(0.5, -0.02, "FES", ha="center", fontsize=9)
    fig.text(-0.01, 0.5, "gap al optimo (o best)", va="center", rotation="vertical", fontsize=9)
    return _png(fig)


def build_report(input_dir, output_path):
    input_dir = Path(input_dir)
    css = """
    body{font-family:Segoe UI,Arial,sans-serif;margin:24px;color:#222;background:#fafafa}
    h1{font-size:22px} h2{font-size:18px;margin-top:28px;border-bottom:2px solid #ddd;padding-bottom:4px}
    table{border-collapse:collapse;margin:10px 0;font-size:13px;background:#fff}
    th,td{border:1px solid #ccc;padding:5px 9px;text-align:right} th{background:#f0f0f0}
    td.l,th.l{text-align:left} .win{color:#157f3b;font-weight:bold} .loss{color:#b00020;font-weight:bold}
    .tie{color:#777} .meta{color:#555;font-size:13px} img{max-width:100%;height:auto}
    .pill{display:inline-block;padding:1px 7px;border-radius:9px;background:#eef;font-size:12px}
    """
    parts = [f"<html><head><meta charset='utf-8'><style>{css}</style></head><body>"]
    parts.append("<h1>Reporte WO base vs WO+SHAP (por agente, regimen MaxFES)</h1>")

    # Metadatos desde el primer summary disponible.
    meta_mf = meta_runs = meta_agents = None
    for _, gdir in GROUPS:
        s = _read_csv(input_dir / gdir / "shap" / "values" / "summary.csv")
        if not s.empty:
            meta_mf = int(s["max_fes"].iloc[0]); meta_agents = int(s["agents"].iloc[0])
            meta_runs = int(s["run_id"].nunique()); break
    parts.append(
        f"<p class='meta'>MaxFES = <b>{meta_mf}</b> &middot; corridas = <b>{meta_runs}</b> "
        f"&middot; agentes = <b>{meta_agents}</b> &middot; generado {datetime.now():%Y-%m-%d %H:%M}</p>"
    )

    total_w = total_t = total_l = 0
    for title, gdir in GROUPS:
        base_dir, shap_dir = input_dir / gdir / "base", input_dir / gdir / "shap"
        base_df = _read_csv(base_dir / "values" / "summary.csv")
        shap_df = _read_csv(shap_dir / "values" / "summary.csv")
        if base_df.empty and shap_df.empty:
            continue
        ref = shap_df if not shap_df.empty else base_df
        problems = sorted(set(ref["problem"]), key=_problem_key)
        max_fes = int(ref["max_fes"].iloc[0])

        parts.append(f"<h2>{title}</h2>")

        # --- Tabla comparativa de fitness final ---
        parts.append("<table><tr>"
                     "<th class='l'>Problema</th><th>opt</th>"
                     "<th>base media&plusmn;std</th><th>base best</th>"
                     "<th>SHAP media&plusmn;std</th><th>SHAP best</th>"
                     "<th>gap SHAP</th><th>W/T/L (pareado)</th></tr>")
        for prob in problems:
            b = _stats(base_df, prob, "final_fitness") if not base_df.empty else _stats(pd.DataFrame(), prob, "final_fitness")
            s = _stats(shap_df, prob, "final_fitness") if not shap_df.empty else _stats(pd.DataFrame(), prob, "final_fitness")
            opt = float(ref[ref["problem"] == prob]["optimum"].iloc[0])
            gap_shap = _stats(shap_df, prob, "gap_to_optimum") if not shap_df.empty else {"mean": None}
            w, t, l = _paired_counts(base_df, shap_df, prob) if (not base_df.empty and not shap_df.empty) else (0, 0, 0)
            total_w += w; total_t += t; total_l += l
            cls = "win" if w > l else ("loss" if l > w else "tie")
            parts.append(
                f"<tr><td class='l'>{prob}</td><td>{_fmt(opt)}</td>"
                f"<td>{_fmt(b['mean'])} &plusmn; {_fmt(b['std'],2)}</td><td>{_fmt(b['best'])}</td>"
                f"<td>{_fmt(s['mean'])} &plusmn; {_fmt(s['std'],2)}</td><td>{_fmt(s['best'])}</td>"
                f"<td>{_fmt(gap_shap['mean'])}</td>"
                f"<td class='{cls}'>{w}/{t}/{l}</td></tr>"
            )
        parts.append("</table>")

        # --- Actividad del controlador (SHAP) ---
        if not shap_df.empty and "interventions" in shap_df.columns:
            ev = _read_csv(shap_dir / "values" / "controller_events.csv")
            parts.append("<table><tr><th class='l'>Problema</th>"
                         "<th>interv. media</th><th>FES_shap media</th>"
                         "<th>random</th><th>guided</th><th>feature dominante (top)</th></tr>")
            for prob in problems:
                sp = shap_df[shap_df["problem"] == prob]
                interv = float(sp["interventions"].mean()) if not sp.empty else None
                fshap = float(sp["fes_shap"].mean()) if not sp.empty else None
                nr = int(sp["n_reinit_random"].sum()) if "n_reinit_random" in sp else 0
                ng = int(sp["n_reinit_guided"].sum()) if "n_reinit_guided" in sp else 0
                top = "&mdash;"
                if not ev.empty and "shap_dominant_feature" in ev.columns:
                    sub = ev[ev["problem"] == prob]["shap_dominant_feature"].value_counts()
                    if not sub.empty:
                        top = ", ".join(f"{k} ({v})" for k, v in sub.head(3).items())
                parts.append(
                    f"<tr><td class='l'>{prob}</td><td>{_fmt(interv,3)}</td><td>{_fmt(fshap,4)}</td>"
                    f"<td>{nr}</td><td>{ng}</td><td class='l'>{top}</td></tr>"
                )
            parts.append("</table>")

        # --- Grafico de convergencia ---
        try:
            png = _convergence_figure(title, problems, base_dir, shap_dir, base_df, shap_df, max_fes)
            parts.append(f"<img src='data:image/png;base64,{png}'/>")
        except Exception as exc:  # pragma: no cover - el reporte no debe caerse por un plot
            parts.append(f"<p class='meta'>[grafico no disponible: {exc}]</p>")

    parts.insert(
        3,
        f"<p class='meta'>Resumen pareado global (SHAP vs base): "
        f"<span class='win'>{total_w} wins</span> / "
        f"<span class='tie'>{total_t} ties</span> / "
        f"<span class='loss'>{total_l} losses</span></p>",
    )
    parts.append("</body></html>")

    output_path = Path(output_path)
    output_path.write_text("\n".join(parts), encoding="utf-8")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Reporte HTML WO base vs WO+SHAP.")
    parser.add_argument("--input", required=True, help="Carpeta raiz con cec/ y tmlap_*/")
    parser.add_argument("--output", required=True, help="Ruta del HTML de salida.")
    args = parser.parse_args()
    out = build_report(args.input, args.output)
    print(f"Reporte generado: {out}")


if __name__ == "__main__":
    main()
