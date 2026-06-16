"""Resumen CONDENSADO de resultados para slides (piramide de agregacion).

Lee la salida del runner unificado (`runners/run_ablation.py`):

    experiments/cec2022_d<dim>_fes<MaxFES>/<mode>/values/summary.csv
    experiments/cec2022_d<dim>_fes<MaxFES>/<mode>/curves/conv_curve_F<n>_fes<MaxFES>_run<r>.csv
    experiments/cec2022_d<dim>_fes<MaxFES>/shap/values/controller_events.csv

y produce, para una presentacion de ~15 min, una jerarquia de salidas (de lo mas
agregado a lo mas detallado) como PNGs sueltos de calidad-slide:

- Nivel 0 (`veredicto.md`): por cada MaxFES, Wilcoxon signed-rank pareado base vs shap
  por funcion; conteo mejor/igual/peor (criterio CEC: better/equal/worse).
- Nivel 1 (`tabla_resumen_fes<X>.png` + `.csv`): tabla F1-F12 para el MaxFES destacado.
- Nivel 2 (`efecto_maxfes.png`): gap medio agregado vs MaxFES, series base/shap.
- Nivel 3 (`convergencia/`): convergencia por funcion (12 CEC) + TMLAP, base+shap con banda ±std.
- Nivel 4 (`actividad_controlador.png` + `.csv`): actividad del controlador SHAP.

Tolera datos PARCIALES: si falta `base/`, un MaxFES o `controller_events.csv`, genera lo
disponible y lo marca, sin romper. El pareo intersecta run_ids comunes.

Uso:
    python -m analysis.presentation_summary --input experiments \\
        --out-dir experiments/presentacion --highlight-fes auto --dim 10

# TODO: cuando se agreguen competidores (>=3 algoritmos), incorporar el test de
# Friedman + post-hoc; hoy solo hay base vs shap (2 condiciones) -> Wilcoxon.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import shapiro, wilcoxon

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.report_html import (  # noqa: E402
    _mean_curve,
    _problem_key,
    _read_csv,
    _stats,
)

plt.rcParams.update({
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "legend.fontsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "lines.linewidth": 2.0,
    "figure.figsize": (10, 6),
})

COLOR_BASE = "#1f77b4"
COLOR_SHAP = "#ff7f0e"
CLASS_COLORS = {"mejor": "#d7f0d0", "igual": "#eeeeee", "peor": "#f6d0d0", "sin datos": "#ffffff"}
# Notacion del paper WO (Han 2024, Tabla 6): + = shap mejor, = igual, - peor.
CLASS_SYMBOL = {"mejor": "+", "igual": "=", "peor": "−", "sin datos": "·"}
# Las 6 senales de control del WO = features del controlador SHAP (orden Han 2024).
SIGNAL_NAMES = ("alpha", "beta", "A", "R", "danger_signal", "safety_signal")


def parse_args():
    p = argparse.ArgumentParser(description="Resumen condensado de resultados para slides.")
    p.add_argument("--input", default="experiments")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--highlight-fes", default="auto",
                   help="MaxFES a destacar en la tabla por funcion. 'auto' = mayor disponible.")
    p.add_argument("--dim", type=int, default=10)
    return p.parse_args()


def discover_runs(input_dir, dim):
    """Devuelve {max_fes: rundir} ordenado para las carpetas cec2022_d<dim>_fes*."""
    runs = {}
    pattern = re.compile(r"fes(\d+)$")
    for d in Path(input_dir).glob(f"cec2022_d{dim}_fes*"):
        if not d.is_dir():
            continue
        m = pattern.search(d.name)
        if m:
            runs[int(m.group(1))] = d
    return dict(sorted(runs.items()))


def load_mode(rundir, mode):
    """(summary_df, curves_dir, events_df). DataFrames vacios si faltan."""
    base = Path(rundir) / mode
    summary = _read_csv(base / "values" / "summary.csv")
    events = _read_csv(base / "values" / "controller_events.csv")
    return summary, base / "curves", events


def _problems_in(*dfs):
    labels = set()
    for df in dfs:
        if not df.empty and "problem" in df.columns:
            labels.update(df["problem"].astype(str).unique())
    return sorted(labels, key=_problem_key)


def classify_function(base_df, shap_df, problem, tol=1e-8, alpha=0.05):
    """Clasifica una funcion como mejor/igual/peor para shap (minimizacion) via Wilcoxon.

    Devuelve dict {label, n_common, p, median_diff, clase}.
    """
    out = {"label": problem, "n_common": 0, "p": None, "median_diff": None, "clase": "sin datos"}
    if base_df.empty or shap_df.empty:
        return out
    b = base_df[base_df["problem"] == problem].set_index("run_id")["final_fitness"]
    s = shap_df[shap_df["problem"] == problem].set_index("run_id")["final_fitness"]
    common = b.index.intersection(s.index)
    if len(common) == 0:
        return out
    bv = b.loc[common].to_numpy(float)
    sv = s.loc[common].to_numpy(float)
    diff = sv - bv  # < 0 => shap mejor (minimizacion)
    out["n_common"] = int(len(common))
    out["median_diff"] = float(np.median(diff))
    if np.all(np.abs(diff) <= tol):
        out["clase"] = "igual"
        out["p"] = 1.0
        return out
    try:
        _, p = wilcoxon(bv, sv, zero_method="wilcox", alternative="two-sided")
    except ValueError:
        p = 1.0
    out["p"] = float(p)
    if p < alpha:
        out["clase"] = "mejor" if out["median_diff"] < 0 else "peor"
    else:
        out["clase"] = "igual"
    return out


def verdict_for_run(base_df, shap_df):
    """Conteo {mejor, igual, peor, sin datos} sobre las funciones de un MaxFES."""
    counts = {"mejor": 0, "igual": 0, "peor": 0, "sin datos": 0}
    rows = []
    for prob in _problems_in(base_df, shap_df):
        c = classify_function(base_df, shap_df, prob)
        counts[c["clase"]] += 1
        rows.append(c)
    return counts, rows


def write_verdict_md(runs_by_fes, out_path):
    """Nivel 0: tabla MaxFES x (W|T|L) en notacion del paper WO (+/=/−) + frase."""
    lines = ["# Veredicto base vs shap (Wilcoxon signed-rank, alpha=0.05)\n",
             "Notacion estilo paper WO (Han 2024): por funcion, **+** = shap mejor, ",
             "**=** = sin diferencia significativa, **−** = shap peor (minimizacion). ",
             "(W|T|L) = (mejor | igual | peor) sobre las funciones evaluadas.\n",
             "| MaxFES | n func | corridas (base/shap) | (W\\|T\\|L) shap vs base | sin base |",
             "|---|---|---|---|---|"]
    best_line = None
    unbalanced = []
    for fes, rundir in runs_by_fes.items():
        base_df, _, _ = load_mode(rundir, "base")
        shap_df, _, _ = load_mode(rundir, "shap")
        counts, _ = verdict_for_run(base_df, shap_df)
        n = sum(counts.values())
        nb = int(base_df["run_id"].nunique()) if not base_df.empty else 0
        ns = int(shap_df["run_id"].nunique()) if not shap_df.empty else 0
        wtl = f"({counts['mejor']}|{counts['igual']}|{counts['peor']})"
        flag = " ⚠" if (nb and ns and nb != ns) else ""
        if flag:
            unbalanced.append((fes, nb, ns))
        lines.append(f"| {fes:,} | {n} | {nb}/{ns}{flag} | {wtl} | {counts['sin datos']} |")
        if not base_df.empty:
            best_line = (fes, counts)
    if unbalanced:
        det = "; ".join(f"MaxFES={f:,}: base={b} vs shap={s}" for f, b, s in unbalanced)
        lines += ["", f"> ⚠ **Corridas desbalanceadas** ({det}). El pareo usa solo los run_ids "
                  "comunes, por lo que el poder estadistico en esos MaxFES es bajo (Wilcoxon "
                  "tiende a 'igual')."]
    if best_line:
        fes, c = best_line
        lines += ["", f"**Resumen (MaxFES={fes:,}):** (W|T|L) = "
                  f"({c['mejor']}|{c['igual']}|{c['peor']}) — shap mejora en {c['mejor']}, "
                  f"empata en {c['igual']} y pierde en {c['peor']} de las funciones."]
    else:
        lines += ["", "**Resumen:** comparacion base vs shap aun no disponible "
                  "(falta el modo base en los datos descargados)."]
    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


COLS = ["función", "óptimo", "Best base", "Best shap", "Avg±Std base", "Avg±Std shap",
        "H", "p (Wilcoxon)"]


def function_table(base_df, shap_df, problems):
    """Nivel 1 (estilo Tabla 4/5 del paper WO, condensada): por funcion, Best y Avg±Std
    de base y shap + columna H (+/=/−) de Wilcoxon.

    Devuelve (df_display, bold_cells, counts). `bold_cells` = lista de (fila, columna)
    a poner en negrita (el mejor de base/shap en Best y en Avg). `counts` = W|T|L.
    """
    rows, bold = [], []
    counts = {"mejor": 0, "igual": 0, "peor": 0, "sin datos": 0}
    for i, prob in enumerate(problems):
        b = _stats(base_df, prob, "final_fitness") if not base_df.empty else {"mean": None, "std": None, "best": None}
        s = _stats(shap_df, prob, "final_fitness") if not shap_df.empty else {"mean": None, "std": None, "best": None}
        c = classify_function(base_df, shap_df, prob)
        counts[c["clase"]] += 1
        opt = _optimum_for(shap_df, base_df, prob)
        rows.append({
            "función": prob,
            "óptimo": _num(opt),
            "Best base": _num(b["best"]), "Best shap": _num(s["best"]),
            "Avg±Std base": _pm(b["mean"], b["std"]), "Avg±Std shap": _pm(s["mean"], s["std"]),
            "H": CLASS_SYMBOL[c["clase"]],
            "p (Wilcoxon)": _pfmt(c["p"]),
        })
        # Negrita en el mejor (menor, minimizacion) de cada par.
        _mark_best(bold, i, b["best"], s["best"], "Best base", "Best shap")
        _mark_best(bold, i, b["mean"], s["mean"], "Avg±Std base", "Avg±Std shap")
    return pd.DataFrame(rows, columns=COLS), bold, counts


def _mark_best(bold, row, base_val, shap_val, base_col, shap_col):
    if base_val is None or shap_val is None or not np.isfinite(base_val) or not np.isfinite(shap_val):
        return
    if shap_val < base_val:
        bold.append((row, shap_col))
    elif base_val < shap_val:
        bold.append((row, base_col))


def _pm(mean, std):
    if mean is None or not np.isfinite(mean):
        return "—"
    s = std if (std is not None and np.isfinite(std)) else 0.0
    return f"{mean:.4g} ± {s:.2g}"


def _num(v):
    return "—" if (v is None or not np.isfinite(v)) else f"{v:.4g}"


def _pfmt(p):
    if p is None or not np.isfinite(p):
        return "—"
    return "<0.001" if p < 1e-3 else f"{p:.3g}"


def _vals(df, prob):
    """final_fitness finitos de `prob` en `df` (array)."""
    if df.empty or "problem" not in df.columns or prob not in set(df["problem"]):
        return np.array([], dtype=float)
    v = df[df["problem"] == prob]["final_fitness"].astype(float).to_numpy()
    return v[np.isfinite(v)]


def _shapiro_p(vals):
    """p-valor de Shapiro–Wilk; None si n<3 o varianza nula (todos iguales)."""
    if len(vals) < 3 or float(np.ptp(vals)) == 0.0:
        return None
    try:
        return float(shapiro(vals).pvalue)
    except Exception:  # noqa: BLE001
        return None




def render_table_png(df, bold_cells, counts, out_path, title, key_col="función"):
    # Fila resumen (W|T|L) al pie, estilo paper.
    wtl_row = {c: "" for c in df.columns}
    wtl_row[key_col] = "(W|T|L)"
    wtl_row["H"] = f"({counts['mejor']}|{counts['igual']}|{counts['peor']})"
    df_full = pd.concat([df, pd.DataFrame([wtl_row], columns=df.columns)], ignore_index=True)

    fig, ax = plt.subplots(figsize=(16, 0.5 + 0.42 * (len(df_full) + 1)))
    ax.axis("off")
    ax.set_title(title, pad=12)
    tbl = ax.table(cellText=df_full.values, colLabels=df_full.columns, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.auto_set_column_width(col=list(range(len(df_full.columns))))
    tbl.scale(1, 1.5)
    col_idx = {c: i for i, c in enumerate(df_full.columns)}
    # Color de la columna H segun el simbolo; negrita en los mejores Best/Avg.
    sym_to_clase = {v: k for k, v in CLASS_SYMBOL.items()}
    for r in range(len(df)):
        clase = sym_to_clase.get(df.iloc[r]["H"], "sin datos")
        tbl[(r + 1, col_idx["H"])].set_facecolor(CLASS_COLORS.get(clase, "#ffffff"))
    for (r, col) in bold_cells:
        tbl[(r + 1, col_idx[col])].set_text_props(fontweight="bold")
    # Fila resumen resaltada.
    last = len(df_full)
    for c in range(len(df_full.columns)):
        tbl[(last, c)].set_facecolor("#dddddd")
        tbl[(last, c)].set_text_props(fontweight="bold")
    for c in range(len(df_full.columns)):
        tbl[(0, c)].set_facecolor("#404040")
        tbl[(0, c)].set_text_props(color="white", fontweight="bold")
    fig.savefig(out_path)
    plt.close(fig)


def _normality_sr_row(label, base_df, shap_df, prob, key_col, alpha=0.05):
    """Una fila: Shapiro–Wilk por grupo + Wilcoxon signed-rank (pareado). Devuelve (row, clase)."""
    bv, sv = _vals(base_df, prob), _vals(shap_df, prob)
    sw_b, sw_s = _shapiro_p(bv), _shapiro_p(sv)
    if sw_b is None or sw_s is None:
        normal = "—"
    else:
        normal = "sí" if (sw_b >= alpha and sw_s >= alpha) else "no"
    c = classify_function(base_df, shap_df, prob)  # Wilcoxon signed-rank pareado
    row = {
        key_col: label,
        "n (b/s)": f"{len(bv)}/{len(sv)}",
        "SW p base": _pfmt(sw_b), "SW p shap": _pfmt(sw_s),
        "¿normal? (α=0.05)": normal,
        "p (Wilcoxon)": _pfmt(c["p"]),
        "H": CLASS_SYMBOL[c["clase"]],
    }
    return row, c["clase"]


def normality_sr_table(items, out_png, out_csv, title, key_col):
    """Tabla Shapiro–Wilk + Wilcoxon signed-rank (pareado). `items` = (label, base_df, shap_df, prob)."""
    rows, counts = [], {"mejor": 0, "igual": 0, "peor": 0, "sin datos": 0}
    for label, bdf, sdf, prob in items:
        row, clase = _normality_sr_row(label, bdf, sdf, prob, key_col)
        counts[clase] += 1
        rows.append(row)
    cols = [key_col, "n (b/s)", "SW p base", "SW p shap", "¿normal? (α=0.05)", "p (Wilcoxon)", "H"]
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(out_csv, index=False)
    render_table_png(df, [], counts, out_png, title, key_col=key_col)
    return counts


def plot_maxfes_effect(runs_by_fes, out_path):
    """Nivel 2: gap medio agregado (media sobre funciones) vs MaxFES, series base/shap."""
    fes_list = sorted(runs_by_fes)
    base_y, shap_y = [], []
    for fes in fes_list:
        base_df, _, _ = load_mode(runs_by_fes[fes], "base")
        shap_df, _, _ = load_mode(runs_by_fes[fes], "shap")
        base_y.append(_mean_gap(base_df))
        shap_y.append(_mean_gap(shap_df))
    fig, ax = plt.subplots()
    if any(v is not None for v in base_y):
        ax.plot(fes_list, [np.nan if v is None else v for v in base_y],
                "o-", color=COLOR_BASE, label="base")
    if any(v is not None for v in shap_y):
        ax.plot(fes_list, [np.nan if v is None else v for v in shap_y],
                "s-", color=COLOR_SHAP, label="shap")
    ax.set_xscale("log")
    has_base = any(v is not None for v in base_y)
    ax.set_yscale("log")
    ax.set_xlabel("MaxFES (escala log)")
    ax.set_ylabel("gap medio al óptimo (media sobre funciones)")
    ax.set_title("Efecto del presupuesto MaxFES" + ("" if has_base else " — solo shap (falta base)"))
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()
    fig.savefig(out_path)
    plt.close(fig)


def _mean_gap(df):
    if df.empty or "gap_to_optimum" not in df.columns:
        return None
    vals = df["gap_to_optimum"].astype(float)
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if not vals.empty else None


def _safe_load_curves(curves_dir, label):
    """Como report_html._load_curves pero tolera CSV corruptos/truncados (descarga a medio
    bajar): coacciona a numerico, dropea filas no parseables y salta archivos rotos."""
    out = []
    cdir = Path(curves_dir)
    if not cdir.exists():
        return out
    for f in sorted(cdir.glob(f"conv_curve_{label}_fes*_run*.csv")):
        try:
            df = pd.read_csv(f)
            if "fes" not in df.columns or "best_fitness" not in df.columns:
                continue
            fes_v = pd.to_numeric(df["fes"], errors="coerce")
            best_v = pd.to_numeric(df["best_fitness"], errors="coerce")
            mask = fes_v.notna() & best_v.notna()
            if mask.sum() >= 2:
                out.append((fes_v[mask].to_numpy(float), best_v[mask].to_numpy(float)))
        except Exception:  # noqa: BLE001
            continue  # archivo ilegible (p.ej. a medio descargar)
    return out


def _mean_std_curve(curves, max_fes, n=250):
    """Como `_mean_curve` pero devuelve (grid, media, std) interpolando cada corrida a la grilla."""
    grid = np.linspace(0, max_fes, n)
    ys = [np.interp(grid, fes, best, left=best[0], right=best[-1]) for fes, best in curves]
    if not ys:
        return None, None, None
    arr = np.asarray(ys, dtype=float)
    return grid, arr.mean(axis=0), arr.std(axis=0)


def _draw_convergence(ax, rundir, prob, fes, title=None, ylabel=True, legend=True, caption=False):
    """Dibuja convergencia base+shap (media + banda ±1 std) en un eje dado. Devuelve True si dibujó.

    CEC (óptimo finito): y = media − óptimo, en semilogy si es positiva. TMLAP (sin óptimo):
    fitness crudo, escala lineal. En log se acota la banda al piso de la media (evita el
    'plunge' a 1e-12 cuando media−std<0).
    """
    base_df, base_curves, _ = load_mode(rundir, "base")
    shap_df, shap_curves, _ = load_mode(rundir, "shap")
    opt = _optimum_for(shap_df, base_df, prob)
    use_gap = np.isfinite(opt)
    series = []
    for curves_dir, color, name in ((base_curves, COLOR_BASE, "base"),
                                    (shap_curves, COLOR_SHAP, "shap")):
        grid, mean, std = _mean_std_curve(_safe_load_curves(curves_dir, prob), fes)
        if grid is None:
            continue
        center = (mean - opt) if use_gap else mean
        series.append((grid, center, std, color, name))
    if not series:
        return False
    log_ok = use_gap and all(np.all(c > 0) for _, c, _, _, _ in series)
    floor = max(min(float(np.min(c)) for _, c, _, _, _ in series), 1e-12) if log_ok else None
    for grid, center, std, color, name in series:
        lo, hi = center - std, center + std
        if log_ok:
            lo, hi = np.maximum(lo, floor), np.maximum(hi, floor)
            ax.fill_between(grid, lo, hi, color=color, alpha=0.2)
            ax.semilogy(grid, np.maximum(center, floor), color=color, label=name)
        else:
            ax.fill_between(grid, lo, hi, color=color, alpha=0.2)
            ax.plot(grid, center, color=color, label=name)
    if log_ok:
        ax.set_ylim(bottom=floor * 0.5)
    ax.set_xlabel("FES")
    if ylabel:
        ax.set_ylabel("gap al óptimo" if use_gap else "fitness (mejor)")
    if title:
        ax.set_title(title)
    if caption:
        ax.text(0.99, 0.97, "banda = ±1 desv. est. entre corridas", transform=ax.transAxes,
                ha="right", va="top", fontsize=8.5, color="#555")
    ax.grid(True, alpha=0.3)
    if legend:
        ax.legend()
    return True


def plot_convergence_band(rundir, prob, fes, out_png, title=None):
    """Una figura individual (un eje) con la convergencia de `prob`. Reusa `_draw_convergence`."""
    fig, ax = plt.subplots(figsize=(8, 5))
    drew = _draw_convergence(ax, rundir, prob, fes, title=title or prob, caption=True)
    if drew:
        fig.savefig(out_png)
    plt.close(fig)
    return drew


def convergence_grid_cec(input_dir, dim, fes, out_png, ncols=4):
    """Panel único 4×3 (4 columnas × 3 filas): convergencia de las 12 funciones CEC en una imagen."""
    runs_by_fes = discover_runs(input_dir, dim)
    if fes not in runs_by_fes:
        return False
    rundir = runs_by_fes[fes]
    base_df, _, _ = load_mode(rundir, "base")
    shap_df, _, _ = load_mode(rundir, "shap")
    probs = _problems_in(base_df, shap_df)
    if not probs:
        return False
    nrows = int(np.ceil(len(probs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows), squeeze=False)
    drew = False
    for i, prob in enumerate(probs):
        ax = axes[i // ncols][i % ncols]
        if _draw_convergence(ax, rundir, prob, fes, title=prob,
                             ylabel=(i % ncols == 0), legend=(i == 0)):
            drew = True
    for j in range(len(probs), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle(
        f"Convergencia por función — CEC2022 d{dim}, MaxFES={fes:,}  "
        "(media base vs shap; banda = ±1 desv. est. entre corridas)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    if drew:
        fig.savefig(out_png)
    plt.close(fig)
    return drew


def convergence_bands(input_dir, dim, fes, out_dir,
                      tmlap_configs=("dura_30_fes500k",)):
    """Genera convergencia con banda ±std: panel 4×3 CEC (1 imagen) + cada TMLAP, en `out_dir/convergencia/`."""
    conv_dir = Path(out_dir) / "convergencia"
    conv_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    # CEC: un único panel 4×3 con las 12 funciones (en vez de 12 PNG sueltos).
    if convergence_grid_cec(input_dir, dim, fes, conv_dir / "convergencia_cec_panel.png"):
        n += 1
    for cfg in tmlap_configs:
        rundir = Path(input_dir) / cfg
        if not rundir.exists():
            continue
        base_df, _, _ = load_mode(rundir, "base")
        shap_df, _, _ = load_mode(rundir, "shap")
        probs = _problems_in(base_df, shap_df)
        if not probs:
            continue
        mf = _max_fes_of(rundir)
        title = f"Convergencia TMLAP {cfg}\n({probs[0]})"
        if plot_convergence_band(rundir, probs[0], mf, conv_dir / f"convergencia_tmlap_{cfg}.png", title):
            n += 1
    return n


def _optimum_for(shap_df, base_df, prob):
    for df in (shap_df, base_df):
        if not df.empty and prob in set(df["problem"]):
            sub = df[df["problem"] == prob]["optimum"]
            if not sub.empty:
                return float(sub.iloc[0])
    return np.nan


def plot_boxplots_per_function(base_df, shap_df, problems, out_path, fes, ncols=4):
    """Nivel 1.5 (Fig. 6 del paper WO): una grilla con 1 panel por funcion; en cada
    panel, boxplots base vs shap del gap final, con ESCALA PROPIA por funcion (log si
    todos los valores son positivos). Tolera datos parciales."""
    from matplotlib.patches import Patch

    if not problems:
        return False
    nrows = int(np.ceil(len(problems) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows), squeeze=False)
    drew = False
    for idx, prob in enumerate(problems):
        ax = axes[idx // ncols][idx % ncols]
        data, labels, colors = [], [], []
        for df, color, name in ((base_df, COLOR_BASE, "base"), (shap_df, COLOR_SHAP, "shap")):
            if df.empty or "problem" not in df.columns or prob not in set(df["problem"]):
                continue
            vals = df[df["problem"] == prob]["gap_to_optimum"].astype(float)
            vals = vals[np.isfinite(vals)]
            if vals.empty:
                continue
            data.append(vals.to_numpy(float))
            labels.append(name)
            colors.append(color)
        if not data:
            ax.axis("off")
            ax.set_title(f"{prob} (sin datos)")
            continue
        drew = True
        bp = ax.boxplot(data, tick_labels=labels, widths=0.5, patch_artist=True)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        if all(np.min(d) > 0 for d in data):
            ax.set_yscale("log")
        ax.set_title(prob)
        ax.grid(True, axis="y", alpha=0.3)
        if idx % ncols == 0:
            ax.set_ylabel("gap al óptimo")
    for j in range(len(problems), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    if drew:
        fig.legend(handles=[Patch(facecolor=COLOR_BASE, alpha=0.6, label="base"),
                            Patch(facecolor=COLOR_SHAP, alpha=0.6, label="shap")],
                   loc="upper right")
        fig.suptitle(f"Distribución del gap final por función — MaxFES={fes:,}", fontsize=15)
        fig.savefig(out_path)
    plt.close(fig)
    return drew


def _feature_contributions(shap_vals, events_df=None):
    """(shares, dom_counts) por senal desde `shap_values.csv`.

    `shares[s]` = cuota media |SHAP_s| / Σ|SHAP| (np.nan si no hay datos).
    `dom_counts[s]` = nº de explicaciones en que esa senal fue la dominante.
    """
    feat_cols = [f"shap_{s}" for s in SIGNAL_NAMES]
    shares = {s: np.nan for s in SIGNAL_NAMES}
    if not shap_vals.empty and all(c in shap_vals.columns for c in feat_cols):
        mag = shap_vals[feat_cols].abs()
        denom = mag.sum(axis=1).replace(0, np.nan)
        mean_share = mag.div(denom, axis=0).mean()
        for s, c in zip(SIGNAL_NAMES, feat_cols):
            shares[s] = float(mean_share[c])
    dom_counts = {s: 0 for s in SIGNAL_NAMES}
    dom_src = None
    if not shap_vals.empty and "dominant_feature" in shap_vals.columns:
        dom_src = shap_vals["dominant_feature"]
    elif events_df is not None and not events_df.empty and "shap_dominant_feature" in events_df.columns:
        dom_src = events_df["shap_dominant_feature"]
    if dom_src is not None:
        vc = dom_src.value_counts()
        for s in SIGNAL_NAMES:
            dom_counts[s] = int(vc.get(s, 0))
    return shares, dom_counts


def plot_features_shap(shares, dom_counts, title, out_png):
    """Barra horizontal de las 6 senales (cuota media |SHAP|; fallback a conteo dominante)."""
    have_shares = any(np.isfinite(shares[s]) for s in SIGNAL_NAMES)
    metric = shares if have_shares else {s: float(dom_counts[s]) for s in SIGNAL_NAMES}
    order = sorted(SIGNAL_NAMES, key=lambda s: metric[s] if np.isfinite(metric[s]) else -1.0)
    fig, ax = plt.subplots(figsize=(8, 5))
    if have_shares:
        vals = [shares[s] if np.isfinite(shares[s]) else 0.0 for s in order]
        ax.barh(list(order), vals, color="#6a3d9a")
        ax.set_xlabel("cuota media de contribución   |SHAP| / Σ|SHAP|")
        for y, s in enumerate(order):
            ax.text(vals[y], y, f"  dominante {dom_counts[s]}×",
                    va="center", fontsize=9.5, color="#333")
        ax.set_xlim(0, max(vals) * 1.25 if max(vals) > 0 else 1)
    else:
        vals = [dom_counts[s] for s in order]
        ax.barh(list(order), vals, color="#6a3d9a")
        ax.set_xlabel("nº de veces dominante")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3)
    fig.savefig(out_png)
    plt.close(fig)


def controller_activity(shap_df, events_df, shap_vals, out_dir):
    """Nivel 4: figura de acciones de rescate + figura de features (highlight) + CSVs."""
    if shap_df.empty:
        raise ValueError("sin summary de shap")
    interv = float(shap_df["interventions"].mean()) if "interventions" in shap_df else np.nan
    nr = int(shap_df["n_reinit_random"].sum()) if "n_reinit_random" in shap_df else 0
    ng = int(shap_df["n_reinit_guided"].sum()) if "n_reinit_guided" in shap_df else 0
    shares, dom_counts = _feature_contributions(shap_vals, events_df)

    # --- CSVs ---
    top = sorted(((v, k) for k, v in dom_counts.items()), reverse=True)[:3]
    pd.DataFrame([{
        "intervenciones_media": interv,
        "reinit_random_total": nr,
        "reinit_guided_total": ng,
        "feature_dominante_top": "; ".join(f"{k}({v})" for v, k in top if v > 0) or "—",
    }]).to_csv(out_dir / "actividad_controlador.csv", index=False)
    pd.DataFrame([
        {"senal": s, "cuota_media_SHAP": shares[s], "veces_dominante": dom_counts[s]}
        for s in SIGNAL_NAMES
    ]).to_csv(out_dir / "features_shap.csv", index=False)

    # --- Figura: acciones de rescate ---
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(["reinit_random", "reinit_guided"], [nr, ng], color=[COLOR_BASE, COLOR_SHAP])
    ax.set_title(f"Acciones de rescate del controlador\n(intervención media/corrida = {interv:.2g})")
    ax.set_ylabel("conteo total de acciones")
    ax.grid(True, axis="y", alpha=0.3)
    for x, v in enumerate((nr, ng)):
        ax.text(x, v, str(v), ha="center", va="bottom", fontsize=12, weight="bold")
    fig.savefig(out_dir / "acciones_rescate.png")
    plt.close(fig)

    # --- Figura: las 6 senales (config destacada) ---
    plot_features_shap(shares, dom_counts, "Contribución SHAP por señal (las 6)",
                       out_dir / "features_shap.png")


def features_per_experiment(input_dir, dim, out_dir, min_runs=15):
    """Un `features_shap_<tag>.png` por cada experimento con modo shap (CEC por MaxFES + TMLAP).

    Omite los experimentos con menos de ``min_runs`` corridas (p.ej. shap@5M con n=5,
    estadisticamente no concluyente) para no generar figuras enganhosas.
    """
    n = 0
    for mf, rundir in discover_runs(input_dir, dim).items():
        sv = _read_csv(Path(rundir) / "shap" / "values" / "shap_values.csv")
        ev = _read_csv(Path(rundir) / "shap" / "values" / "controller_events.csv")
        if sv.empty:
            continue
        runs = int(sv["run_id"].nunique()) if "run_id" in sv.columns else 0
        if runs < min_runs:
            print(f"  [omitido CEC fes{mf}: n={runs} < {min_runs}]")
            continue
        shares, dom = _feature_contributions(sv, ev)
        title = f"SHAP por señal — CEC2022 d{dim}, MaxFES={mf:,}\n({len(sv)} expl., {runs} corridas)"
        plot_features_shap(shares, dom, title, out_dir / f"features_shap_cec_d{dim}_fes{mf}.png")
        n += 1
    for name, rundir in discover_tmlap_runs(input_dir).items():
        sv = _read_csv(Path(rundir) / "shap" / "values" / "shap_values.csv")
        ev = _read_csv(Path(rundir) / "shap" / "values" / "controller_events.csv")
        if sv.empty:
            continue
        runs = int(sv["run_id"].nunique()) if "run_id" in sv.columns else 0
        if runs < min_runs:
            print(f"  [omitido TMLAP {name}: n={runs} < {min_runs}]")
            continue
        shares, dom = _feature_contributions(sv, ev)
        title = f"SHAP por señal — TMLAP {name}\n({len(sv)} expl., {runs} corridas)"
        plot_features_shap(shares, dom, title, out_dir / f"features_shap_{name}.png")
        n += 1
    return n


def features_comparison_cec(input_dir, dim, out_dir, budgets=(5000, 50000, 500000)):
    """Comparativa: cuota media de las 6 señales en CEC para varios MaxFES (muestra el shift).

    Excluye 5·10⁶ por defecto (shap n=5, no concluyente). Barras agrupadas por señal,
    una serie por presupuesto, con paleta clara→oscura según el presupuesto crece.
    """
    runs_by_fes = discover_runs(input_dir, dim)
    series = {}
    for mf in budgets:
        if mf not in runs_by_fes:
            continue
        sv = _read_csv(Path(runs_by_fes[mf]) / "shap" / "values" / "shap_values.csv")
        if sv.empty:
            continue
        shares, _ = _feature_contributions(sv)
        series[mf] = shares
    if len(series) < 2:
        return False
    mean_imp = {s: np.nanmean([series[mf][s] for mf in series]) for s in SIGNAL_NAMES}
    sig_order = sorted(SIGNAL_NAMES,
                       key=lambda s: -(mean_imp[s] if np.isfinite(mean_imp[s]) else -1.0))
    x = np.arange(len(sig_order))
    mfs = sorted(series)
    w = 0.8 / len(mfs)
    palette = ["#c6dbef", "#6baed6", "#2171b5", "#08306b"]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, mf in enumerate(mfs):
        vals = [series[mf][s] if np.isfinite(series[mf][s]) else 0.0 for s in sig_order]
        ax.bar(x + i * w - 0.4 + w / 2, vals, w, label=f"MaxFES={mf:,}",
               color=palette[i % len(palette)])
    ax.set_xticks(x)
    ax.set_xticklabels(sig_order, rotation=15)
    ax.set_ylabel("cuota media de contribución   |SHAP| / Σ|SHAP|")
    ax.set_title(f"Contribución SHAP por señal según el presupuesto (CEC2022 d{dim})")
    ax.legend(title="presupuesto")
    ax.grid(True, axis="y", alpha=0.3)
    fig.savefig(out_dir / "features_shap_comparativa_cec.png")
    plt.close(fig)
    return True


def discover_tmlap_runs(input_dir):
    """Carpetas no-CEC con base|shap/values/summary.csv (p.ej. dura_51_fes50k)."""
    runs = {}
    for d in sorted(Path(input_dir).iterdir()):
        if not d.is_dir() or d.name.startswith("cec2022") or d.name == "presentacion":
            continue
        if (d / "base" / "values" / "summary.csv").exists() or \
           (d / "shap" / "values" / "summary.csv").exists():
            runs[d.name] = d
    return runs


def summarize_tmlap(runs, out_dir):
    """Resumen TMLAP (caso aplicado): tabla + boxplots + convergencia + veredicto.

    Cada carpeta es una corrida (instancia x MaxFES). Como TMLAP no tiene optimo
    exacto (gap NaN), todo se basa en `final_fitness`. Una carpeta = una fila/panel.
    """
    if not runs:
        return
    rows, bold = [], []
    counts = {"mejor": 0, "igual": 0, "peor": 0, "sin datos": 0}
    specs = []  # (etiqueta, rundir, problem, base_df, shap_df, base_curves, shap_curves)
    for i, (name, rundir) in enumerate(runs.items()):
        base_df, base_curves, _ = load_mode(rundir, "base")
        shap_df, shap_curves, _ = load_mode(rundir, "shap")
        for prob in _problems_in(base_df, shap_df):
            b = _stats(base_df, prob, "final_fitness") if not base_df.empty else {"mean": None, "std": None, "best": None}
            s = _stats(shap_df, prob, "final_fitness") if not shap_df.empty else {"mean": None, "std": None, "best": None}
            c = classify_function(base_df, shap_df, prob)
            counts[c["clase"]] += 1
            nb = int(base_df["run_id"].nunique()) if not base_df.empty else 0
            ns = int(shap_df["run_id"].nunique()) if not shap_df.empty else 0
            sw_b, sw_s = _shapiro_p(_vals(base_df, prob)), _shapiro_p(_vals(shap_df, prob))
            normal = ("—" if (sw_b is None or sw_s is None)
                      else ("sí" if (sw_b >= 0.05 and sw_s >= 0.05) else "no"))
            ridx = len(rows)
            rows.append({
                "corrida": name,
                "Best base": _num(b["best"]), "Best shap": _num(s["best"]),
                "Avg±Std base": _pm(b["mean"], b["std"]), "Avg±Std shap": _pm(s["mean"], s["std"]),
                "n (b/s)": f"{nb}/{ns}",
                "SW p base": _pfmt(sw_b), "SW p shap": _pfmt(sw_s), "¿normal? (α=0.05)": normal,
                "p (Wilcoxon)": _pfmt(c["p"]), "H": CLASS_SYMBOL[c["clase"]],
            })
            _mark_best(bold, ridx, b["best"], s["best"], "Best base", "Best shap")
            _mark_best(bold, ridx, b["mean"], s["mean"], "Avg±Std base", "Avg±Std shap")
            specs.append((name, rundir, prob, base_df, shap_df, base_curves, shap_curves))

    if not rows:
        return
    cols = ["corrida", "Best base", "Best shap", "Avg±Std base", "Avg±Std shap", "n (b/s)",
            "SW p base", "SW p shap", "¿normal? (α=0.05)", "p (Wilcoxon)", "H"]
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(out_dir / "tabla_tmlap.csv", index=False)
    _render_tmlap_table(df, bold, counts, out_dir / "tabla_tmlap.png")

    # Veredicto TMLAP
    lines = ["# Veredicto TMLAP base vs shap (Wilcoxon, alpha=0.05)\n",
             "Caso aplicado (sin optimo exacto -> sobre `final_fitness`, minimizacion).",
             "**+** shap mejor, **=** igual, **−** shap peor.\n",
             "| corrida | n (base/shap) | H |", "|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['corrida']} | {r['n (b/s)']} | {r['H']} |")
    lines += ["", f"**(W|T|L) = ({counts['mejor']}|{counts['igual']}|{counts['peor']})** "
              "sobre las corridas TMLAP evaluadas."]
    (out_dir / "veredicto_tmlap.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    _plot_tmlap_boxplots(specs, out_dir / "boxplots_tmlap.png")


def _render_tmlap_table(df, bold_cells, counts, out_path):
    wtl_row = {c: "" for c in df.columns}
    wtl_row["corrida"] = "(W|T|L)"
    wtl_row["H"] = f"({counts['mejor']}|{counts['igual']}|{counts['peor']})"
    df_full = pd.concat([df, pd.DataFrame([wtl_row], columns=df.columns)], ignore_index=True)
    fig, ax = plt.subplots(figsize=(max(13, 1.7 * len(df_full.columns)),
                                    0.6 + 0.45 * (len(df_full) + 1)))
    ax.axis("off")
    ax.set_title("TMLAP (instancia dura) — base vs shap "
                 "(Shapiro–Wilk + Wilcoxon signed-rank pareado, α=0.05)", pad=12)
    tbl = ax.table(cellText=df_full.values, colLabels=df_full.columns, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.auto_set_column_width(col=list(range(len(df_full.columns))))
    tbl.scale(1, 1.5)
    col_idx = {c: i for i, c in enumerate(df_full.columns)}
    sym_to_clase = {v: k for k, v in CLASS_SYMBOL.items()}
    for r in range(len(df)):
        clase = sym_to_clase.get(df.iloc[r]["H"], "sin datos")
        tbl[(r + 1, col_idx["H"])].set_facecolor(CLASS_COLORS.get(clase, "#ffffff"))
    for (r, col) in bold_cells:
        tbl[(r + 1, col_idx[col])].set_text_props(fontweight="bold")
    last = len(df_full)
    for c in range(len(df_full.columns)):
        tbl[(last, c)].set_facecolor("#dddddd")
        tbl[(last, c)].set_text_props(fontweight="bold")
        tbl[(0, c)].set_facecolor("#404040")
        tbl[(0, c)].set_text_props(color="white", fontweight="bold")
    fig.savefig(out_path)
    plt.close(fig)


def _plot_tmlap_boxplots(specs, out_path):
    n = len(specs)
    fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 4.2), squeeze=False)
    for idx, (name, _, prob, base_df, shap_df, _, _) in enumerate(specs):
        ax = axes[0][idx]
        data, labels, colors = [], [], []
        for df, color, mname in ((base_df, COLOR_BASE, "base"), (shap_df, COLOR_SHAP, "shap")):
            if df.empty or prob not in set(df["problem"]):
                continue
            vals = df[df["problem"] == prob]["final_fitness"].astype(float)
            vals = vals[np.isfinite(vals)]
            if vals.empty:
                continue
            data.append(vals.to_numpy(float)); labels.append(mname); colors.append(color)
        if not data:
            ax.axis("off"); ax.set_title(f"{name} (sin datos)"); continue
        bp = ax.boxplot(data, tick_labels=labels, widths=0.5, patch_artist=True)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color); patch.set_alpha(0.6)
        ax.set_title(name)
        ax.grid(True, axis="y", alpha=0.3)
        if idx == 0:
            ax.set_ylabel("fitness final (costo)")
    fig.suptitle("TMLAP — distribución del fitness final", fontsize=15)
    fig.savefig(out_path)
    plt.close(fig)


def _max_fes_of(rundir):
    for mode in ("base", "shap"):
        df, _, _ = load_mode(rundir, mode)
        if not df.empty and "max_fes" in df.columns:
            return int(df["max_fes"].iloc[0])
    return 1


def main():
    args = parse_args()
    runs_by_fes = discover_runs(args.input, args.dim)
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.input) / "presentacion"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not runs_by_fes:
        print(f"No se encontraron carpetas cec2022_d{args.dim}_fes* en {args.input}.")
        return
    print(f"MaxFES detectados: {', '.join(f'{k:,}' for k in runs_by_fes)}")

    if args.highlight_fes == "auto":
        highlight = max(runs_by_fes)
    else:
        highlight = int(args.highlight_fes)
        if highlight not in runs_by_fes:
            highlight = max(runs_by_fes)
            print(f"--highlight-fes no disponible; usando {highlight:,}.")

    # Nivel 0
    try:
        write_verdict_md(runs_by_fes, out_dir / "veredicto.md")
        print(f"[nivel 0] {out_dir / 'veredicto.md'}")
    except Exception as exc:  # noqa: BLE001
        print(f"[nivel 0 no disponible: {exc}]")

    hl_dir = runs_by_fes[highlight]
    base_df, _, _ = load_mode(hl_dir, "base")
    shap_df, _, shap_events = load_mode(hl_dir, "shap")
    shap_vals = _read_csv(hl_dir / "shap" / "values" / "shap_values.csv")
    problems = _problems_in(base_df, shap_df)

    # Nivel 1
    try:
        df, bold, counts = function_table(base_df, shap_df, problems)
        df.to_csv(out_dir / f"tabla_resumen_fes{highlight}.csv", index=False)
        render_table_png(df, bold, counts, out_dir / f"tabla_resumen_fes{highlight}.png",
                         f"Resultados por función — CEC2022 d{args.dim}, MaxFES={highlight:,}  "
                         "(base = WO · shap = WO+SHAP · H: +/=/− Wilcoxon pareado α=0.05)")
        print(f"[nivel 1] tabla_resumen_fes{highlight}.png/.csv")
    except Exception as exc:  # noqa: BLE001
        print(f"[nivel 1 no disponible: {exc}]")

    # Nivel 1b: Shapiro-Wilk (normalidad por grupo) + Wilcoxon signed-rank (pareado) — CEC y TMLAP
    try:
        cec_items = [(p, base_df, shap_df, p) for p in problems]
        normality_sr_table(
            cec_items, out_dir / "shapiro_wilcoxon_cec.png", out_dir / "shapiro_wilcoxon_cec.csv",
            f"CEC2022 d{args.dim}, MaxFES={highlight:,} — Shapiro–Wilk (normalidad) + "
            "Wilcoxon signed-rank pareado (α=0.05)", key_col="función")
        tmlap_items = []
        for name, rundir in discover_tmlap_runs(args.input).items():
            tb, _, _ = load_mode(rundir, "base")
            ts, _, _ = load_mode(rundir, "shap")
            for prob in _problems_in(tb, ts):
                tmlap_items.append((name, tb, ts, prob))
        if tmlap_items:
            normality_sr_table(
                tmlap_items, out_dir / "shapiro_wilcoxon_tmlap.png",
                out_dir / "shapiro_wilcoxon_tmlap.csv",
                "TMLAP — Shapiro–Wilk (normalidad) + Wilcoxon signed-rank pareado (α=0.05)",
                key_col="corrida")
        print("[nivel 1b] shapiro_wilcoxon_cec.png + shapiro_wilcoxon_tmlap.png")
    except Exception as exc:  # noqa: BLE001
        print(f"[nivel 1b no disponible: {exc}]")

    # Nivel 1.5: boxplots por funcion (Fig. 6 del paper), 1 panel por funcion
    try:
        ok = plot_boxplots_per_function(base_df, shap_df, problems,
                                        out_dir / "boxplots_por_funcion.png", highlight)
        print("[nivel 1.5] boxplots_por_funcion.png" if ok else "[nivel 1.5 no disponible: sin datos]")
    except Exception as exc:  # noqa: BLE001
        print(f"[nivel 1.5 no disponible: {exc}]")

    # Nivel 2
    try:
        plot_maxfes_effect(runs_by_fes, out_dir / "efecto_maxfes.png")
        print(f"[nivel 2] efecto_maxfes.png")
    except Exception as exc:  # noqa: BLE001
        print(f"[nivel 2 no disponible: {exc}]")

    # Nivel 3: convergencia por funcion/instancia con banda ±std (CEC + TMLAP)
    try:
        n = convergence_bands(args.input, args.dim, highlight, out_dir)
        print(f"[nivel 3] convergencia/ ({n} figuras con banda ±std)")
    except Exception as exc:  # noqa: BLE001
        print(f"[nivel 3 no disponible: {exc}]")

    # Nivel 4
    try:
        controller_activity(shap_df, shap_events, shap_vals, out_dir)
        print("[nivel 4] acciones_rescate.png + features_shap.png/.csv + actividad_controlador.csv")
    except Exception as exc:  # noqa: BLE001
        print(f"[nivel 4 no disponible: {exc}]")

    # Nivel 4b: un features_shap por experimento (CEC por MaxFES + TMLAP)
    try:
        n = features_per_experiment(args.input, args.dim, out_dir)
        print(f"[nivel 4b] features_shap por experimento ({n} figuras)")
    except Exception as exc:  # noqa: BLE001
        print(f"[nivel 4b no disponible: {exc}]")

    # Nivel 4c: comparativa CEC por MaxFES (muestra el shift danger->safety)
    try:
        if features_comparison_cec(args.input, args.dim, out_dir):
            print("[nivel 4c] features_shap_comparativa_cec.png")
    except Exception as exc:  # noqa: BLE001
        print(f"[nivel 4c no disponible: {exc}]")

    # Bloque TMLAP (caso aplicado)
    try:
        tmlap_runs = discover_tmlap_runs(args.input)
        if tmlap_runs:
            print(f"TMLAP detectado: {', '.join(tmlap_runs)}")
            summarize_tmlap(tmlap_runs, out_dir)
            print("[tmlap] tabla_tmlap.png/.csv, boxplots_tmlap.png, veredicto_tmlap.md")
        else:
            print("[tmlap] sin carpetas TMLAP")
    except Exception as exc:  # noqa: BLE001
        print(f"[tmlap no disponible: {exc}]")

    print(f"\nSalidas en: {out_dir}")


if __name__ == "__main__":
    main()
