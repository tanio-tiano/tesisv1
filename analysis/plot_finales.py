"""Figuras del informe de resultados (campana 'Experimentos finales').

Genera 6 figuras PNG en docs/figs_resultados/:

  fig1_calidad_cec.png        boxplots base vs shap por funcion (CEC d10 @ 5e5)
  fig2_calidad_mlpap.png      diferencia relativa pareada por escala (MLPAP)
  fig3_convergencia.png       curvas media+-std base vs shap (4 casos)
  fig4_estancamiento.png      stock de estancamiento base vs shap + significancia
  fig5_shap_shares.png        cuota media |SHAP| por senal y dataset
  fig6_ramas.png              eficacia guided vs random por dataset

Uso:
    python -m analysis.plot_finales
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "experiments" / "Experimentos finales"
FIGS = REPO / "docs" / "figs_resultados"

C_BASE = "#4878CF"   # azul
C_SHAP = "#EE854A"   # naranja
SIGNALS = ["alpha", "beta", "A", "R", "danger_signal", "safety_signal"]
SIGNAL_COLORS = ["#4878CF", "#EE854A", "#6ACC65", "#D65F5F", "#956CB4", "#8C613C"]


def scale_of(name):
    return "2XL" if name.startswith("2XL") else ("XL" if name.startswith("XL") else name[0])


def load(cfg, mode):
    return pd.read_csv(cfg / mode / "values" / "summary.csv")


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def paired_p(b, s):
    d = np.asarray(s, float) - np.asarray(b, float)
    if np.allclose(d, 0):
        return 1.0
    try:
        return float(wilcoxon(s, b).pvalue)
    except ValueError:
        return 1.0


# ---------------------------------------------------------------------------
def fig1_calidad_cec():
    cfg = ROOT / "cec2022_d10_fes500000"
    b, s = load(cfg, "base"), load(cfg, "shap")
    funcs = sorted(b["problem"].unique(), key=lambda x: int(x[1:]))
    fig, axes = plt.subplots(3, 4, figsize=(13, 8))
    for ax, f in zip(axes.ravel(), funcs):
        bb = b[b.problem == f].final_fitness.values
        ss = s[s.problem == f].final_fitness.values
        bp = ax.boxplot([bb, ss], tick_labels=["base", "shap"], widths=0.55,
                        patch_artist=True, showfliers=False)
        for patch, color in zip(bp["boxes"], (C_BASE, C_SHAP)):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        ax.set_title(f, fontsize=11)
        ax.tick_params(labelsize=8)
    fig.suptitle("CEC2022 D=10, MaxFES=5e5 - fitness final por funcion (30 corridas)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIGS / "fig1_calidad_cec.png", dpi=150)
    plt.close(fig)


def fig2_calidad_mlpap():
    ml = ROOT / "mlpap_all_fes500000_r30"
    data = {sc: [] for sc in ("S", "M", "L", "XL", "2XL")}
    for d in sorted(p for p in ml.iterdir() if p.is_dir() and not p.name.startswith("_")):
        b = load(d, "base").sort_values("run_id")
        s = load(d, "shap").sort_values("run_id")
        rel = 100.0 * (s.final_fitness.values - b.final_fitness.values) / b.final_fitness.values
        data[scale_of(d.name)].extend(rel.tolist())
    fig, ax = plt.subplots(figsize=(9, 5))
    scales = ["S", "M", "L", "XL", "2XL"]
    bp = ax.boxplot([data[sc] for sc in scales], tick_labels=scales, widths=0.5,
                    patch_artist=True, showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor(C_SHAP)
        patch.set_alpha(0.7)
    ax.axhline(0.0, color="black", lw=1, ls="--")
    for i, sc in enumerate(scales, start=1):
        m = float(np.mean(data[sc]))
        ax.annotate(f"media {m:+.1f}%", (i, m), textcoords="offset points",
                    xytext=(0, -18), ha="center", fontsize=9)
    ax.set_ylabel("diferencia relativa pareada (shap - base) / base  [%]")
    ax.set_xlabel("escala MLPAP (n = 20 / 50 / 100 / 250 / 1000)")
    ax.set_title("MLPAP - diferencia relativa pareada por corrida (negativo = shap mejor)")
    fig.tight_layout()
    fig.savefig(FIGS / "fig2_calidad_mlpap.png", dpi=150)
    plt.close(fig)


def _mean_std_curve(curves_dir, grid):
    ys = []
    for f in sorted(curves_dir.glob("conv_curve_*.csv")):
        df = pd.read_csv(f)
        ys.append(np.interp(grid, df["fes"].values, df["best_fitness"].values))
    ys = np.asarray(ys)
    return ys.mean(axis=0), ys.std(axis=0)


def fig3_convergencia():
    cases = [
        ("MLPAP L01 (n=100)", ROOT / "mlpap_all_fes500000_r30" / "L01"),
        ("MLPAP 2XL01 (n=1000)", ROOT / "mlpap_all_fes500000_r30" / "2XL01"),
        ("CEC2022 F6 D=10 @ 5e5", ROOT / "cec2022_d10_fes500000"),
        ("TMLAP dura (24c x 8h)", ROOT / "ablation_b4_dura_30"),
    ]
    grid = np.linspace(0, 500000, 250)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (title, cfg) in zip(axes.ravel(), cases):
        for mode, color in (("base", C_BASE), ("shap", C_SHAP)):
            cdir = cfg / mode / "curves"
            if "F6" in title:  # solo la funcion F6 dentro de la config CEC
                files = sorted(cdir.glob("conv_curve_F6_*.csv"))
                ys = np.asarray([
                    np.interp(grid, pd.read_csv(f)["fes"].values,
                              pd.read_csv(f)["best_fitness"].values)
                    for f in files
                ])
                mean, std = ys.mean(axis=0), ys.std(axis=0)
            else:
                mean, std = _mean_std_curve(cdir, grid)
            ax.plot(grid, mean, color=color, label=mode, lw=1.8)
            ax.fill_between(grid, np.maximum(mean - std, 1e-12), mean + std,
                            color=color, alpha=0.18)
        if "F6" in title:   # el pico inicial aplasta la escala lineal
            ax.set_yscale("log")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("FES")
        ax.set_ylabel("best fitness (media +- std)")
        ax.legend(fontsize=9)
    fig.suptitle("Convergencia base vs shap (30 corridas pareadas)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIGS / "fig3_convergencia.png", dpi=150)
    plt.close(fig)


def _stock_groups():
    """Devuelve lista (label, df_base, df_shap) pareadas por unidad."""
    groups = []
    ml = ROOT / "mlpap_all_fes500000_r30"
    buf = {}
    for d in sorted(p for p in ml.iterdir() if p.is_dir() and not p.name.startswith("_")):
        sc = scale_of(d.name)
        bb = load(d, "base"); bb["u"] = d.name
        ss = load(d, "shap"); ss["u"] = d.name
        buf.setdefault(sc, ([], []))
        buf[sc][0].append(bb)
        buf[sc][1].append(ss)
    for sc in ("S", "M", "L", "XL", "2XL"):
        b = pd.concat(buf[sc][0]).sort_values(["u", "run_id"])
        s = pd.concat(buf[sc][1]).sort_values(["u", "run_id"])
        groups.append((f"MLPAP-{sc}", b, s))
    for dim in (10, 20):
        bs, sl = [], []
        for fes in (5000, 50000, 500000, 5000000):
            cfg = ROOT / f"cec2022_d{dim}_fes{fes}"
            bb = load(cfg, "base"); bb["u"] = fes
            ss = load(cfg, "shap"); ss["u"] = fes
            bs.append(bb); sl.append(ss)
        b = pd.concat(bs).sort_values(["u", "problem", "run_id"])
        s = pd.concat(sl).sort_values(["u", "problem", "run_id"])
        groups.append((f"CEC-d{dim}", b, s))
    cfg = ROOT / "ablation_b4_dura_30"
    groups.append(("TMLAP", load(cfg, "base").sort_values("run_id"),
                   load(cfg, "shap").sort_values("run_id")))
    return groups


def fig4_estancamiento():
    groups = _stock_groups()
    labels = [g[0] for g in groups]
    x = np.arange(len(groups))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A: antiguedad de la ultima mejora (staleness), % de cambio.
    changes, ps = [], []
    for _, b, s in groups:
        col = "mean_fes_since_improve_at_end"
        changes.append(100.0 * (s[col].mean() / b[col].mean() - 1.0))
        ps.append(paired_p(b[col].values, s[col].values))
    bars = ax1.bar(x, changes, color=[C_SHAP if c < 0 else "#AAAAAA" for c in changes])
    for xi, c, p in zip(x, changes, ps):
        ax1.annotate(f"{stars(p)}", (xi, c), textcoords="offset points",
                     xytext=(0, 4 if c >= 0 else -14), ha="center", fontsize=9)
    ax1.axhline(0, color="black", lw=1)
    ax1.set_xticks(x, labels, rotation=30, ha="right")
    ax1.set_ylabel("cambio de la antiguedad de la ultima mejora  [%]")
    ax1.set_title("(a) 'Frescura' de los agentes al final\n(negativo = shap mitiga; Wilcoxon pareado)")

    # Panel B: agentes estancados al final (de 30).
    w = 0.38
    b_means = [g[1].n_stagnant_at_end.mean() for g in groups]
    s_means = [g[2].n_stagnant_at_end.mean() for g in groups]
    ax2.bar(x - w / 2, b_means, w, color=C_BASE, label="base")
    ax2.bar(x + w / 2, s_means, w, color=C_SHAP, label="shap")
    for xi, (_, b, s) in zip(x, groups):
        p = paired_p(b.n_stagnant_at_end.values, s.n_stagnant_at_end.values)
        ax2.annotate(stars(p), (xi, max(b.n_stagnant_at_end.mean(),
                                        s.n_stagnant_at_end.mean())),
                     textcoords="offset points", xytext=(0, 4), ha="center", fontsize=9)
    ax2.set_xticks(x, labels, rotation=30, ha="right")
    ax2.set_ylabel("agentes estancados al final (de 30)")
    ax2.set_title("(b) Stock de estancamiento al cierre")
    ax2.legend()
    fig.suptitle("Mitigacion del estancamiento - estado final base vs shap", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIGS / "fig4_estancamiento.png", dpi=150)
    plt.close(fig)


def fig5_shap_shares():
    shares_csv = ROOT / "_analysis" / "interpret_shares.csv"
    df = pd.read_csv(shares_csv)
    order = ["MLPAP-S", "MLPAP-M", "MLPAP-L", "MLPAP-XL", "MLPAP-2XL",
             "CEC-d10", "CEC-d20", "TMLAP-dura"]
    df = df.set_index("dataset").loc[order].reset_index()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    left = np.zeros(len(df))
    for sig, color in zip(SIGNALS, SIGNAL_COLORS):
        vals = df[f"share_{sig}"].values
        ax.barh(df["dataset"], vals, left=left, color=color, label=sig)
        left += vals
    ax.invert_yaxis()
    ax.set_xlabel("cuota media |SHAP| (las 6 suman 1)")
    ax.set_title("Atribucion SHAP por senal de control - 56 547 explicaciones")
    ax.legend(ncol=3, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGS / "fig5_shap_shares.png", dpi=150)
    plt.close(fig)


def fig6_ramas():
    eff = pd.read_csv(ROOT / "_analysis" / "branch_effectiveness.csv")
    datasets = ["MLPAP", "CEC-d10", "CEC-d20", "TMLAP-dura"]
    x = np.arange(len(datasets))
    w = 0.38
    g = [float(eff[(eff.dataset == d) & (eff.action == "reinit_guided")].improved_pct.iloc[0])
         for d in datasets]
    r = [float(eff[(eff.dataset == d) & (eff.action == "reinit_random")].improved_pct.iloc[0])
         for d in datasets]
    ng = [int(eff[(eff.dataset == d) & (eff.action == "reinit_guided")].n.iloc[0])
          for d in datasets]
    nr = [int(eff[(eff.dataset == d) & (eff.action == "reinit_random")].n.iloc[0])
          for d in datasets]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2, g, w, color=C_SHAP, label="reinit_guided (signo SHAP)")
    ax.bar(x + w / 2, r, w, color=C_BASE, label="reinit_random")
    for xi, (gv, rv, ngv, nrv) in enumerate(zip(g, r, ng, nr)):
        ax.annotate(f"{gv:.0f}%\n(n={ngv})", (xi - w / 2, gv),
                    textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8)
        ax.annotate(f"{rv:.0f}%\n(n={nrv})", (xi + w / 2, rv),
                    textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8)
    ax.set_xticks(x, datasets)
    ax.set_ylabel("% de intervenciones que mejoran el fitness del agente")
    ax.set_title("Eficacia de las ramas de rescate (la explicacion prescribe)")
    ax.set_ylim(0, 85)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "fig6_ramas.png", dpi=150)
    plt.close(fig)


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    fig1_calidad_cec()
    print("fig1 OK")
    fig2_calidad_mlpap()
    print("fig2 OK")
    fig3_convergencia()
    print("fig3 OK")
    fig4_estancamiento()
    print("fig4 OK")
    fig5_shap_shares()
    print("fig5 OK")
    fig6_ramas()
    print("fig6 OK")
    print(f"Figuras en: {FIGS}")


if __name__ == "__main__":
    main()
