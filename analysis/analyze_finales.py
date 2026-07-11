"""Analisis integral de la campana 'Experimentos finales' (2026-07).

Cubre las tres lineas del estudio, sobre MLPAP (100 instancias), CEC2022
(d10 y d20 x 4 MaxFES) y TMLAP dura:

1. CALIDAD (Esp. 4): Wilcoxon signed-rank pareado shap vs base, bilateral,
   con correccion de Holm por familia de comparaciones (12 funciones por
   config CEC; 20 instancias por escala MLPAP). Veredicto (+|=|-).
2. MITIGACION (obj. general): n_stagnation_episodes base vs shap (pareado).
3. INTERPRETABILIDAD (Esp. 3): contribuciones SHAP por senal y eficacia de
   las ramas guided/random (improved%).

Salidas: tablas CSV en '<root>/_analysis/' + resumen por consola.

Uso:
    python -m analysis.analyze_finales
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1] / "experiments" / "Experimentos finales"
OUT = ROOT / "_analysis"
ALPHA = 0.05
SIGNALS = ["alpha", "beta", "A", "R", "danger_signal", "safety_signal"]
CEC_DIMS = (10, 20)
CEC_FES = (5000, 50000, 500000, 5000000)


# ---------------------------------------------------------------------------
# Helpers estadisticos.
# ---------------------------------------------------------------------------
def paired_p(base_vals, shap_vals):
    d = np.asarray(shap_vals, dtype=float) - np.asarray(base_vals, dtype=float)
    if np.allclose(d, 0.0):
        return 1.0
    try:
        return float(wilcoxon(shap_vals, base_vals).pvalue)
    except ValueError:
        return 1.0


def holm(pvals, alpha=ALPHA):
    """Devuelve bool[n]: significativo tras correccion de Holm."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    sig = np.zeros(m, dtype=bool)
    for rank, idx in enumerate(order):
        if p[idx] <= alpha / (m - rank):
            sig[idx] = True
        else:
            break
    return sig


def verdict(diff_mean, significant):
    """'+' shap mejor (minimizacion), '-' peor, '=' sin diferencia."""
    if not significant:
        return "="
    return "+" if diff_mean < 0 else "-"


def load(cfg_dir, mode):
    return pd.read_csv(cfg_dir / mode / "values" / "summary.csv")


def scale_of(inst_name):
    if inst_name.startswith("2XL"):
        return "2XL"
    if inst_name.startswith("XL"):
        return "XL"
    return inst_name[0]


# ---------------------------------------------------------------------------
# 1) CEC2022: por funcion x dim x MaxFES.
# ---------------------------------------------------------------------------
def analyze_cec():
    func_rows, cfg_rows = [], []
    for dim in CEC_DIMS:
        for fes in CEC_FES:
            cfg = ROOT / f"cec2022_d{dim}_fes{fes}"
            b_all, s_all = load(cfg, "base"), load(cfg, "shap")
            recs, pvals = [], []
            funcs = sorted(b_all["problem"].unique(), key=lambda x: int(x[1:]))
            for f in funcs:
                b = b_all[b_all.problem == f].sort_values("run_id")
                s = s_all[s_all.problem == f].sort_values("run_id")
                p = paired_p(b.final_fitness.values, s.final_fitness.values)
                p_ep = paired_p(b.n_stagnation_episodes.values,
                                s.n_stagnation_episodes.values)
                recs.append({
                    "dim": dim, "max_fes": fes, "func": f, "p": p,
                    "base_mean": b.final_fitness.mean(),
                    "shap_mean": s.final_fitness.mean(),
                    "diff_mean": s.final_fitness.mean() - b.final_fitness.mean(),
                    "ep_base": b.n_stagnation_episodes.mean(),
                    "ep_shap": s.n_stagnation_episodes.mean(),
                    "p_ep": p_ep,
                })
                pvals.append(p)
            sig = holm(pvals)
            for r, sg in zip(recs, sig):
                r["sig_holm"] = bool(sg)
                r["verdict"] = verdict(r["diff_mean"], sg)
                func_rows.append(r)
            plus = sum(1 for r in recs if r["verdict"] == "+")
            minus = sum(1 for r in recs if r["verdict"] == "-")
            ep_red = 100.0 * (1 - sum(r["ep_shap"] for r in recs)
                              / max(sum(r["ep_base"] for r in recs), 1e-12))
            cfg_rows.append({
                "dim": dim, "max_fes": fes,
                "plus": plus, "eq": len(recs) - plus - minus, "minus": minus,
                "ep_reduction_pct": ep_red,
            })
    fdf, cdf = pd.DataFrame(func_rows), pd.DataFrame(cfg_rows)
    fdf.to_csv(OUT / "cec_by_function.csv", index=False)
    cdf.to_csv(OUT / "cec_by_config.csv", index=False)
    return fdf, cdf


# ---------------------------------------------------------------------------
# 2) MLPAP: por instancia y por escala.
# ---------------------------------------------------------------------------
def analyze_mlpap():
    ml = ROOT / "mlpap_all_fes500000_r30"
    rows = []
    for d in sorted(p for p in ml.iterdir() if p.is_dir() and not p.name.startswith("_")):
        b = load(d, "base").sort_values("run_id")
        s = load(d, "shap").sort_values("run_id")
        rows.append({
            "inst": d.name, "scale": scale_of(d.name),
            "p": paired_p(b.final_fitness.values, s.final_fitness.values),
            "base_mean": b.final_fitness.mean(), "base_std": b.final_fitness.std(ddof=1),
            "shap_mean": s.final_fitness.mean(), "shap_std": s.final_fitness.std(ddof=1),
            "diff_mean": s.final_fitness.mean() - b.final_fitness.mean(),
            "rel_diff_pct": 100.0 * (s.final_fitness.mean() - b.final_fitness.mean())
                            / max(abs(b.final_fitness.mean()), 1e-12),
            "ep_base": b.n_stagnation_episodes.mean(),
            "ep_shap": s.n_stagnation_episodes.mean(),
            "p_ep": paired_p(b.n_stagnation_episodes.values,
                             s.n_stagnation_episodes.values),
            "interv": s.interventions.mean(),
            "guided": s.n_reinit_guided.mean(),
            "random": s.n_reinit_random.mean(),
        })
    mi = pd.DataFrame(rows)
    # Holm por escala (familias de 20 comparaciones).
    mi["sig_holm"] = False
    for scale, g in mi.groupby("scale"):
        mi.loc[g.index, "sig_holm"] = holm(g.p.values)
    mi["verdict"] = [verdict(dm, sg) for dm, sg in zip(mi.diff_mean, mi.sig_holm)]
    mi.to_csv(OUT / "mlpap_by_instance.csv", index=False)

    agg_rows = []
    for scale in ("S", "M", "L", "XL", "2XL"):
        g = mi[mi.scale == scale]
        agg_rows.append({
            "scale": scale, "n_inst": len(g),
            "plus": int((g.verdict == "+").sum()),
            "eq": int((g.verdict == "=").sum()),
            "minus": int((g.verdict == "-").sum()),
            "mean_rel_diff_pct": g.rel_diff_pct.mean(),
            "ep_base_mean": g.ep_base.mean(), "ep_shap_mean": g.ep_shap.mean(),
            "ep_reduction_pct": 100.0 * (1 - g.ep_shap.mean() / max(g.ep_base.mean(), 1e-12)),
            "n_inst_ep_signif": int(((g.p_ep < ALPHA) & (g.ep_shap < g.ep_base)).sum()),
            "interv_mean": g.interv.mean(),
            "guided_share_pct": 100.0 * g.guided.mean()
                                / max(g.guided.mean() + g.random.mean(), 1e-12),
        })
    ma = pd.DataFrame(agg_rows)
    ma.to_csv(OUT / "mlpap_by_scale.csv", index=False)
    return mi, ma


# ---------------------------------------------------------------------------
# 3) TMLAP dura.
# ---------------------------------------------------------------------------
def analyze_tmlap():
    cfg = ROOT / "ablation_b4_dura_30"
    b = load(cfg, "base").sort_values("run_id")
    s = load(cfg, "shap").sort_values("run_id")
    row = {
        "p": paired_p(b.final_fitness.values, s.final_fitness.values),
        "base_mean": b.final_fitness.mean(), "base_std": b.final_fitness.std(ddof=1),
        "shap_mean": s.final_fitness.mean(), "shap_std": s.final_fitness.std(ddof=1),
        "ep_base": b.n_stagnation_episodes.mean(),
        "ep_shap": s.n_stagnation_episodes.mean(),
        "p_ep": paired_p(b.n_stagnation_episodes.values, s.n_stagnation_episodes.values),
        "interv": s.interventions.mean(),
    }
    pd.DataFrame([row]).to_csv(OUT / "tmlap_dura.csv", index=False)
    return row


# ---------------------------------------------------------------------------
# 4) Interpretabilidad: shap_values + eficacia de ramas (controller_events).
# ---------------------------------------------------------------------------
def _shap_shares(df):
    cols = [f"shap_{s}" for s in SIGNALS]
    ab = df[cols].abs()
    tot = ab.sum(axis=1).replace(0, np.nan)
    shares = ab.div(tot, axis=0)
    out = {s: float(shares[f"shap_{s}"].mean()) for s in SIGNALS}
    dom = df[cols].abs().idxmax(axis=1).str.replace("shap_", "", regex=False)
    return out, dom.value_counts().to_dict(), len(df)


def analyze_interpretability():
    rows = []
    # MLPAP por escala.
    ml = ROOT / "mlpap_all_fes500000_r30"
    frames = {}
    for d in sorted(p for p in ml.iterdir() if p.is_dir() and not p.name.startswith("_")):
        f = d / "shap" / "values" / "shap_values.csv"
        if f.exists():
            frames.setdefault(scale_of(d.name), []).append(pd.read_csv(f))
    for scale in ("S", "M", "L", "XL", "2XL"):
        if scale not in frames:
            continue
        df = pd.concat(frames[scale], ignore_index=True)
        shares, dom, n = _shap_shares(df)
        rows.append({"dataset": f"MLPAP-{scale}", "n_explicaciones": n,
                     **{f"share_{s}": shares[s] for s in SIGNALS},
                     "dominante_top": max(dom, key=dom.get)})
    # CEC por dim (agregando los 4 budgets).
    for dim in CEC_DIMS:
        parts = []
        for fes in CEC_FES:
            f = ROOT / f"cec2022_d{dim}_fes{fes}" / "shap" / "values" / "shap_values.csv"
            if f.exists():
                parts.append(pd.read_csv(f))
        df = pd.concat(parts, ignore_index=True)
        shares, dom, n = _shap_shares(df)
        rows.append({"dataset": f"CEC-d{dim}", "n_explicaciones": n,
                     **{f"share_{s}": shares[s] for s in SIGNALS},
                     "dominante_top": max(dom, key=dom.get)})
    # TMLAP.
    f = ROOT / "ablation_b4_dura_30" / "shap" / "values" / "shap_values.csv"
    df = pd.read_csv(f)
    shares, dom, n = _shap_shares(df)
    rows.append({"dataset": "TMLAP-dura", "n_explicaciones": n,
                 **{f"share_{s}": shares[s] for s in SIGNALS},
                 "dominante_top": max(dom, key=dom.get)})
    idf = pd.DataFrame(rows)
    idf.to_csv(OUT / "interpret_shares.csv", index=False)

    # Eficacia de ramas (guided vs random) por dataset.
    eff_rows = []

    def _events_eff(paths, label):
        parts = [pd.read_csv(p) for p in paths if p.exists()]
        if not parts:
            return
        ev = pd.concat(parts, ignore_index=True)
        for action in ("reinit_guided", "reinit_random"):
            sub = ev[ev.action == action]
            if len(sub):
                eff_rows.append({"dataset": label, "action": action, "n": len(sub),
                                 "improved_pct": 100.0 * sub.improved.mean()})

    _events_eff([d / "shap" / "values" / "controller_events.csv"
                 for d in ml.iterdir() if d.is_dir()], "MLPAP")
    for dim in CEC_DIMS:
        _events_eff([ROOT / f"cec2022_d{dim}_fes{fes}" / "shap" / "values" / "controller_events.csv"
                     for fes in CEC_FES], f"CEC-d{dim}")
    _events_eff([ROOT / "ablation_b4_dura_30" / "shap" / "values" / "controller_events.csv"],
                "TMLAP-dura")
    edf = pd.DataFrame(eff_rows)
    edf.to_csv(OUT / "branch_effectiveness.csv", index=False)
    return idf, edf


# ---------------------------------------------------------------------------
def main():
    OUT.mkdir(exist_ok=True)
    fmt = lambda v: f"{v:.3f}"

    print("=" * 100)
    print("1) CALIDAD - CEC2022 (Wilcoxon pareado + Holm por config; '+'=shap mejor)")
    print("=" * 100)
    fdf, cdf = analyze_cec()
    print(cdf.to_string(index=False, float_format=fmt))
    sig_funcs = fdf[fdf.verdict != "="]
    print("\nFunciones con diferencia significativa (tras Holm):")
    if len(sig_funcs):
        print(sig_funcs[["dim", "max_fes", "func", "base_mean", "shap_mean",
                         "p", "verdict"]].to_string(index=False, float_format=fmt))
    else:
        print("  (ninguna)")

    print("\n" + "=" * 100)
    print("2) CALIDAD - MLPAP por escala (Holm dentro de cada escala, 20 instancias)")
    print("=" * 100)
    mi, ma = analyze_mlpap()
    print(ma.to_string(index=False, float_format=fmt))
    sig_inst = mi[mi.verdict != "="]
    print("\nInstancias con diferencia significativa (tras Holm):")
    if len(sig_inst):
        print(sig_inst[["inst", "scale", "base_mean", "shap_mean", "rel_diff_pct",
                        "p", "verdict"]].to_string(index=False, float_format=fmt))
    else:
        print("  (ninguna)")

    print("\n" + "=" * 100)
    print("3) CALIDAD - TMLAP dura")
    print("=" * 100)
    t = analyze_tmlap()
    print(f"  base  {t['base_mean']:.3f} +- {t['base_std']:.3f}")
    print(f"  shap  {t['shap_mean']:.3f} +- {t['shap_std']:.3f}   p={t['p']:.4f}")
    print(f"  episodios: base {t['ep_base']:.1f} vs shap {t['ep_shap']:.1f}  "
          f"(p={t['p_ep']:.4f}; interv media {t['interv']:.1f})")

    print("\n" + "=" * 100)
    print("4) MITIGACION - episodios de estancamiento (variable de respuesta)")
    print("=" * 100)
    ep_cec = fdf.groupby(["dim", "max_fes"]).agg(
        ep_base=("ep_base", "mean"), ep_shap=("ep_shap", "mean"),
        n_signif=("p_ep", lambda s: int((s < ALPHA).sum()))).reset_index()
    ep_cec["reduccion_pct"] = 100.0 * (1 - ep_cec.ep_shap / ep_cec.ep_base)
    print("CEC (medias por config; n_signif = funciones con p_ep<0.05 de 12):")
    print(ep_cec.to_string(index=False, float_format=fmt))
    print("\nMLPAP (por escala; n_inst_ep_signif = instancias con reduccion p<0.05 de 20):")
    print(ma[["scale", "ep_base_mean", "ep_shap_mean", "ep_reduction_pct",
              "n_inst_ep_signif"]].to_string(index=False, float_format=fmt))

    print("\n" + "=" * 100)
    print("5) INTERPRETABILIDAD - cuota media |SHAP| por senal y senal dominante")
    print("=" * 100)
    idf, edf = analyze_interpretability()
    print(idf.to_string(index=False, float_format=fmt))
    print("\nEficacia de las ramas (improved = mejoro el fitness del agente):")
    print(edf.to_string(index=False, float_format=fmt))

    print(f"\nCSV guardados en: {OUT}")


if __name__ == "__main__":
    main()
