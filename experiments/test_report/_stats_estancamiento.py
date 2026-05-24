"""Recopila estadisticas de deteccion de estancamiento e intervencion SHAP a
partir de la telemetria de experiments/test_report/. Solo lee CSV e imprime
cifras (no escribe nada). Insumo para el informe markdown.
"""
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
FEATURES = ["alpha", "beta", "A", "R", "danger_signal", "safety_signal"]
DATASETS = [
    ("CEC2022", "cec"),
    ("TMLAP simple", "tmlap_simple"),
    ("TMLAP mediana", "tmlap_mediana"),
    ("TMLAP dura", "tmlap_dura"),
]


def load(folder, name):
    p = BASE / folder / "shap" / "values" / name
    return pd.read_csv(p) if p.exists() else None


def base_shap(folder):
    out = {}
    for tag in ("base", "shap"):
        p = BASE / folder / tag / "values" / "summary.csv"
        out[tag] = pd.read_csv(p) if p.exists() else None
    return out


print("=" * 70)
for tag, folder in DATASETS:
    ev = load(folder, "controller_events.csv")
    summ = load(folder, "summary.csv")
    nonev = load(folder, "controller_non_events.csv")
    if ev is None:
        print(f"[{tag}] sin eventos"); continue
    runs = ev.run_id.nunique()
    maxfes = sorted(ev.max_fes.unique())
    nA = int((ev.action == "reinit_random").sum())
    nB = int((ev.action == "reinit_guided").sum())
    impr = int(ev.improved.sum())
    imprA = int(ev[ev.action == "reinit_random"].improved.sum())
    imprB = int(ev[ev.action == "reinit_guided"].improved.sum())
    fsi = ev.fes_since_improve
    first = ev.sort_values("fes").groupby("run_id").first()
    # distribucion temporal (cuartiles de fes/max_fes)
    frac = ev.fes / ev.max_fes
    q = pd.cut(frac, [0, .25, .5, .75, 1.0], labels=["0-25%", "25-50%", "50-75%", "75-100%"])
    qd = q.value_counts().reindex(["0-25%", "25-50%", "50-75%", "75-100%"])
    feat = ev.shap_dominant_feature.value_counts()
    share_ge90 = int((ev.dominant_share >= 0.90).sum())

    print(f"### {tag}  | runs={runs} maxfes={maxfes}")
    print(f"  activaciones={len(ev)}  (ramaA={nA}, ramaB={nB})")
    print(f"  por corrida prom = {len(ev)/runs:.1f}")
    print(f"  mejoraron = {impr}/{len(ev)} ({100*impr/len(ev):.1f}%)  | A={imprA}/{nA} ({100*imprA/max(nA,1):.1f}%)  B={imprB}/{nB} ({100*imprB/max(nB,1):.1f}%)")
    print(f"  fes_since_improve: min={int(fsi.min())} mediana={int(fsi.median())} prom={fsi.mean():.0f} max={int(fsi.max())}")
    print(f"  1a activacion fes: min={int(first.fes.min())} prom={first.fes.mean():.0f} max={int(first.fes.max())}")
    print(f"  distribucion temporal: " + " ".join(f"{k}={int(v)}" for k, v in qd.items()))
    print(f"  share>=0.90 (=>ramaB): {share_ge90}  ({100*share_ge90/len(ev):.1f}%)")
    print(f"  feature dominante (top): " + ", ".join(f"{k}={v}" for k, v in feat.head(6).items()))
    if summ is not None:
        print(f"  fes_shap/corrida: prom={summ.fes_shap.mean():.0f}  ({100*summ.fes_shap.mean()/summ.max_fes.iloc[0]:.1f}% de MaxFES)")
        print(f"  intervenciones/corrida: prom={summ.interventions.mean():.1f}")
    if nonev is not None and not nonev.empty:
        print(f"  bloqueos (non-events)={len(nonev)}: " + ", ".join(f"{k}={v}" for k, v in nonev.reason.value_counts().head(8).items()))
    else:
        print("  bloqueos (non-events)=0")
    print("-" * 70)

# SHAP values: que senal domina globalmente (|shap| medio) en CEC
print("\n### |SHAP| medio por senal (CEC2022)")
sh = load("cec", "shap_values.csv")
if sh is not None:
    cols = [f"shap_{f}" for f in FEATURES if f"shap_{f}" in sh.columns]
    means = {c.replace("shap_", ""): float(sh[c].abs().mean()) for c in cols}
    tot = sum(means.values()) or 1.0
    for k, v in sorted(means.items(), key=lambda kv: -kv[1]):
        print(f"  {k:14s} |SHAP|medio={v:.4g}  share={100*v/tot:.1f}%")

# Comparativo base vs shap (final fitness y gap)
print("\n### Base vs SHAP (final_fitness medio)")
for tag, folder in DATASETS:
    bs = base_shap(folder)
    if bs["base"] is None or bs["shap"] is None:
        print(f"  [{tag}] incompleto"); continue
    for label, df in (("CEC por problema", bs),):
        pass
    b = bs["base"].groupby("problem").final_fitness.mean()
    s = bs["shap"].groupby("problem").final_fitness.mean()
    common = b.index.intersection(s.index)
    wins = int((s[common] < b[common]).sum())
    print(f"  [{tag}] problemas={len(common)}  SHAP mejor en {wins}/{len(common)}  "
          f"| media base={b[common].mean():.4g}  media shap={s[common].mean():.4g}")
print("=" * 70)
