"""Evolucion temporal de las contribuciones SHAP a lo largo del FES (F12, 1 corrida).

Por cada explicacion SHAP (que ocurre en una intervencion), grafica el share %
de cada una de las 6 senales de control -> muestra COMO va cambiando que senal
"explica" el estancamiento conforme avanza la busqueda. Las explicaciones con
todas las contribuciones nulas se marcan como "sin contribucion".

Uso:  python experiments/F12_estancamiento/plot_evolucion_shap.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BASE = Path(__file__).resolve().parent
FEATS = ["shap_alpha", "shap_beta", "shap_A", "shap_R", "shap_danger_signal", "shap_safety_signal"]
LABELS = ["alpha", "beta", "A", "R", "danger", "safety"]
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#17becf"]

d = pd.read_csv(BASE / "values" / "shap_values.csv").sort_values("fes").reset_index(drop=True)
ab = d[FEATS].abs()
total = ab.sum(axis=1)
share = ab.div(total.replace(0, np.nan), axis=0).fillna(0.0) * 100.0   # % por fila
degenerate = (total == 0).values                                        # SHAP todo cero

x = np.arange(len(d))
fes = d["fes"].astype(int).values

fig, ax = plt.subplots(figsize=(11.5, 6.2))
bottom = np.zeros(len(d))
for col, lab, color in zip(FEATS, LABELS, COLORS):
    vals = share[col].values
    ax.bar(x, vals, bottom=bottom, color=color, label=lab, width=0.72, edgecolor="white", linewidth=0.5)
    bottom += vals

# marcar las explicaciones degeneradas (todo SHAP = 0)
for i in np.where(degenerate)[0]:
    ax.bar(x[i], 100, color="#dddddd", width=0.72, edgecolor="#999", hatch="//", zorder=0)
    ax.text(x[i], 50, "SHAP=0", ha="center", va="center", fontsize=7, color="#555", rotation=90)

ax.set_xticks(x)
ax.set_xticklabels([f"{f:,}" for f in fes], rotation=45, ha="right", fontsize=8)
ax.set_xlabel("FES en que ocurrió la explicación (orden temporal →)")
ax.set_ylabel("Contribución de cada señal  (|SHAP| normalizado, %)")
ax.set_title("F12 (CEC2022) — evolución de las contribuciones SHAP a lo largo de la búsqueda")
ax.set_ylim(0, 100)
ax.legend(title="Señal de control", ncol=6, loc="upper center",
          bbox_to_anchor=(0.5, -0.18), fontsize=8.5, frameon=False)
ax.grid(True, axis="y", alpha=0.25)

# anotar la señal dominante encima de cada barra
for i, row in d.iterrows():
    if not degenerate[i]:
        dom = str(row["dominant_feature"]).replace("_signal", "")
        ax.text(x[i], 101.5, dom, ha="center", va="bottom", fontsize=7, rotation=90, color="#333")

fig.tight_layout()
out = BASE / "grafico_F12_evolucion_shap.png"
fig.savefig(out, dpi=160, bbox_inches="tight")
print("PNG:", out)
