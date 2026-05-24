"""Grafico de convergencia de F12 (CEC2022) marcando los puntos donde se activo
el detector de estancamiento, MAS un panel inferior con el tramo real de
estancamiento de cada activacion (una corrida WO+SHAP, MaxFES=50000).

Panel superior: convergencia del GBest + punto de cada activacion.
Panel inferior (alineado por FES): por cada activacion, una barra desde la
ultima mejora del agente hasta la deteccion. El tramo solido = la VENTANA de
estancamiento (5000 FES = 10%); el tramo tenue = espera extra hasta intervenir
(por cooldowns/orden). Color = rama (A aleatorio / B guiado); relleno del
marcador = el agente mejoro tras el reinit.

Uso:  python experiments/F12_estancamiento/plot_estancamiento.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

BASE = Path(__file__).resolve().parent
OPT = 2700.0          # optimo de F12 (CEC2022)
MAXFES = 50000
WINDOW = 5000         # 10% de MaxFES (ventana de estancamiento)
GUARD = 2500          # 5% de MaxFES (guard window)
C_A, C_B = "#d4631a", "#7c4fc0"   # naranja=random, violeta=guiado

cur = np.loadtxt(BASE / "curves" / "conv_curve_F12_fes50000_run1.csv",
                 delimiter=",", skiprows=1)
ev = pd.read_csv(BASE / "values" / "controller_events.csv").sort_values("fes").reset_index(drop=True)
ev["last_improve"] = ev["fes"] - ev["fes_since_improve"]
fes_c, best_c = cur[:, 0], cur[:, 1]

fig, (ax, axg) = plt.subplots(
    2, 1, figsize=(11.2, 8.6), sharex=True,
    gridspec_kw={"height_ratios": [3.0, 2.0], "hspace": 0.12},
)

# ---------------- Panel superior: convergencia ----------------
ax.plot(fes_c, best_c, color="#1f77b4", lw=1.7, zorder=3)
ax.axhline(OPT, color="#2e7d32", ls="--", lw=1.1, zorder=1)
ax.set_ylim(OPT - 12, best_c.max() + (best_c.max() - OPT) * 0.06)
y0, y1 = ax.get_ylim()
ax.axvspan(0, GUARD, color="#000000", alpha=0.04, zorder=0)
ax.axvline(GUARD, color="#888", ls=":", lw=1.0, zorder=1)
ax.text(GUARD + 350, y1, "guard 5% (2500)", color="#666", fontsize=7.5,
        va="top", ha="left", rotation=90)
ax.text(MAXFES * 0.995, OPT, f"  óptimo F12 = {OPT:.0f}", color="#2e7d32",
        va="bottom", ha="right", fontsize=9)

for _, r in ev.iterrows():
    f = float(r["fes"])
    yv = float(np.interp(f, fes_c, best_c))
    col = C_B if r["action"] == "reinit_guided" else C_A
    improved = bool(r["improved"])
    ax.vlines(f, y0, yv, color=col, lw=0.9, alpha=0.35, zorder=2)
    ax.scatter([f], [yv], s=140, marker="o", zorder=6,
               facecolor=col if improved else "white",
               edgecolor=col, linewidths=1.9)
    ax.annotate(f"a{int(r['agent_index'])}", (f, yv), textcoords="offset points",
                xytext=(0, 10), ha="center", fontsize=7.3, color=col, weight="bold")

ax.set_ylabel("Aptitud del mejor global (GBest)")
ax.set_title("F12 (CEC2022) — convergencia y ventana de estancamiento por activación")
ax.grid(True, alpha=0.25)
ax.legend(handles=[
    Line2D([0], [0], color="#1f77b4", lw=1.7, label="Mejor global (GBest)"),
    Line2D([0], [0], color=C_A, marker="o", ls="", mfc=C_A, mec=C_A, label="Rama A · aleatorio"),
    Line2D([0], [0], color=C_B, marker="o", ls="", mfc=C_B, mec=C_B, label="Rama B · guiado"),
    Line2D([0], [0], color="#555", marker="o", ls="", mfc="#555", mec="#555", label="relleno = mejoró"),
    Line2D([0], [0], color="#555", marker="o", ls="", mfc="white", mec="#555", label="hueco = no mejoró"),
], loc="upper right", fontsize=8.0, framealpha=0.96, ncol=1)

# ---------------- Panel inferior: tramo de estancamiento ----------------
n = len(ev)
axg.axvspan(0, GUARD, color="#000000", alpha=0.04, zorder=0)
yticklabels = []
for k, r in ev.iterrows():
    yk = n - 1 - k  # el primero (menor FES) arriba
    li = float(r["last_improve"])
    det = float(r["fes"])
    win_end = li + WINDOW
    col = C_B if r["action"] == "reinit_guided" else C_A
    improved = bool(r["improved"])
    # ventana de estancamiento (5000) -> solido
    axg.barh(yk, WINDOW, left=li, height=0.55, color=col, alpha=0.9,
             zorder=3, edgecolor="none")
    # espera extra hasta intervenir -> tenue
    if det > win_end:
        axg.barh(yk, det - win_end, left=win_end, height=0.55, color=col,
                 alpha=0.22, zorder=3, edgecolor="none")
    # marca del umbral (cuando cumplio la ventana)
    axg.vlines(win_end, yk - 0.30, yk + 0.30, color="#222", lw=0.9, zorder=4)
    # marcador de deteccion (= intervencion)
    axg.scatter([det], [yk], s=70, marker="o", zorder=5,
                facecolor=col if improved else "white", edgecolor=col, linewidths=1.6)
    # longitud total del estancamiento
    axg.annotate(f"{int(r['fes_since_improve'])} FES", (det, yk),
                 textcoords="offset points", xytext=(7, 0), va="center",
                 ha="left", fontsize=6.8, color="#444")
    yticklabels.append((yk, f"a{int(r['agent_index'])}"))

axg.set_yticks([t[0] for t in yticklabels])
axg.set_yticklabels([t[1] for t in yticklabels], fontsize=7.5)
axg.set_ylim(-0.7, n - 0.3)
axg.set_xlim(0, MAXFES)
axg.set_xlabel("Evaluaciones de la función objetivo (FES)")
axg.set_ylabel("Activación (agente)")
axg.grid(True, axis="x", alpha=0.25)
axg.legend(handles=[
    Patch(facecolor="#888", alpha=0.9, label="ventana de estancamiento = 5.000 FES (10%)"),
    Patch(facecolor="#888", alpha=0.22, label="espera extra hasta intervenir"),
    Line2D([0], [0], color="#222", lw=0.9, label="umbral cumplido (10%)"),
], loc="lower right", fontsize=7.6, framealpha=0.96)

# caja informativa (en el panel superior)
n_a = int((ev["action"] == "reinit_random").sum())
n_b = int((ev["action"] == "reinit_guided").sum())
n_imp = int(ev["improved"].sum())
info = (f"MaxFES = {MAXFES:,}   ·   ventana = {WINDOW:,} (10%)\n"
        f"Activaciones = {n}  ({n_a} aleatorio / {n_b} guiado)   ·   mejoraron {n_imp}/{n}\n"
        f"Final = {best_c[-1]:.2f}  (gap {best_c[-1] - OPT:.2f})")
ax.text(0.015, 0.05, info, transform=ax.transAxes, fontsize=8.0, va="bottom",
        ha="left", bbox=dict(boxstyle="round,pad=0.5", fc="#f4f6f8", ec="#b8bdc4"))

fig.savefig(BASE / "grafico_F12_convergencia_estancamiento.png", dpi=160, bbox_inches="tight")
print("PNG:", BASE / "grafico_F12_convergencia_estancamiento.png")
