"""Genera 2 flujogramas (matplotlib) del WO + controlador SHAP por agente:

1. DISEÑO: flujo conceptual/metodologico (lo que hace el algoritmo).
2. IMPLEMENTACION: modulos y funciones (como esta codificado).

Salida: dos PNG + un PDF combinado (2 paginas) en la carpeta de salida.

Uso:
    python -m analysis.make_diagrams --output "C:/ruta/de/salida"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon  # noqa: E402

C_PROC = ("#dbeafe", "#1f77b4")   # proceso (azul)
C_DEC = ("#fde8c3", "#cf8a04")    # decision (ambar)
C_END = ("#d6f0dd", "#2e7d32")    # inicio/fin (verde)
C_A = ("#ffe0cc", "#d4631a")      # rama A random (naranjo)
C_B = ("#e7dcf6", "#7c4fc0")      # rama B guiada (violeta)
C_OUT = ("#f0f0f0", "#777777")    # salidas/datos (gris)


def box(ax, cx, cy, w, h, title, sub=None, color=C_PROC):
    fc, ec = color
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.2,rounding_size=1.2",
        linewidth=1.3, edgecolor=ec, facecolor=fc, zorder=3))
    if sub:
        ax.text(cx, cy + h * 0.17, title, ha="center", va="center",
                fontsize=8.2, weight="bold", color="#15233b", zorder=4)
        ax.text(cx, cy - h * 0.27, sub, ha="center", va="center",
                fontsize=6.6, family="monospace", color="#4a4a4a", zorder=4)
    else:
        ax.text(cx, cy, title, ha="center", va="center",
                fontsize=8.2, weight="bold", color="#15233b", zorder=4)


def diamond(ax, cx, cy, w, h, text, color=C_DEC):
    fc, ec = color
    pts = [(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)]
    ax.add_patch(Polygon(pts, closed=True, linewidth=1.3, edgecolor=ec,
                         facecolor=fc, zorder=3))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=7.3,
            weight="bold", color="#5a3d00", zorder=4)


def arr(ax, p1, p2, label=None, rad=0.0, color="#333", dx=1.6):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=12,
                 linewidth=1.1, color=color, zorder=2,
                 connectionstyle=f"arc3,rad={rad}"))
    if label:
        ax.text((p1[0] + p2[0]) / 2 + dx, (p1[1] + p2[1]) / 2, label,
                fontsize=7, color=color, ha="left", va="center", weight="bold", zorder=4)


def route(ax, points, label=None, color="#333"):
    for a, b in zip(points[:-1], points[1:]):
        last = b is points[-1]
        if last:
            arr(ax, a, b, color=color)
        else:
            ax.plot([a[0], b[0]], [a[1], b[1]], color=color, linewidth=1.1, zorder=2)
    if label:
        ax.text(points[0][0] + 1.5, points[0][1] - 2, label, fontsize=7,
                color=color, ha="left", va="top", weight="bold", zorder=4)


def _frame(ax, title):
    ax.set_xlim(0, 100); ax.set_ylim(0, 134); ax.axis("off")
    ax.text(50, 132, title, ha="center", va="top", fontsize=13, weight="bold",
            color="#14213d")


def design_diagram(ax):
    _frame(ax, "Diseño — WO + controlador SHAP por agente (régimen MaxFES)")
    box(ax, 50, 124, 50, 8, "Inicializar población",
        "uniforme CEC / repair+LS TMLAP  (cuenta FES init)", C_END)
    diamond(ax, 50, 110, 30, 11, "¿FES < MaxFES?", C_DEC)
    box(ax, 16, 110, 24, 9, "FIN", "reportar best@MaxFES y gap", C_END)
    box(ax, 50, 96, 56, 9, "Evaluar población  (FES_búsqueda)",
        "actualiza GBest y pbest/last_improve por agente", C_PROC)
    box(ax, 50, 83, 56, 8, "Señales WO por FES",
        "alpha=1-FES/MaxFES, beta, danger, safety", C_PROC)
    diamond(ax, 50, 68, 40, 13, "¿Agente estancado?\nFES sin mejora >= 10% MaxFES\n(guard 5% / late 95%)", C_DEC)
    box(ax, 50, 53, 50, 8, "SHAP exacto del agente",
        "64 coaliciones sobre las 6 señales  (FES_shap)", C_PROC)
    diamond(ax, 50, 39, 38, 12, "¿feature dominante\n>= 90% de contribución?", C_DEC)
    box(ax, 24, 24, 40, 10, "Rama A: reinit ALEATORIO",
        "uniforme  lb+(ub-lb)*rand", C_A)
    box(ax, 76, 24, 40, 10, "Rama B: reinit GUIADO",
        "paso WO, señal dominante amplificada", C_B)
    box(ax, 50, 12, 56, 9, "Aplicar reinit (SIEMPRE)",
        "GBest preservado · reset reloj del agente", C_PROC)
    box(ax, 50, 2.5, 50, 7, "Movimiento WO poblacional (4 regímenes)", color=C_PROC)

    arr(ax, (50, 120), (50, 115.5))                       # init -> decision FES
    arr(ax, (35, 110), (28, 110), "No")                   # decision -> FIN
    arr(ax, (50, 104.5), (50, 100.5), "Sí")               # decision -> evaluar
    arr(ax, (50, 91.5), (50, 87))                          # evaluar -> señales
    arr(ax, (50, 79), (50, 74.5))                          # señales -> estancado?
    arr(ax, (50, 61.5), (50, 57), "Sí")                    # estancado sí -> SHAP
    arr(ax, (50, 49), (50, 45))                            # SHAP -> contribución?
    arr(ax, (40, 36), (30, 29), "No")                      # -> Rama A
    arr(ax, (60, 36), (70, 29), "Sí")                      # -> Rama B
    arr(ax, (24, 19), (44, 16.5))                          # A -> aplicar
    arr(ax, (76, 19), (56, 16.5))                          # B -> aplicar
    arr(ax, (50, 7.5), (50, 6))                            # aplicar -> movimiento
    # rama "No estancado": salta a movimiento por la izquierda
    route(ax, [(30, 68), (8, 68), (8, 2.5), (25, 2.5)], "No", color="#777")
    # loop de iteracion: movimiento -> decision FES (por la derecha)
    route(ax, [(75, 2.5), (95, 2.5), (95, 110), (65, 110)], color="#1f77b4")


def impl_diagram(ax):
    _frame(ax, "Implementación — módulos y funciones (run_wo_shap.run_one)")
    box(ax, 50, 124, 64, 9, "FESBudget(max_fes) + counting_objective",
        "wo_core/fes.py  ->  eval_search / eval_shap / eval_intervention", C_OUT)
    box(ax, 50, 112, 64, 8, "problem.initial_population(...)",
        "problems/*  (TMLAP: on_eval -> budget.init)", C_PROC)
    diamond(ax, 50, 99, 30, 10, "while not\nbudget.exhausted()", C_DEC)
    box(ax, 16, 99, 24, 8, "salidas CSV", "summary / events / shap_values", C_OUT)
    box(ax, 50, 86, 66, 9, "evaluate_and_update_leaders(eval_search, budget)",
        "wo_core/walrus.py  (corte exacto en MaxFES)", C_PROC)
    box(ax, 50, 75, 66, 8, "update pbest[i] / last_improve_fes[i]  +  signals",
        "iteration_signals(fes_start, max_fes)  ·  walrus.py", C_PROC)
    diamond(ax, 50, 61, 40, 12, "should_consider_intervention\n(state, budget)\ncontroller.py — gates en FES", C_DEC)
    box(ax, 50, 47, 64, 9, "explain_fitness(state, value_fn)",
        "controller + agent_sim.make_value_function_for_agent (64 coal.)", C_PROC)
    diamond(ax, 50, 34, 40, 11, "decide(state, shap_info)\nshare dominante >= 0.90 ?", C_DEC)
    box(ax, 24, 20, 40, 9, "dispatch_rescue_single",
        "reinit_random  (actions.py)", C_A)
    box(ax, 76, 20, 40, 9, "dispatch_rescue_single",
        "reinit_guided  (actions.py)", C_B)
    box(ax, 50, 8.5, 66, 8, "eval_intervention + aplicar + register_decision",
        "reinit SIEMPRE · reset reloj · apply_wo_movement (walrus)", C_PROC)

    arr(ax, (50, 119.5), (50, 116))
    arr(ax, (50, 108), (50, 104))
    arr(ax, (35, 99), (28, 99), "no")
    arr(ax, (50, 94), (50, 90.5), "sí")
    arr(ax, (50, 81.5), (50, 79))
    arr(ax, (50, 71), (50, 67))
    arr(ax, (50, 55), (50, 51.5), "candidato")
    arr(ax, (50, 42.5), (50, 39.5))
    arr(ax, (40, 31), (30, 24.5), "No")
    arr(ax, (60, 31), (70, 24.5), "Sí")
    arr(ax, (24, 15.5), (44, 12.5))
    arr(ax, (76, 15.5), (56, 12.5))
    # gate "no interviene" -> directo a movimiento (apply_wo_movement) por la izq.
    route(ax, [(30, 61), (8, 61), (8, 8.5), (17, 8.5)], "no", color="#777")
    # loop de iteracion por la derecha
    route(ax, [(83, 8.5), (95, 8.5), (95, 99), (65, 99)], color="#1f77b4")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)

    specs = [("diseno", design_diagram, "Diagrama_diseno_WO_SHAP.png"),
             ("impl", impl_diagram, "Diagrama_implementacion_WO_SHAP.png")]
    pngs = []
    for _, fn, fname in specs:
        fig, ax = plt.subplots(figsize=(8.6, 11.7))
        fn(ax)
        fig.tight_layout()
        p = out / fname
        fig.savefig(p, dpi=170, bbox_inches="tight")
        plt.close(fig); pngs.append(p)
        print("PNG:", p)

    pdf_path = out / "Diagramas_flujo_WO_SHAP.pdf"
    with PdfPages(pdf_path) as pdf:
        for _, fn, _f in specs:
            fig, ax = plt.subplots(figsize=(8.6, 11.7))
            fn(ax); fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
    print("PDF:", pdf_path)


if __name__ == "__main__":
    main()
