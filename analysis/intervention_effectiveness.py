"""Analisis de efectividad por intervencion del controlador SHAP sobre TMLAP.

Lee los CSVs ``controller_events.csv`` y las curvas de convergencia generadas
por ``WO_tmlap_shap/run_wo_tmlap_online_shap.py`` y clasifica cada intervencion
en cuatro categorias siguiendo la idea de la Seccion 4.13 del informe:

* ``immediate_improved``: el rescate hace bajar el best en la misma iteracion
  (``delta_best < -threshold``). En la generacion actual de CSVs solo este
  campo esta disponible directamente desde ``controller_events.csv``.
* ``post_improved``: tras una ventana de ``--post-window`` iteraciones el
  best esta por debajo del pre-fitness por al menos el umbral relativo.
* ``post_neutral``: ni mejora ni empeora dentro del umbral.
* ``post_worsened``: la curva subio tras la intervencion (no deberia ocurrir
  porque el WO monotoniza el best, pero queda registrado por consistencia).

El umbral usa la misma regla que el resto del proyecto:
``threshold = max(1e-10, 1e-6 * max(|pre|, 1))``.

Uso::

    python tmlap_intervention_effectiveness.py \
        --root . --instances simple mediana dura \
        --post-window 10 \
        --output tmlap_effectiveness_summary.csv
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def improvement_threshold(value, abs_eps=1e-10, rel_eps=1e-6):
    return max(abs_eps, rel_eps * max(abs(float(value)), 1.0))


def classify(pre_value, post_value):
    delta = float(pre_value) - float(post_value)
    threshold = improvement_threshold(pre_value)
    if delta > threshold:
        return "improved", delta
    if delta < -threshold:
        return "worsened", delta
    return "neutral", delta


def load_curve(curve_path):
    df = pd.read_csv(curve_path)
    if "best_fitness" not in df.columns:
        raise ValueError(f"{curve_path} no tiene columna best_fitness")
    return df["best_fitness"].astype(float).to_numpy()


def analyse_instance(values_dir, curves_dir, post_window):
    events_path = Path(values_dir) / "controller_events.csv"
    if not events_path.exists():
        return None
    events = pd.read_csv(events_path)
    if events.empty:
        return None

    enriched_rows = []
    cache_curves = {}
    for _, row in events.iterrows():
        run_id = int(row["run_id"])
        iteration = int(row["iteration"])
        pre = float(row["before_best"])
        immediate = float(row["after_best"])
        imm_status, imm_delta = classify(pre, immediate)

        curve_path = Path(curves_dir) / f"convergence_run{run_id}.csv"
        if run_id not in cache_curves:
            cache_curves[run_id] = load_curve(curve_path)
        curve = cache_curves[run_id]
        post_iter = min(iteration + post_window, len(curve) - 1)
        post_value = float(curve[post_iter])
        post_status, post_delta = classify(pre, post_value)

        enriched_rows.append(
            {
                "run_id": run_id,
                "iteration": iteration,
                "action": row.get("action", ""),
                "fraction": float(row.get("fraction", np.nan)),
                "selected_agents": int(row.get("n_selected_agents", 0)),
                "dominant_feature": row.get("shap_dominant_feature", ""),
                "pre_best": pre,
                "immediate_best": immediate,
                "post_best": post_value,
                "post_iter_idx": post_iter,
                "immediate_status": imm_status,
                "post_status": post_status,
                "immediate_delta": imm_delta,
                "post_delta": post_delta,
            }
        )
    return pd.DataFrame(enriched_rows)


def aggregate(events_df, instance_name):
    n = len(events_df)
    if n == 0:
        return None
    rates_post = events_df["post_status"].value_counts(normalize=True).to_dict()
    rates_imm = events_df["immediate_status"].value_counts(normalize=True).to_dict()
    action_breakdown = (
        events_df.groupby(["action", "post_status"]).size().unstack(fill_value=0)
    )
    feature_breakdown = (
        events_df.groupby(["dominant_feature", "post_status"]).size().unstack(fill_value=0)
    )

    return {
        "instance": instance_name,
        "n_events": int(n),
        "n_runs": int(events_df["run_id"].nunique()),
        "events_per_run": float(n / max(events_df["run_id"].nunique(), 1)),
        "immediate_improved_rate": float(rates_imm.get("improved", 0.0)),
        "immediate_neutral_rate": float(rates_imm.get("neutral", 0.0)),
        "immediate_worsened_rate": float(rates_imm.get("worsened", 0.0)),
        "post_improved_rate": float(rates_post.get("improved", 0.0)),
        "post_neutral_rate": float(rates_post.get("neutral", 0.0)),
        "post_worsened_rate": float(rates_post.get("worsened", 0.0)),
        "median_post_delta_when_improved": float(
            events_df.loc[events_df["post_status"] == "improved", "post_delta"].median()
            if (events_df["post_status"] == "improved").any()
            else np.nan
        ),
        "action_breakdown": action_breakdown,
        "feature_breakdown": feature_breakdown,
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directorio raiz que contiene las carpetas run5_shap_sim_*",
    )
    p.add_argument(
        "--instances",
        nargs="+",
        default=["simple", "mediana", "dura"],
        help="Lista de sufijos de instancia a analizar.",
    )
    p.add_argument(
        "--prefix",
        type=str,
        default="run5_shap_sim_",
        help="Prefijo comun de las carpetas a analizar.",
    )
    p.add_argument(
        "--post-window",
        type=int,
        default=10,
        help="Ventana en iteraciones para evaluar el outcome (informe Sec. 4.13).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("tmlap_effectiveness_summary.csv"),
        help="Archivo CSV de salida con el resumen agregado.",
    )
    p.add_argument(
        "--detail-output",
        type=Path,
        default=None,
        help="Si se entrega, escribe la tabla por evento con clasificacion incluida.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    detail_frames = []
    rows = []
    breakdowns = []
    for instance in args.instances:
        folder = args.root / f"{args.prefix}{instance}"
        values_dir = folder / "values"
        curves_dir = folder / "curves"
        if not values_dir.exists():
            print(f"[skip] no existe {values_dir}")
            continue
        df = analyse_instance(values_dir, curves_dir, args.post_window)
        if df is None or df.empty:
            print(f"[skip] {instance}: sin eventos registrados")
            continue
        df["instance"] = instance
        detail_frames.append(df)
        agg = aggregate(df, instance)
        breakdowns.append(agg)
        rows.append(
            {
                "instance": agg["instance"],
                "n_events": agg["n_events"],
                "n_runs": agg["n_runs"],
                "events_per_run": agg["events_per_run"],
                "immediate_improved_rate": agg["immediate_improved_rate"],
                "immediate_neutral_rate": agg["immediate_neutral_rate"],
                "immediate_worsened_rate": agg["immediate_worsened_rate"],
                "post_improved_rate": agg["post_improved_rate"],
                "post_neutral_rate": agg["post_neutral_rate"],
                "post_worsened_rate": agg["post_worsened_rate"],
                "median_post_delta_when_improved": agg["median_post_delta_when_improved"],
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(args.output, index=False)
    print("\n=== Resumen por instancia ===")
    if not summary.empty:
        pd.set_option("display.float_format", "{:.3f}".format)
        print(summary.to_string(index=False))
    print(f"\nResumen guardado en {args.output}")

    if args.detail_output and detail_frames:
        detail = pd.concat(detail_frames, ignore_index=True)
        detail.to_csv(args.detail_output, index=False)
        print(f"Detalle por evento guardado en {args.detail_output}")

    print("\n=== Desglose por accion x outcome (ventana post) ===")
    for agg in breakdowns:
        print(f"\n[{agg['instance']}]")
        if agg["action_breakdown"].empty:
            print("  (sin acciones registradas)")
            continue
        print(agg["action_breakdown"].to_string())

    print("\n=== Desglose por feature dominante x outcome (ventana post) ===")
    for agg in breakdowns:
        print(f"\n[{agg['instance']}]")
        if agg["feature_breakdown"].empty:
            print("  (sin features dominantes registrados)")
            continue
        print(agg["feature_breakdown"].to_string())


if __name__ == "__main__":
    main()
