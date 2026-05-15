"""Efectividad por intervencion del controlador SHAP-online sobre TMLAP.

Lee `controller_events.csv` + `curves/convergence_runN.csv` de cada carpeta de
salida y clasifica cada intervencion en {improved, neutral, worsened} usando
dos criterios:

  - inmediato: ``delta_best = after_best - before_best``
  - post-10:   ``curve[t+10] - curve[t]`` (alineado con la ventana fija de 10
                iteraciones que usa el controlador on-line para evaluar el
                impacto de un rescate).

Usa la misma tolerancia ``improvement_threshold`` del controlador (rel 1e-6,
abs 1e-10) para clasificar.

Salida: un CSV por carpeta + un consolidado global y una tabla en consola.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
INSTANCE_FOLDERS = {
    "simple": PROJECT_ROOT / "run5_shap_sim_simple",
    "mediana": PROJECT_ROOT / "run5_shap_sim_mediana",
    "dura": PROJECT_ROOT / "run5_shap_sim_dura",
}
POST_WINDOW = 10


def improvement_threshold(fitness: float, abs_eps: float = 1e-10, rel_eps: float = 1e-6) -> float:
    return max(abs_eps, rel_eps * max(abs(float(fitness)), 1.0))


def classify(delta: float, reference: float) -> str:
    threshold = improvement_threshold(reference)
    if delta < -threshold:
        return "improved"
    if delta > threshold:
        return "worsened"
    return "neutral"


def load_curve(curves_dir: Path, run_id: int) -> np.ndarray:
    path = curves_dir / f"convergence_run{run_id}.csv"
    df = pd.read_csv(path)
    return df["best_fitness"].to_numpy(dtype=float)


def analyse_instance(label: str, folder: Path) -> pd.DataFrame:
    events_path = folder / "values" / "controller_events.csv"
    curves_dir = folder / "curves"
    if not events_path.exists():
        raise FileNotFoundError(f"No existe {events_path}")
    if not curves_dir.exists():
        raise FileNotFoundError(f"No existe {curves_dir}")

    events = pd.read_csv(events_path)
    rows = []
    for run_id, run_events in events.groupby("run_id", sort=True):
        curve = load_curve(curves_dir, int(run_id))
        max_iter = len(curve)
        for _, event in run_events.iterrows():
            iteration = int(event["iteration"])
            before = float(event["before_best"])
            after = float(event["after_best"])
            immediate_status = classify(after - before, before)
            post_index = min(iteration + POST_WINDOW, max_iter - 1)
            post_best = float(curve[post_index])
            post_delta = post_best - before
            post_status = classify(post_delta, before)
            rows.append(
                {
                    "instance": label,
                    "run_id": int(run_id),
                    "event_id": int(event["event_id"]),
                    "iteration": iteration,
                    "action": event.get("action", ""),
                    "fraction": float(event.get("fraction", np.nan)),
                    "shap_dominant_feature": event.get("shap_dominant_feature", ""),
                    "n_selected_agents": int(event.get("n_selected_agents", 0)),
                    "improved_agents": int(event.get("improved_agents", 0))
                    if not pd.isna(event.get("improved_agents", np.nan))
                    else np.nan,
                    "before_best": before,
                    "after_best": after,
                    "immediate_delta": after - before,
                    "immediate_status": immediate_status,
                    "post_iteration": post_index,
                    "post_best": post_best,
                    "post_delta": post_delta,
                    "post_status": post_status,
                }
            )
    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for instance, sub in df.groupby("instance", sort=False):
        n_events = len(sub)
        if n_events == 0:
            continue
        immediate_counts = sub["immediate_status"].value_counts().to_dict()
        post_counts = sub["post_status"].value_counts().to_dict()
        rows.append(
            {
                "instance": instance,
                "n_events": n_events,
                "immediate_improved_rate": immediate_counts.get("improved", 0) / n_events,
                "immediate_neutral_rate": immediate_counts.get("neutral", 0) / n_events,
                "immediate_worsened_rate": immediate_counts.get("worsened", 0) / n_events,
                "post10_improved_rate": post_counts.get("improved", 0) / n_events,
                "post10_neutral_rate": post_counts.get("neutral", 0) / n_events,
                "post10_worsened_rate": post_counts.get("worsened", 0) / n_events,
                "mean_immediate_delta": float(sub["immediate_delta"].mean()),
                "mean_post_delta": float(sub["post_delta"].mean()),
                "median_post_delta": float(sub["post_delta"].median()),
                "best_post_delta": float(sub["post_delta"].min()),
                "mean_improved_agents": float(
                    sub["improved_agents"].dropna().mean()
                )
                if sub["improved_agents"].notna().any()
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def by_action(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = df.groupby(["instance", "action"], sort=False, dropna=False)
    for (instance, action), sub in grouped:
        n_events = len(sub)
        post_counts = sub["post_status"].value_counts().to_dict()
        rows.append(
            {
                "instance": instance,
                "action": action,
                "n_events": n_events,
                "post10_improved_rate": post_counts.get("improved", 0) / n_events,
                "post10_neutral_rate": post_counts.get("neutral", 0) / n_events,
                "post10_worsened_rate": post_counts.get("worsened", 0) / n_events,
                "mean_post_delta": float(sub["post_delta"].mean()),
            }
        )
    return pd.DataFrame(rows)


def by_dominant_feature(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = df.groupby(["instance", "shap_dominant_feature"], sort=False, dropna=False)
    for (instance, feature), sub in grouped:
        n_events = len(sub)
        post_counts = sub["post_status"].value_counts().to_dict()
        rows.append(
            {
                "instance": instance,
                "shap_dominant_feature": feature,
                "n_events": n_events,
                "post10_improved_rate": post_counts.get("improved", 0) / n_events,
                "mean_post_delta": float(sub["post_delta"].mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    per_event_frames = []
    for label, folder in INSTANCE_FOLDERS.items():
        if not folder.exists():
            print(f"[aviso] omito {label}: {folder} no existe")
            continue
        df = analyse_instance(label, folder)
        per_event_frames.append(df)
        out_path = folder / "values" / "effectiveness_per_event.csv"
        df.to_csv(out_path, index=False)
        print(f"[ok] {label}: {len(df)} eventos -> {out_path}")

    if not per_event_frames:
        print("Sin datos a procesar.")
        return

    full = pd.concat(per_event_frames, ignore_index=True)
    agg = aggregate(full)
    by_act = by_action(full)
    by_feat = by_dominant_feature(full)

    consolidated_path = PROJECT_ROOT / "tmlap_effectiveness_summary.csv"
    by_action_path = PROJECT_ROOT / "tmlap_effectiveness_by_action.csv"
    by_feature_path = PROJECT_ROOT / "tmlap_effectiveness_by_feature.csv"
    agg.to_csv(consolidated_path, index=False)
    by_act.to_csv(by_action_path, index=False)
    by_feat.to_csv(by_feature_path, index=False)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")

    print("\n=== Efectividad consolidada por instancia ===")
    print(agg.to_string(index=False))
    print("\n=== Efectividad por accion (post-10) ===")
    print(by_act.to_string(index=False))
    print("\n=== Efectividad por feature SHAP dominante (post-10) ===")
    print(by_feat.to_string(index=False))

    print(f"\nArchivos generados:")
    print(f"  {consolidated_path}")
    print(f"  {by_action_path}")
    print(f"  {by_feature_path}")


if __name__ == "__main__":
    main()
