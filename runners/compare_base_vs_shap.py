"""Runner pareado: WO_base vs WO_shap con misma semilla por corrida.

Permite Wilcoxon signed-rank emparejado en lugar de Mann-Whitney. Reusa los
modulos de runners/run_wo_base.py y runners/run_wo_shap.py compartiendo el
adaptador del problema.

Uso:

    python -m runners.compare_base_vs_shap \\
        --problem tmlap:1.instancia_simple.txt \\
        --runs 30 --profile soft --init-mode random \\
        --output experiments/compare_tmlap_simple_30runs

    python -m runners.compare_base_vs_shap \\
        --problem cec2022:F6 --dim 10 \\
        --runs 30 --profile soft \\
        --output experiments/compare_cec_F6_30runs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from problems.cec2022 import FUNCTION_IDS as CEC_FUNCTION_IDS
from problems.factory import parse_problem_spec
from shap_controller import PROFILE_DEFAULTS

from runners.run_wo_base import run_one as run_one_base
from runners.run_wo_shap import run_one as run_one_shap


def parse_args():
    parser = argparse.ArgumentParser(description="WO_base vs WO_shap pareado")
    parser.add_argument("--problem", required=True)
    parser.add_argument("--dim", type=int, default=10)
    parser.add_argument("--clients", type=int, default=None)
    parser.add_argument("--hubs", type=int, default=None)
    parser.add_argument("--agents", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--profile", choices=sorted(PROFILE_DEFAULTS), default="soft")
    parser.add_argument(
        "--init-mode", choices=["local_search", "random"], default="local_search"
    )
    parser.add_argument("--shapley-steps", type=int, default=3)
    parser.add_argument("--acceptance-mode", choices=["diversity", "strict"], default="diversity")
    parser.add_argument("--rescue-scale", type=float, default=1.0)
    parser.add_argument("--neutral-cooldown-multiplier", type=float, default=1.5)
    parser.add_argument("--rejected-cooldown-multiplier", type=float, default=2.5)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _make_problems(args):
    if args.problem.startswith("cec2022:"):
        target = args.problem.split(":", 1)[1].strip()
        if target.lower() == "all":
            return [
                (parse_problem_spec(f"cec2022:F{fid}", dim=args.dim), f"F{fid}")
                for fid in CEC_FUNCTION_IDS
            ]
        problem = parse_problem_spec(args.problem, dim=args.dim)
        return [(problem, f"F{problem.function_id}")]
    if args.problem.startswith("tmlap:"):
        problem = parse_problem_spec(args.problem, clients=args.clients, hubs=args.hubs)
        return [(problem, problem.name)]
    raise ValueError(f"Familia desconocida: {args.problem!r}")


def _build_stats(base_rows, shap_rows):
    """Stats por algoritmo (best/worst/mean/std)."""
    rows = []
    for tag, source in (("WO_base", base_rows), ("WO_shap", shap_rows)):
        df = pd.DataFrame(source)
        for problem_label, data in df.groupby("problem", sort=True):
            rows.append(
                {
                    "algorithm": tag,
                    "problem": problem_label,
                    "runs": int(data["run_id"].nunique()),
                    "fitness_best": float(data["final_fitness"].min()),
                    "fitness_worst": float(data["final_fitness"].max()),
                    "fitness_mean": float(data["final_fitness"].mean()),
                    "fitness_median": float(data["final_fitness"].median()),
                    "fitness_std": float(data["final_fitness"].std(ddof=1)),
                    "time_mean_seconds": float(data["elapsed_seconds"].mean()),
                    "interventions_mean": float(data["interventions"].mean()) if "interventions" in data.columns else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _build_paired(base_rows, shap_rows):
    """Une fila por fila (mismo run_id y problema) y calcula deltas."""
    df_base = pd.DataFrame(base_rows).add_suffix("_base")
    df_shap = pd.DataFrame(shap_rows).add_suffix("_shap")
    df = df_base.merge(
        df_shap,
        left_on=["run_id_base", "problem_base"],
        right_on=["run_id_shap", "problem_shap"],
        how="inner",
    )
    df["delta_base_minus_shap"] = (
        df["final_fitness_base"] - df["final_fitness_shap"]
    )
    df["winner"] = np.where(
        df["delta_base_minus_shap"] > 0, "shap",
        np.where(df["delta_base_minus_shap"] < 0, "base", "tie"),
    )
    return df


def _build_tests(paired_df):
    """Wilcoxon emparejado y sign test sobre delta = base - shap."""
    results = []
    for problem_label, data in paired_df.groupby("problem_base", sort=True):
        delta = data["delta_base_minus_shap"].astype(float).to_numpy()
        nonzero = delta[np.abs(delta) > 1e-12]
        result = {"problem": problem_label, "n_pairs": int(len(delta)), "n_nonzero": int(len(nonzero))}
        if len(nonzero) >= 2:
            try:
                stat, p = stats.wilcoxon(nonzero)
                result["wilcoxon_statistic"] = float(stat)
                result["wilcoxon_p"] = float(p)
            except Exception as exc:
                result["wilcoxon_error"] = str(exc)
        else:
            result["wilcoxon_p"] = np.nan
            result["wilcoxon_note"] = "insufficient_nonzero_pairs"
        n_pos = int(np.sum(delta > 0))
        n_neg = int(np.sum(delta < 0))
        result["sign_positive"] = n_pos
        result["sign_negative"] = n_neg
        if n_pos + n_neg > 0:
            try:
                result["sign_p"] = float(
                    stats.binomtest(min(n_pos, n_neg), n=n_pos + n_neg, p=0.5).pvalue
                )
            except Exception:
                result["sign_p"] = np.nan
        else:
            result["sign_p"] = np.nan
        results.append(result)
    return pd.DataFrame(results)


def main():
    args = parse_args()
    problems = _make_problems(args)
    output_dir = Path(args.output)
    values_dir = output_dir / "values"
    curves_dir = output_dir / "curves"
    values_dir.mkdir(parents=True, exist_ok=True)
    curves_dir.mkdir(parents=True, exist_ok=True)

    base_rows = []
    shap_rows = []
    all_events = []
    all_shap_values = []

    for problem_idx, (problem, label) in enumerate(problems):
        print(f"\n== {label} ({getattr(problem, 'family', 'unknown')}) ==", flush=True)
        for run_id in range(1, args.runs + 1):
            base_result = run_one_base(problem, args, run_id, problem_idx, label)
            shap_result = run_one_shap(problem, args, run_id, problem_idx, label)

            base_rows.append(base_result["row"])
            shap_rows.append(shap_result["row"])

            np.savetxt(
                curves_dir / f"conv_curve_{label}_run{run_id}_base.csv",
                base_result["curve"], delimiter=",", header="best_fitness", comments="",
            )
            np.savetxt(
                curves_dir / f"conv_curve_{label}_run{run_id}_shap.csv",
                shap_result["curve"], delimiter=",", header="best_fitness", comments="",
            )
            if not shap_result["events"].empty:
                ev = shap_result["events"].copy()
                ev.insert(0, "problem", label)
                ev.insert(0, "run_id", run_id)
                all_events.append(ev)
            if not shap_result["shap_rows"].empty:
                sr = shap_result["shap_rows"].copy()
                sr.insert(0, "problem", label)
                sr.insert(0, "run_id", run_id)
                all_shap_values.append(sr)

            delta = base_result["row"]["final_fitness"] - shap_result["row"]["final_fitness"]
            print(
                f"  run={run_id}/{args.runs} "
                f"base={base_result['row']['final_fitness']:.6g} "
                f"shap={shap_result['row']['final_fitness']:.6g} "
                f"delta_base_minus_shap={delta:.6g} "
                f"interventions={shap_result['row']['interventions']}",
                flush=True,
            )

    base_df = pd.DataFrame(base_rows)
    shap_df = pd.DataFrame(shap_rows)
    base_df.to_csv(values_dir / "summary_base.csv", index=False)
    shap_df.to_csv(values_dir / "summary_shap.csv", index=False)

    stats_df = _build_stats(base_rows, shap_rows)
    stats_df.to_csv(values_dir / "statistics.csv", index=False)

    paired = _build_paired(base_rows, shap_rows)
    paired.to_csv(values_dir / "paired.csv", index=False)

    tests = _build_tests(paired)
    tests.to_csv(values_dir / "tests.csv", index=False)

    if all_events:
        pd.concat(all_events, ignore_index=True).to_csv(
            values_dir / "controller_events.csv", index=False
        )
    if all_shap_values:
        pd.concat(all_shap_values, ignore_index=True).to_csv(
            values_dir / "shap_values.csv", index=False
        )

    print(f"\nSalidas en: {values_dir}")
    print("  - summary_base.csv, summary_shap.csv")
    print("  - statistics.csv")
    print("  - paired.csv (1 fila por par base/shap)")
    print("  - tests.csv (Wilcoxon emparejado + sign test)")
    if all_events:
        print("  - controller_events.csv (eventos del controlador SHAP)")
    if all_shap_values:
        print("  - shap_values.csv (atribuciones por evento)")


if __name__ == "__main__":
    main()
