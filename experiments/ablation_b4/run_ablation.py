"""Ablation B4 — ¿aporta SHAP al REINICIO de los agentes estancados?

Compara 3 brazos con las MISMAS semillas (comparación pareada), bajo el mismo
MaxFES y el mismo flujo WO/FES:

  * base : WO puro, sin controlador (referencia).
  * blind: detecta estancamiento (FES, por agente) + reinit ciego  (w = 1).
  * shap : detecta estancamiento + SHAP por agente + reinit MODULADO (w = dominant_share).

La acción de reinicio es una MUTACIÓN modulada (A1 + B4):

    x_nuevo = (1 - w) * x_actual + w * (lb + (ub - lb) * rand)

donde w = cuota de contribución de la señal dominante (|SHAP_dom| / Σ|SHAP|).
En 'blind' w=1 (reinit uniforme total, sin gastar FES en SHAP).

Soporta una función (``cec2022:F12``) o todas (``cec2022:all``). Registra la
distribución de ``w`` (modo shap) para verificar si degenera a reinit ciego.

Uso:
    python experiments/ablation_b4/run_ablation.py --problem cec2022:all \\
        --dim 10 --agents 30 --max-fes 50000 --runs 30 \\
        --modes base,blind,shap --output experiments/ablation_b4_allcec
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from problems.cec2022 import FUNCTION_IDS as CEC_FUNCTION_IDS
from problems.factory import parse_problem_spec
from shap_controller import SHAPFitnessController, improvement_threshold
from shap_controller.features import FEATURE_BASELINE_DEFAULTS, FEATURE_COLUMNS
from wo_core.agent_sim import make_value_function_for_agent
from wo_core.fes import FESBudget, counting_objective
from wo_core.walrus import (
    apply_wo_movement,
    apply_wo_movement_single,
    evaluate_and_update_leaders,
    iteration_signals,
    walrus_role_counts,
)


def parse_args():
    ap = argparse.ArgumentParser(description="Ablation B4: aporte de SHAP al reinicio.")
    ap.add_argument("--problem", required=True, help="cec2022:F12  o  cec2022:all")
    ap.add_argument("--dim", type=int, default=10)
    ap.add_argument("--agents", type=int, default=30)
    ap.add_argument("--max-fes", type=int, default=50000)
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--modes", type=str, default="base,elite",
                    help="Configuraciones a correr: 'base' (WO solo) y 'elite' (WO + controlador).")
    ap.add_argument("--init-mode", choices=["local_search", "random"], default="random",
                    help="TMLAP: 'random' (default) inicia con repair SIN local_search -> el WO "
                         "trabaja desde 0 (sin que el inicializador optimice). 'local_search' lo "
                         "deja optimizar en el init (NO recomendado para analizar el WO).")
    ap.add_argument("--output", required=True)
    return ap.parse_args()


_TMLAP_INSTANCES = ["1.instancia_simple.txt", "2.instancia_mediana.txt", "3.instancia_dura.txt"]


def make_problems(args):
    spec = args.problem.strip()
    if spec.lower() == "cec2022:all":
        return [(parse_problem_spec(f"cec2022:F{fid}", dim=args.dim), f"F{fid}")
                for fid in CEC_FUNCTION_IDS]
    if spec.lower() == "tmlap:all":
        out = []
        for fname in _TMLAP_INSTANCES:
            p = parse_problem_spec(f"tmlap:{fname}", clients=None, hubs=None)
            out.append((p, getattr(p, "name", fname)))
        return out
    if spec.startswith("tmlap:"):
        p = parse_problem_spec(spec, clients=None, hubs=None)
        return [(p, getattr(p, "name", "tmlap"))]
    problem = parse_problem_spec(spec, dim=args.dim)
    return [(problem, f"F{getattr(problem, 'function_id', '?')}")]


def _gbest_profile(eval_shap, controller, positions, fitness_values, lb, ub, dim,
                   role_counts, best_pos, second_pos, best_score, signal_state,
                   max_fes, fes_start, rng):
    """SHAP del GBest -> perfil de cuotas {senal: |SHAP|/Sum|SHAP|}. Gasta 1 explicacion."""
    gi = int(np.argmin(fitness_values))            # slot temporal para insertar el GBest
    frozen = positions.copy()
    frozen[gi, :] = np.asarray(best_pos, dtype=float)
    vf = make_value_function_for_agent(
        eval_shap, gi, frozen, lb, ub, dim, role_counts,
        best_pos, second_pos, signal_state, max_fes, fes_start,
        controller.shapley_steps, rng=rng,
    )
    info = controller.explain_fitness(
        {**signal_state, "agent_index": gi, "agent_fitness": float(best_score),
         "fes_since_improve": 0, "fes": 0},
        value_function=vf,
    )
    vals = (info or {}).get("values", {})
    ab = {f: abs(float(vals.get(f, 0.0))) for f in FEATURE_COLUMNS}
    tot = sum(ab.values())
    prof = ({f: 1.0 / len(FEATURE_COLUMNS) for f in FEATURE_COLUMNS} if tot <= 0.0
            else {f: ab[f] / tot for f in FEATURE_COLUMNS})
    return prof, info


def _reposition_elite(positions, i, profile, signal_state, lb, ub, dim, n_agents,
                      role_counts, best_pos, second_pos, rng, amp):
    """Reposiciona al agente i con un paso WO cuyas 6 senales se modulan por el perfil
    del GBest:  senal' = base + (1 + amp*w_senal)*(senal - base)."""
    mc, fc, cc = role_counts
    base = FEATURE_BASELINE_DEFAULTS

    def m(name):
        return base[name] + (1.0 + amp * profile.get(name, 0.0)) * (signal_state[name] - base[name])

    work = positions.copy()
    gbest_row = np.asarray(best_pos, dtype=float)
    return apply_wo_movement_single(
        work, i, lb, ub, dim, n_agents, mc, fc, cc,
        best_pos, second_pos, gbest_row,
        m("alpha"), m("beta"), m("R"), m("danger_signal"), m("safety_signal"), rng=rng,
    )


def _peak_ram_mb():
    """RAM pico del proceso (MB). ru_maxrss: bytes en macOS, KB en Linux; NaN si no aplica."""
    try:
        import resource
        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return ru / (1024 * 1024) if sys.platform == "darwin" else ru / 1024
    except Exception:
        return float("nan")


def run_one(problem, mode, args, max_fes, seed):
    rng = np.random.default_rng(seed)
    start = time.perf_counter()
    t_init = 0.0   # tiempo de inicializacion (s)
    t_shap = 0.0   # tiempo acumulado en explicaciones SHAP (s)
    n_agents = args.agents
    dim, lb, ub = problem.dim, problem.lb, problem.ub
    lb_arr, ub_arr = np.asarray(lb, dtype=float), np.asarray(ub, dtype=float)

    budget = FESBudget(max_fes)
    eval_search = counting_objective(problem.evaluate, budget, "search")
    eval_shap = counting_objective(problem.evaluate, budget, "shap")
    eval_interv = counting_objective(problem.evaluate, budget, "intervention")

    _t0 = time.perf_counter()
    if getattr(problem, "family", None) == "tmlap":
        positions = problem.initial_population(
            n_agents, rng, init_mode=args.init_mode,
            on_eval=lambda: budget.spend("init", 1),
        )
    else:
        positions = problem.initial_population(n_agents, rng)
    t_init = time.perf_counter() - _t0
    best_pos = np.zeros(dim); second_pos = np.zeros(dim)
    best_score = float("inf"); second_score = float("inf")
    male_count, female_count, child_count = walrus_role_counts(n_agents)
    role_counts = (male_count, female_count, child_count)

    controller = SHAPFitnessController(max_fes=max_fes, n_agents=n_agents, random_state=seed)
    window = controller.profile.stagnation_window
    guard = controller.profile.guard_window
    late = controller.profile.late_fes
    cooldown = controller.profile.action_cooldown
    max_interv = controller.max_interventions
    shap_cost = controller.shap_cost_estimate

    pbest = np.full(n_agents, np.inf)
    last_improve_fes = np.zeros(n_agents, dtype=np.int64)
    last_action_fes = {}
    n_interventions = 0
    initial_fitness = np.nan
    w_records = []  # traza por explicación SHAP: fes, agente, w, dominante y 6 contribuciones
    curve = []      # convergencia del WO: (fes, best_score) al inicio de cada iteración

    while not budget.exhausted():
        fes_start = budget.total
        (positions, fitness_values, best_score, best_pos,
         second_score, second_pos) = evaluate_and_update_leaders(
            positions, lb, ub, eval_search,
            best_score, best_pos, second_score, second_pos, budget=budget,
        )
        if not np.isfinite(second_score):
            second_pos = best_pos.copy()
        if not np.isfinite(initial_fitness):
            initial_fitness = float(best_score)
        curve.append((int(budget.total), float(best_score)))

        for i in range(n_agents):
            f = fitness_values[i]
            if not np.isfinite(f):
                continue
            if not np.isfinite(pbest[i]) or f < pbest[i] - improvement_threshold(pbest[i]):
                pbest[i] = f
                last_improve_fes[i] = budget.total

        if budget.exhausted():
            break

        alpha, beta, A, R, danger_signal, safety_signal = iteration_signals(
            fes_start, max_fes, rng
        )
        signal_state = {
            "alpha": float(alpha), "beta": float(beta), "A": float(A), "R": float(R),
            "danger_signal": float(danger_signal), "safety_signal": float(safety_signal),
        }

        if mode == "elite":
            gbest_profile = None   # perfil del GBest (1 SHAP por iteración, on-demand)
            fes_since = budget.total - last_improve_fes
            order = np.argsort(-fes_since)
            for idx in order:
                i = int(idx)
                if budget.exhausted():
                    break
                fsi = int(budget.total - last_improve_fes[i])
                if fsi < window:
                    break
                if n_interventions >= max_interv:
                    break
                if budget.total < guard or budget.total > late:
                    break
                last = last_action_fes.get(i)
                if last is not None and budget.total - last < cooldown:
                    continue
                if not budget.can_afford(shap_cost):
                    break

                if gbest_profile is None:      # perfil del GBest (1 SHAP por iteración)
                    _ts = time.perf_counter()
                    gbest_profile, _ginfo = _gbest_profile(
                        eval_shap, controller, positions, fitness_values, lb, ub, dim,
                        role_counts, best_pos, second_pos, best_score, signal_state,
                        max_fes, fes_start, rng)
                    t_shap += time.perf_counter() - _ts
                    _rec = {"fes": int(budget.total), "agent_index": -1,
                            "dominant_feature": str((_ginfo or {}).get("dominant_feature", "")),
                            "dominant_value": float((_ginfo or {}).get("dominant_value", 0.0)),
                            "absolute_pressure": float((_ginfo or {}).get("absolute_pressure", 0.0))}
                    for _f, _v in ((_ginfo or {}).get("values", {}) or {}).items():
                        _rec[f"shap_{_f}"] = float(_v)
                    w_records.append(_rec)   # agent_index=-1 marca el perfil del GBest

                candidate = _reposition_elite(
                    positions, i, gbest_profile, signal_state, lb, ub, dim, n_agents,
                    role_counts, best_pos, second_pos, rng, controller.amplification_factor)
                candidate = np.clip(candidate, lb_arr, ub_arr)
                cand_fit = float(eval_interv(candidate))

                positions[i, :] = candidate
                fitness_values[i] = cand_fit
                last_improve_fes[i] = budget.total
                last_action_fes[i] = int(budget.total)
                n_interventions += 1
                if cand_fit < pbest[i]:
                    pbest[i] = cand_fit
                if cand_fit < best_score:
                    best_score = cand_fit; best_pos = candidate.copy()
                elif cand_fit < second_score:
                    second_score = cand_fit; second_pos = candidate.copy()

        if budget.exhausted():
            break

        gbest_x = np.tile(best_pos, (n_agents, 1))
        positions = apply_wo_movement(
            positions, lb, ub, dim, n_agents,
            male_count, female_count, child_count,
            best_pos, second_pos, gbest_x,
            alpha, beta, R, danger_signal, safety_signal, rng=rng,
        )

    elapsed = time.perf_counter() - start
    optimum = getattr(problem, "optimum", np.nan)
    optimum_f = float(optimum) if optimum is not None and np.isfinite(float(optimum)) else np.nan
    gap = float(best_score - optimum_f) if np.isfinite(optimum_f) else np.nan

    row = {
        "mode": mode, "seed": int(seed),
        "initial_fitness": float(initial_fitness), "final_fitness": float(best_score),
        "optimum": optimum_f, "gap": gap, "n_interventions": int(n_interventions),
        "elapsed_seconds": float(elapsed),
        "t_init_seconds": float(t_init),
        "t_shap_seconds": float(t_shap),
        "ram_peak_mb": float(_peak_ram_mb()),
    }
    row.update(budget.as_dict())
    return row, w_records, curve


def main():
    args = parse_args()
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    problems = make_problems(args)
    out = Path(args.output); (out / "values").mkdir(parents=True, exist_ok=True)

    rows, w_all, curves_all = [], [], []
    for p_idx, (problem, label) in enumerate(problems):
        print(f"\n=== {label} (dim={problem.dim}) ===", flush=True)
        for run_id in range(1, args.runs + 1):
            seed = int(args.seed + p_idx * 1000 + run_id - 1)   # único por (func,run); compartido entre modos
            for mode in modes:
                r, wrec, crv = run_one(problem, mode, args, args.max_fes, seed)
                r["problem"] = label; r["run_id"] = run_id
                rows.append(r)
                for _rec in wrec:
                    w_all.append({"problem": label, "run_id": run_id, "seed": seed, **_rec})
                if crv:  # curva de convergencia muestreada (~60 puntos) para no inflar
                    arr = np.asarray(crv, dtype=float)
                    k = max(1, len(arr) // 60)
                    samp = arr[::k]
                    if not np.array_equal(samp[-1], arr[-1]):
                        samp = np.vstack([samp, arr[-1]])
                    for fes_v, best_v in samp:
                        curves_all.append({"problem": label, "run_id": run_id, "mode": mode,
                                           "fes": int(fes_v), "best_fitness": float(best_v)})
            print(f"  run={run_id:2d} done (seed={seed})", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / "values" / "ablation_summary.csv", index=False)
    if w_all:
        # Traza SHAP completa por explicacion: FES (momento) + 6 contribuciones + dominante.
        cols = ["problem", "run_id", "seed", "fes", "agent_index",
                "dominant_feature", "dominant_value", "absolute_pressure",
                "shap_alpha", "shap_beta", "shap_A", "shap_R",
                "shap_danger_signal", "shap_safety_signal"]
        trace = pd.DataFrame(w_all)
        trace = trace[[c for c in cols if c in trace.columns]]
        trace.to_csv(out / "values" / "shap_trace.csv", index=False)
    if curves_all:
        # Convergencia del WO (best vs FES), muestreada, por corrida y modo.
        pd.DataFrame(curves_all).to_csv(out / "values" / "curves.csv", index=False)

    # --- Resumen por instancia (Wilcoxon pareado: elite vs base) ---
    try:
        from scipy.stats import wilcoxon
    except Exception:
        wilcoxon = None
    print("\n" + "=" * 72)
    print(f"ABLATION {args.problem} | MaxFES={args.max_fes} | runs={args.runs}")
    print("=" * 72)
    print(f"{'instancia':26s} {'base':>11s} {'elite':>11s} {'elite<base':>11s} {'p(elite-base)':>14s}")
    summary_rows = []
    for label in df["problem"].unique():
        sub = df[df["problem"] == label]
        piv = sub.pivot(index="run_id", columns="mode", values="final_fitness")
        means = {m: piv[m].mean() for m in modes if m in piv}
        line = f"{label:26s} " + " ".join(
            f"{means.get(m, float('nan')):11.3f}" for m in ["base", "elite"])
        rec = {"problem": label, "mean_base": means.get("base", np.nan),
               "mean_elite": means.get("elite", np.nan)}
        if "elite" in piv and "base" in piv:
            wins = int((piv["elite"] < piv["base"]).sum())
            line += f" {wins:5d}/{len(piv):<5d}"
            rec["elite_wins_vs_base"] = wins
            rec["n"] = int(len(piv))
            if wilcoxon is not None:
                try:
                    _, p = wilcoxon(piv["elite"], piv["base"])
                    line += f" {p:14.4f}"
                    rec["p_elite_vs_base"] = float(p)
                except Exception:
                    line += f" {'--':>14s}"
        summary_rows.append(rec)
        print(line, flush=True)
    pd.DataFrame(summary_rows).to_csv(out / "values" / "ablation_by_function.csv", index=False)
    print(f"\nCSV: {out / 'values'}")


if __name__ == "__main__":
    main()
