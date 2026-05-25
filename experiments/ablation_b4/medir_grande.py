"""Mide tiempos en la instancia grande (1000c x 500h) para ESTIMAR cuanto tardaria
una corrida completa del ablation, sin esperar las horas. Cronometra por separado:
carga de instancia, inicializacion (random), 1 evaluacion poblacional, 1 movimiento
WO y 1 explicacion SHAP; luego extrapola al MaxFES dado.

Uso (desde la raiz del repo):
    python experiments/ablation_b4/medir_grande.py [max_fes] [agents]
"""
import sys, time
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from problems.factory import parse_problem_spec
from shap_controller import SHAPFitnessController
from wo_core.agent_sim import make_value_function_for_agent
from wo_core.fes import FESBudget, counting_objective
from wo_core.walrus import (apply_wo_movement, evaluate_and_update_leaders,
                            iteration_signals, walrus_role_counts)

MAXFES = int(sys.argv[1]) if len(sys.argv) > 1 else 50000
N = int(sys.argv[2]) if len(sys.argv) > 2 else 30
rng = np.random.default_rng(1234)

t = time.perf_counter()
problem = parse_problem_spec("tmlap:4.instancia_grande.txt", clients=None, hubs=None)
t_load = time.perf_counter() - t
dim, lb, ub = problem.dim, problem.lb, problem.ub
print(f"Instancia: {problem.name} | dim(clientes)={dim} hubs={problem.n_hubs} | agentes={N} | MaxFES={MAXFES}")
print(f"[1] cargar instancia : {t_load:.3f}s")

budget = FESBudget(MAXFES)
t = time.perf_counter()
positions = problem.initial_population(N, rng, init_mode="random",
                                       on_eval=lambda: budget.spend("init", 1))
t_init = time.perf_counter() - t
print(f"[2] INICIALIZACION   : {t_init:.3f}s   (init_mode=random, fes_init={budget.total})")

eval_search = counting_objective(problem.evaluate, budget, "search")
best_pos = np.zeros(dim); second_pos = np.zeros(dim)
best_score = second_score = float("inf")
mc, fc, cc = walrus_role_counts(N)
t = time.perf_counter()
positions, fit, best_score, best_pos, second_score, second_pos = evaluate_and_update_leaders(
    positions, lb, ub, eval_search, best_score, best_pos, second_score, second_pos, budget=budget)
t_eval = time.perf_counter() - t
print(f"[3] evaluar {N} agentes : {t_eval:.3f}s   ({t_eval/N*1000:.2f} ms/agente)")

if not np.isfinite(second_score): second_pos = best_pos.copy()
alpha, beta, A, R, danger, safety = iteration_signals(budget.total, MAXFES, rng)
gbest_x = np.tile(best_pos, (N, 1))
t = time.perf_counter()
positions = apply_wo_movement(positions, lb, ub, dim, N, mc, fc, cc,
                              best_pos, second_pos, gbest_x, alpha, beta, R, danger, safety, rng=rng)
t_move = time.perf_counter() - t
print(f"[4] movimiento WO    : {t_move:.3f}s")

ss = {"alpha": alpha, "beta": beta, "A": A, "R": R, "danger_signal": danger, "safety_signal": safety}
ctrl = SHAPFitnessController(MAXFES, N, 1234)
eval_shap = counting_objective(problem.evaluate, budget, "shap")
ctrl.record_state(ss, best_score)
vf = make_value_function_for_agent(eval_shap, 0, positions, lb, ub, dim, (mc, fc, cc),
                                   best_pos, second_pos, ss, MAXFES, budget.total, ctrl.shapley_steps, rng=rng)
t = time.perf_counter()
ctrl.explain_fitness({**ss, "agent_index": 0, "agent_fitness": float(fit[0]),
                      "fes_since_improve": MAXFES, "fes": budget.total}, value_function=vf)
t_shap = time.perf_counter() - t
print(f"[5] 1 SHAP (64x{ctrl.shapley_steps} pasos): {t_shap:.3f}s")

# ---- Extrapolacion ----
iters = MAXFES // N
t_iter = t_eval + t_move
t_base = t_init + t_iter * iters
n_shap = ctrl.max_interventions
t_shap_mode = t_base + t_shap * n_shap
print("\n" + "=" * 60)
print(f"ESTIMACION (MaxFES={MAXFES}, ~{iters} iteraciones, agentes={N})")
print("=" * 60)
print(f"  init                         : {t_init:6.1f}s")
print(f"  busqueda+movimiento (x{iters}) : {t_iter*iters:6.1f}s  ({t_iter*iters/60:.1f} min)")
print(f"  SHAP (~{n_shap} explicaciones)    : {t_shap*n_shap:6.1f}s")
print(f"  -> 1 run modo base/blind/wfix : ~{t_base/60:.1f} min")
print(f"  -> 1 run modo shap            : ~{t_shap_mode/60:.1f} min")
print(f"  -> 1 CORRIDA (4 modos)        : ~{(3*t_base+t_shap_mode)/60:.1f} min")
print(f"  -> 30 corridas (4 modos)      : ~{30*(3*t_base+t_shap_mode)/3600:.1f} h")
print(f"  -> 51 corridas (4 modos)      : ~{51*(3*t_base+t_shap_mode)/3600:.1f} h")
