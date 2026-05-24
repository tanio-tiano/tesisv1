"""Simulador SHAP **por agente** (independiente del problema).

El controlador SHAP atribuye el fitness *de un agente estancado* a las 6 senales
de control globales del WO (alpha, beta, A, R, danger_signal, safety_signal). La
``value function`` que consume ``SHAPFitnessController.explain_fitness`` simula
**solo ese agente** durante ``steps`` pasos del WO partiendo de su posicion
actual, con la poblacion congelada (para los regimenes acoplados), y devuelve el
mejor fitness alcanzado por el agente.

Costo por explicacion: ``2^6 coaliciones x steps`` evaluaciones del agente, es
decir ``64 * steps`` FES (p. ej. 192 con steps=3) — ~40x mas barato que la
version poblacional. Como la logica solo depende de ``objective`` (que ya cuenta
FES) y de la geometria ``lb/ub/dim``, este modulo es generico: lo usan por igual
los adaptadores CEC y TMLAP.
"""

from __future__ import annotations

import numpy as np

from .walrus import apply_wo_movement_single, iteration_signals


N_FEATURES = 6  # alpha, beta, A, R, danger_signal, safety_signal


def shap_cost_estimate(steps, n_features=N_FEATURES):
    """FES que consume una explicacion SHAP por agente: ``2^n_features * steps``."""
    return int((1 << int(n_features)) * max(1, int(steps)))


def _simulate_agent(
    objective,
    agent_index,
    frozen_positions,
    lb,
    ub,
    dim,
    male_count,
    female_count,
    child_count,
    best_pos,
    second_pos,
    coalition_state,
    current_state,
    max_fes,
    fes_start,
    steps,
    rng,
):
    """Simula ``steps`` pasos single-agent y devuelve el mejor fitness del agente."""
    work = np.asarray(frozen_positions, dtype=float).copy()
    pos = work[agent_index, :].astype(float)
    gbest_row = np.asarray(best_pos, dtype=float)
    n_agents = work.shape[0]
    best = float("inf")

    for offset in range(max(1, int(steps))):
        if offset == 0:
            # offset 0: senales de la coalicion (con fallback al estado actual).
            alpha = float(coalition_state.get("alpha", current_state["alpha"]))
            beta = float(coalition_state.get("beta", current_state["beta"]))
            R = float(coalition_state.get("R", current_state["R"]))
            danger = float(coalition_state.get("danger_signal", current_state["danger_signal"]))
            safety = float(coalition_state.get("safety_signal", current_state["safety_signal"]))
        else:
            # offset>0: el schedule sigue el reloj de FES (phi = fes_start/max_fes);
            # las componentes estocasticas (R, safety, ...) se regeneran con rng.
            alpha, beta, _A, R, danger, safety = iteration_signals(fes_start, max_fes, rng)

        work[agent_index, :] = pos
        pos = apply_wo_movement_single(
            work, agent_index, lb, ub, dim, n_agents,
            male_count, female_count, child_count,
            best_pos, second_pos, gbest_row,
            alpha, beta, R, danger, safety, rng=rng,
        )
        fitness = float(objective(pos))  # gasta 1 FES en el bucket de SHAP
        if fitness < best:
            best = fitness
    return best


def make_value_function_for_agent(
    objective,
    agent_index,
    frozen_positions,
    lb,
    ub,
    dim,
    role_counts,
    best_pos,
    second_pos,
    current_state,
    max_fes,
    fes_start,
    steps,
    rng=None,
):
    """Construye la ``value function`` por agente para Shapley exacto.

    ``objective`` debe contar FES en el bucket de SHAP (ver
    ``wo_core.fes.counting_objective(..., 'shap')``). El estado del ``rng`` se
    guarda y restaura en cada llamada para no contaminar la corrida principal.
    Devuelve ``v(coalition_state) -> float``.
    """
    frozen = np.asarray(frozen_positions, dtype=float).copy()
    state_snapshot = dict(current_state)
    male_count, female_count, child_count = role_counts

    if rng is None:
        outer_state = np.random.get_state()
    else:
        outer_state = rng.bit_generator.state

    def value_function(coalition_state):
        if rng is None:
            saved = np.random.get_state()
            np.random.set_state(outer_state)
            try:
                return _simulate_agent(
                    objective, agent_index, frozen, lb, ub, dim,
                    male_count, female_count, child_count,
                    best_pos, second_pos, coalition_state, state_snapshot,
                    max_fes, fes_start, steps, None,
                )
            finally:
                np.random.set_state(saved)
        else:
            saved = rng.bit_generator.state
            rng.bit_generator.state = outer_state
            try:
                return _simulate_agent(
                    objective, agent_index, frozen, lb, ub, dim,
                    male_count, female_count, child_count,
                    best_pos, second_pos, coalition_state, state_snapshot,
                    max_fes, fes_start, steps, rng,
                )
            finally:
                rng.bit_generator.state = saved

    return value_function
