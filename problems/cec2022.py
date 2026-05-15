"""Adaptador del benchmark CEC 2022 (envuelve ``opfunu.cec_based.cec2022``)."""

import re

import numpy as np
from opfunu.cec_based import cec2022

from wo_core.diversity import normalize_diversity_by_domain, population_diversity
from wo_core.initialization import uniform_population
from wo_core.walrus import (
    apply_wo_movement,
    evaluate_and_update_leaders,
    iteration_signals,
    r_signal_from_alpha_and_danger,
    walrus_role_counts,
)


FUNCTION_IDS = tuple(range(1, 13))
BENCHMARK_NAME = "cec2022"
YEAR = "2022"


def _function_class(function_id):
    class_name = f"F{int(function_id)}{YEAR}"
    if not hasattr(cec2022, class_name):
        raise ValueError(f"No existe {class_name} en opfunu.cec_based.cec2022.")
    return getattr(cec2022, class_name)


def _family_for(function_id):
    if function_id <= 5:
        return "basic"
    if function_id <= 8:
        return "hybrid"
    return "composition"


def _clean_name(raw_name, function_id):
    text = str(raw_name or f"F{function_id}").strip()
    return re.sub(r"^F\d+:\s*", "", text).strip() or f"F{function_id}"


def parse_function_id(label):
    """Acepta 'F6' o 6 y devuelve int valido en FUNCTION_IDS."""
    if isinstance(label, str):
        normalized = label.strip().upper().lstrip("F")
        function_id = int(normalized)
    else:
        function_id = int(label)
    if function_id not in FUNCTION_IDS:
        raise ValueError(f"CEC 2022 define F{FUNCTION_IDS[0]}-F{FUNCTION_IDS[-1]}.")
    return function_id


class CECProblem:
    """Funcion CEC 2022 (F1-F12) adaptada al contrato ``WOProblem``."""

    def __init__(self, function_id, dim=10):
        self.function_id = parse_function_id(function_id)
        self.dim = int(dim)
        self.family = _family_for(self.function_id)
        cls = _function_class(self.function_id)
        self._inner = cls(ndim=self.dim)
        self.lb = self._normalize_bounds(getattr(self._inner, "lb", -100.0))
        self.ub = self._normalize_bounds(getattr(self._inner, "ub", 100.0))
        self.optimum = float(getattr(self._inner, "f_global", np.nan))
        self.name = _clean_name(getattr(self._inner, "name", ""), self.function_id)
        self.benchmark = BENCHMARK_NAME

    def _normalize_bounds(self, values):
        array = np.asarray(values, dtype=float).reshape(-1)
        if array.size == 1:
            return float(array[0])
        if array.size != self.dim:
            return np.resize(array, self.dim).astype(float)
        return array.astype(float)

    def project(self, x):
        return np.clip(np.asarray(x, dtype=float), self.lb, self.ub)

    def evaluate(self, x):
        return float(self._inner.evaluate(self.project(x)))

    def initial_population(self, n_agents, rng=None, **kwargs):
        return uniform_population(n_agents, self.dim, self.lb, self.ub, rng=rng)

    # ------------------------------------------------------------------
    # Value function para Shapley exacto sobre fitness.
    # ------------------------------------------------------------------
    def make_value_function_for_shapley(
        self,
        state,
        positions,
        max_iter,
        steps,
        rng=None,
    ):
        """Devuelve una closure ``v(coalition_state) -> float`` que simula k
        pasos del WO partiendo de ``positions`` con las senales de la coalicion.

        El rng global se restaura al estado previo a cada llamada para no
        contaminar la corrida principal.
        """
        initial_positions = np.asarray(positions, dtype=float).copy()
        current_state = dict(state)

        if rng is None:
            outer_state = np.random.get_state()
        else:
            outer_state = rng.bit_generator.state

        def value_function(coalition_state):
            # Guardar y restaurar el rng para no afectar la corrida principal.
            if rng is None:
                saved = np.random.get_state()
                np.random.set_state(outer_state)
                try:
                    return _simulate(self, initial_positions, coalition_state,
                                     current_state, max_iter, steps, None)
                finally:
                    np.random.set_state(saved)
            else:
                saved = rng.bit_generator.state
                rng.bit_generator.state = outer_state
                try:
                    return _simulate(self, initial_positions, coalition_state,
                                     current_state, max_iter, steps, rng)
                finally:
                    rng.bit_generator.state = saved

        return value_function


def _rescale_population_to_diversity(positions, lb, ub, target_diversity):
    """Re-escala posiciones alrededor del centroide para alcanzar una diversidad."""
    target = float(target_diversity)
    if not np.isfinite(target) or target <= 0:
        return positions.copy()
    current = population_diversity(positions)
    if not np.isfinite(current) or current <= 1e-12:
        return positions.copy()
    centroid = np.mean(positions, axis=0)
    scale = target / current
    return np.clip(centroid + (positions - centroid) * scale, lb, ub)


def _simulate(problem, initial_positions, coalition_state, current_state, max_iter, steps, rng):
    """Simula ``steps`` pasos del WO desde la coalicion. Retorna best fitness."""
    target_diversity = _coalition_target_diversity(coalition_state, current_state)
    positions = _rescale_population_to_diversity(
        initial_positions, problem.lb, problem.ub, target_diversity
    )

    n_agents = positions.shape[0]
    male_count, female_count, child_count = walrus_role_counts(n_agents)

    best_score = float("inf")
    best_pos = np.zeros(problem.dim, dtype=float)
    second_score = float("inf")
    second_pos = np.zeros(problem.dim, dtype=float)
    (
        positions,
        _fit,
        best_score,
        best_pos,
        second_score,
        second_pos,
    ) = evaluate_and_update_leaders(
        positions, problem.lb, problem.ub, problem.evaluate,
        best_score, best_pos, second_score, second_pos,
    )
    if not np.isfinite(second_score):
        second_pos = best_pos.copy()

    gbest_x = np.tile(best_pos, (n_agents, 1))
    start_iter = int(
        np.clip(round(float(coalition_state.get("iteration", 0))), 0, max_iter - 1)
    )

    for offset in range(max(1, int(steps))):
        iteration = min(start_iter + offset, max_iter - 1)
        if offset == 0:
            alpha = float(coalition_state.get("alpha", 1 - iteration / max(max_iter, 1)))
            beta = float(coalition_state.get(
                "beta",
                1 - 1 / (1 + np.exp((0.5 * max_iter - iteration) / max(max_iter, 1) * 10)),
            ))
            danger = float(coalition_state.get("danger_signal", 0.0))
            safety = float(coalition_state.get("safety_signal", 0.5))
            r_signal = r_signal_from_alpha_and_danger(alpha, danger)
        else:
            alpha, beta, r_signal, danger, safety = iteration_signals(iteration, max_iter, rng)

        positions = apply_wo_movement(
            positions, problem.lb, problem.ub, problem.dim, n_agents,
            male_count, female_count, child_count,
            best_pos, second_pos, gbest_x,
            alpha, beta, r_signal, danger, safety, rng=rng,
        )
        (
            positions,
            _fit,
            best_score,
            best_pos,
            second_score,
            second_pos,
        ) = evaluate_and_update_leaders(
            positions, problem.lb, problem.ub, problem.evaluate,
            best_score, best_pos, second_score, second_pos,
        )
        gbest_x = np.tile(best_pos, (n_agents, 1))
        if not np.isfinite(second_score):
            second_pos = best_pos.copy()

    return float(best_score)


def _coalition_target_diversity(coalition_state, current_state):
    """Convierte 'diversity' normalizada de la coalicion a escala cruda."""
    normalized = float(coalition_state.get("diversity", np.nan))
    domain_scale = float(current_state.get("diversity_domain_scale", np.nan))
    if (
        np.isfinite(normalized) and np.isfinite(domain_scale)
        and normalized > 0 and domain_scale > 0
    ):
        return normalized * domain_scale
    return float(current_state.get("raw_diversity", np.nan))


def get_function_metadata(function_id, dim=10):
    """Devuelve {name, family, optimum} sin crear el objeto opfunu pesado dos veces."""
    p = CECProblem(function_id, dim=dim)
    return {"name": p.name, "family": p.family, "optimum": p.optimum}
