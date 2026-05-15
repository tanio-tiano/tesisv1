"""Adaptador del Two-stage Multi-Level Allocation Problem (TMLAP).

Cada cliente se asigna a exactamente un hub. El costo es la suma de:

- distancias cliente-hub para la asignacion elegida;
- costos fijos de los hubs abiertos (i.e. hubs que reciben >= 1 cliente).

Restricciones: capacidad por hub y distancia maxima cliente-hub (D_max).

Posiciones en espacio continuo: ``positions[i, j]`` en ``[0, n_hubs - 1]``.
El ``decode`` redondea a entero. El ``repair`` corrige factibilidad
(capacidades + D_max) priorizando los clientes con menos hubs factibles.

Para WO, la inicializacion puede ser:

- ``init_mode="random"`` (recomendado para instancias grandes): solo repair.
- ``init_mode="local_search"``: repair + 1 pasada de local search (costo
  O(n_clients^2 * n_hubs^2); usar solo en instancias chicas).
"""

import ast
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wo_core.diversity import population_diversity
from wo_core.walrus import (
    apply_wo_movement,
    evaluate_and_update_leaders,
    iteration_signals,
    r_signal_from_alpha_and_danger,
    walrus_role_counts,
)


@dataclass(frozen=True)
class TMLAPProblem:
    """Adaptador TMLAP. Inmutable; los metodos calculan, no mutan estado."""

    name: str
    distances: np.ndarray
    fixed_costs: np.ndarray
    capacities: np.ndarray
    d_max: float
    family: str = "tmlap"
    optimum: object = None  # se completa via solve_exact_by_backtracking si aplica
    benchmark: str = "tmlap"

    @property
    def n_clients(self):
        return int(self.distances.shape[0])

    @property
    def n_hubs(self):
        return int(self.distances.shape[1])

    @property
    def dim(self):
        return int(self.distances.shape[0])

    @property
    def lb(self):
        return np.zeros(self.n_clients, dtype=float)

    @property
    def ub(self):
        return np.full(self.n_clients, self.n_hubs - 1, dtype=float)

    # ------------------------------------------------------------------
    # Helpers especificos de TMLAP.
    # ------------------------------------------------------------------
    def feasible_hubs_for_client(self, client):
        hubs = np.where(self.distances[client, :] <= self.d_max)[0]
        if hubs.size == 0:
            hubs = np.array([int(np.argmin(self.distances[client, :]))], dtype=int)
        return hubs.astype(int)

    def decode(self, position):
        values = np.rint(np.asarray(position, dtype=float)).astype(int)
        return np.clip(values, 0, self.n_hubs - 1)

    def position_from_assignment(self, assignment, rng, jitter=0.04):
        values = np.asarray(assignment, dtype=float)
        if jitter > 0:
            values = values + rng.normal(0.0, jitter, size=self.n_clients)
        return np.clip(values, self.lb, self.ub)

    def violation_stats(self, assignment):
        assignment = np.asarray(assignment, dtype=int)
        loads = np.bincount(assignment, minlength=self.n_hubs)
        distance_excess = np.maximum(
            self.distances[np.arange(self.n_clients), assignment] - self.d_max, 0.0
        )
        capacity_excess = np.maximum(loads - self.capacities, 0)
        return {
            "is_feasible": bool(np.sum(distance_excess) == 0 and np.sum(capacity_excess) == 0),
            "distance_violations": int(np.sum(distance_excess > 0)),
            "distance_excess": float(np.sum(distance_excess)),
            "capacity_excess": int(np.sum(capacity_excess)),
            "capacity_pressure": float(np.sum(capacity_excess) / max(self.n_clients, 1)),
            "loads": loads.astype(int),
        }

    def objective_assignment(self, assignment):
        """Costo de una asignacion ya reparada (entera)."""
        assignment = np.asarray(assignment, dtype=int)
        used = np.zeros(self.n_hubs, dtype=bool)
        used[assignment] = True
        distance_cost = float(np.sum(self.distances[np.arange(self.n_clients), assignment]))
        fixed_cost = float(np.sum(self.fixed_costs[used]))
        return distance_cost + fixed_cost

    def repair(self, assignment):
        raw = np.asarray(assignment, dtype=int).copy()
        repaired = np.full(self.n_clients, -1, dtype=int)
        remaining = self.capacities.astype(int).copy()
        opened = np.zeros(self.n_hubs, dtype=bool)

        feasible_counts = [
            len(self.feasible_hubs_for_client(client)) for client in range(self.n_clients)
        ]
        order = sorted(range(self.n_clients), key=lambda c: (feasible_counts[c], c))

        for client in order:
            raw_hub = int(raw[client])
            allowed = self.feasible_hubs_for_client(client)
            available = [hub for hub in allowed if remaining[hub] > 0]
            if not available:
                available = [hub for hub in range(self.n_hubs) if remaining[hub] > 0]
            if not available:
                available = list(range(self.n_hubs))

            if raw_hub in available and self.distances[client, raw_hub] <= self.d_max:
                chosen = raw_hub
            else:
                def hub_score(hub):
                    open_cost = 0.0 if opened[hub] else self.fixed_costs[hub]
                    dmax_penalty = max(self.distances[client, hub] - self.d_max, 0.0) * 1000.0
                    capacity_penalty = 0.0 if remaining[hub] > 0 else 1000.0
                    return (
                        self.distances[client, hub]
                        + 0.05 * open_cost
                        + dmax_penalty
                        + capacity_penalty
                    )

                chosen = int(min(available, key=hub_score))

            repaired[client] = chosen
            remaining[chosen] -= 1
            opened[chosen] = True

        return repaired

    def local_search(self, assignment, rng, max_passes=2):
        """Una busqueda local greedy sobre la asignacion. Costo O(n_clients*n_feas*repair)."""
        current = self.repair(assignment)
        current_value = self.objective_assignment(current)
        for _ in range(max_passes):
            improved = False
            clients = np.arange(self.n_clients)
            rng.shuffle(clients)
            for client in clients:
                original_hub = int(current[client])
                candidate_hubs = list(self.feasible_hubs_for_client(client))
                rng.shuffle(candidate_hubs)
                for hub in candidate_hubs:
                    hub = int(hub)
                    if hub == original_hub:
                        continue
                    trial = current.copy()
                    trial[client] = hub
                    trial = self.repair(trial)
                    value = self.objective_assignment(trial)
                    if value < current_value - 1e-12:
                        current = trial
                        current_value = value
                        improved = True
                        break
            if not improved:
                break
        return current

    def random_feasible_assignment(self, rng, init_mode="local_search"):
        """Inicializa con repair (opcionalmente + 1 pasada de local_search)."""
        random_raw = rng.integers(0, self.n_hubs, size=self.n_clients)
        if init_mode == "random":
            return self.repair(random_raw)
        return self.local_search(self.repair(random_raw), rng, max_passes=1)

    # ------------------------------------------------------------------
    # Contrato WOProblem.
    # ------------------------------------------------------------------
    def evaluate(self, x):
        """Decodifica x continuo, repara, y devuelve el costo."""
        repaired = self.repair(self.decode(x))
        return float(self.objective_assignment(repaired))

    def initial_population(self, n_agents, rng=None, *, init_mode="local_search"):
        if rng is None:
            rng = np.random.default_rng()
        positions = np.zeros((n_agents, self.n_clients), dtype=float)
        for idx in range(n_agents):
            assignment = self.random_feasible_assignment(rng, init_mode=init_mode)
            positions[idx, :] = self.position_from_assignment(assignment, rng, jitter=0.12)
        return positions

    def make_value_function_for_shapley(self, state, positions, max_iter, steps, rng=None):
        """Devuelve closure ``v(coalition_state) -> float`` simulando k pasos del WO."""
        initial_positions = np.asarray(positions, dtype=float).copy()
        current_state = dict(state)

        if rng is None:
            outer_state = np.random.get_state()
        else:
            outer_state = rng.bit_generator.state

        def value_function(coalition_state):
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


# ----------------------------------------------------------------------
# Carga desde archivo .txt (formato self.n_clientes / self.n_hubs / ...).
# ----------------------------------------------------------------------
def _extract_scalar(text, key):
    pattern = rf"self\.{key}\s*=\s*([0-9]+(?:\.[0-9]+)?)"
    match = re.search(pattern, text)
    if not match:
        return None
    raw = match.group(1)
    return float(raw) if "." in raw else int(raw)


def _extract_literal(text, key):
    match = re.search(rf"self\.{key}\s*=", text)
    if not match:
        return None
    start = text.find("[", match.end())
    if start < 0:
        return None
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return ast.literal_eval(text[start : index + 1])
    raise ValueError(f"No se encontro cierre de lista para self.{key}")


def load_problem(path, clients=None, hubs=None):
    """Carga una instancia TMLAP desde un archivo .txt con asignaciones ``self.X``."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    n_clients = _extract_scalar(text, "n_clientes") or _extract_scalar(text, "n_clients")
    n_hubs = _extract_scalar(text, "n_hubs")
    distances = _extract_literal(text, "distancias")
    fixed_costs = _extract_literal(text, "costos_fijos") or _extract_literal(text, "costs")
    capacities = _extract_literal(text, "capacidad")
    d_max = _extract_scalar(text, "D_max")
    if any(value is None for value in [n_clients, n_hubs, distances, fixed_costs, capacities, d_max]):
        raise ValueError(f"No pude parsear la instancia TMLAP: {path}")

    distances = np.asarray(distances, dtype=float)
    fixed_costs = np.asarray(fixed_costs, dtype=float)
    capacities = np.asarray(capacities, dtype=int)

    if distances.shape != (int(n_clients), int(n_hubs)):
        raise ValueError(
            f"La matriz de distancias tiene forma {distances.shape}, "
            f"pero la instancia declara {(n_clients, n_hubs)}."
        )

    selected_clients = min(int(clients or n_clients), int(n_clients))
    selected_hubs = min(int(hubs or n_hubs), int(n_hubs))
    distances = distances[:selected_clients, :selected_hubs]
    fixed_costs = fixed_costs[:selected_hubs]
    capacities = capacities[:selected_hubs]

    if int(np.sum(capacities)) < selected_clients:
        raise ValueError(
            "La subinstancia no tiene capacidad suficiente: "
            f"capacidad={int(np.sum(capacities))}, clientes={selected_clients}."
        )

    name = f"{path.stem}_{selected_hubs}h_{selected_clients}c"
    return TMLAPProblem(
        name=name,
        distances=distances,
        fixed_costs=fixed_costs,
        capacities=capacities,
        d_max=float(d_max),
    )


def solve_exact_by_backtracking(problem, max_clients=12, max_hubs=8):
    """Backtracking exacto. Solo viable para instancias pequenas."""
    if problem.n_clients > max_clients or problem.n_hubs > max_hubs:
        return np.nan, None

    feasible = [list(problem.feasible_hubs_for_client(c)) for c in range(problem.n_clients)]
    order = sorted(range(problem.n_clients), key=lambda c: len(feasible[c]))
    min_future = np.zeros(problem.n_clients + 1)
    for pos in range(problem.n_clients - 1, -1, -1):
        client = order[pos]
        min_future[pos] = min_future[pos + 1] + float(
            np.min(problem.distances[client, feasible[client]])
        )

    best_value = float("inf")
    best_assignment = None
    assignment = np.full(problem.n_clients, -1, dtype=int)
    loads = np.zeros(problem.n_hubs, dtype=int)
    opened = np.zeros(problem.n_hubs, dtype=bool)

    def backtrack(pos, cost):
        nonlocal best_value, best_assignment
        if cost + min_future[pos] >= best_value:
            return
        if pos == problem.n_clients:
            if cost < best_value:
                best_value = float(cost)
                best_assignment = assignment.copy()
            return
        client = order[pos]
        hubs = sorted(
            feasible[client],
            key=lambda hub: problem.distances[client, hub]
            + (0 if opened[hub] else problem.fixed_costs[hub]),
        )
        for hub in hubs:
            hub = int(hub)
            if loads[hub] >= problem.capacities[hub]:
                continue
            was_open = bool(opened[hub])
            added = problem.distances[client, hub] + (
                0.0 if was_open else problem.fixed_costs[hub]
            )
            assignment[client] = hub
            loads[hub] += 1
            opened[hub] = True
            backtrack(pos + 1, cost + added)
            loads[hub] -= 1
            if loads[hub] == 0:
                opened[hub] = was_open
            assignment[client] = -1

    backtrack(0, 0.0)
    return best_value, best_assignment


# ----------------------------------------------------------------------
# Simulacion para Shapley (analoga a la de cec2022.py pero con repair en eval).
# ----------------------------------------------------------------------
def _rescale_population_to_diversity(positions, lb, ub, target_diversity):
    target = float(target_diversity)
    if not np.isfinite(target) or target <= 0:
        return positions.copy()
    current = population_diversity(positions)
    if not np.isfinite(current) or current <= 1e-12:
        return positions.copy()
    centroid = np.mean(positions, axis=0)
    scale = target / current
    return np.clip(centroid + (positions - centroid) * scale, lb, ub)


def _coalition_target_diversity(coalition_state, current_state):
    normalized = float(coalition_state.get("diversity", np.nan))
    domain_scale = float(current_state.get("diversity_domain_scale", np.nan))
    if (
        np.isfinite(normalized) and np.isfinite(domain_scale)
        and normalized > 0 and domain_scale > 0
    ):
        return normalized * domain_scale
    return float(current_state.get("raw_diversity", np.nan))


def _simulate(problem, initial_positions, coalition_state, current_state, max_iter, steps, rng):
    positions = _rescale_population_to_diversity(
        initial_positions,
        problem.lb,
        problem.ub,
        _coalition_target_diversity(coalition_state, current_state),
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
