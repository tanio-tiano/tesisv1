"""Dinamica del Walrus Optimizer (Han et al. 2024).

Implementacion unica, independiente del problema. Soporta:

- 4 regimenes de movimiento (eq. 9, 12-14, 17, 18-21 del paper).
- Senales temporales alpha, beta, danger_signal, safety_signal.
- Roles macho/hembra/cria (eq. proporcion 0.4 / 0.4 / 0.2).
- ``gbest_x`` actualizado con X_best vigente cada iteracion (paper fiel).
- ``rng`` opcional (``np.random.Generator``). Si None, usa ``np.random`` global.
"""

import numpy as np

from .halton import halton
from .levy_flight import levy_flight


def _rand(rng, size=None):
    if rng is None:
        return np.random.rand() if size is None else np.random.rand(size)
    return rng.random() if size is None else rng.random(size=size)


def _permutation(n, rng):
    if rng is None:
        return np.random.permutation(n)
    return rng.permutation(n)


def walrus_role_counts(n_agents):
    """Devuelve (machos, hembras, crias) usando la proporcion fija del paper.

    Las crias son ~10% del total; el resto se divide en partes iguales entre
    machos y hembras (con macho >= hembra cuando la division no es entera).
    """
    if n_agents < 3:
        females = n_agents // 2
        males = n_agents - females
        return males, females, 0

    children = int(round(n_agents * 0.10))
    children = min(max(1, children), n_agents - 2)
    adults = n_agents - children
    males = int(np.ceil(adults / 2))
    females = adults - males
    return males, females, children


def enforce_bounds(positions, lb, ub):
    """Proyecta cada agente al rectangulo [lb, ub]."""
    return np.clip(positions, lb, ub)


def evaluate_and_update_leaders(
    positions, lb, ub, objective, best_score, best_pos, second_score, second_pos
):
    """Evalua la poblacion proyectada y actualiza X_best/X_second.

    ``objective`` es una funcion ``np.ndarray -> float`` (la del problema).
    Retorna ``(positions_proj, fitness_values, best_score, best_pos, second_score, second_pos)``.
    """
    positions = np.clip(positions, lb, ub)
    fitness_values = np.zeros(positions.shape[0], dtype=float)
    for i in range(positions.shape[0]):
        fitness = float(objective(positions[i, :]))
        fitness_values[i] = fitness
        if fitness < best_score:
            best_score = fitness
            best_pos = positions[i, :].copy()
        if fitness > best_score and fitness < second_score:
            second_score = fitness
            second_pos = positions[i, :].copy()
    return positions, fitness_values, best_score, best_pos, second_score, second_pos


def iteration_signals(iteration, max_iter, rng=None):
    """Calcula las 5 senales del WO en la iteracion ``iteration`` de ``max_iter``.

    Retorna ``(alpha, beta, r_signal, danger_signal, safety_signal)``.
    Consume 2 numeros aleatorios del rng (uno para r, otro para safety).
    """
    alpha = 1 - iteration / max(max_iter, 1)
    beta = 1 - 1 / (1 + np.exp((0.5 * max_iter - iteration) / max(max_iter, 1) * 10))
    r_signal = 2 * _rand(rng) - 1
    danger_signal = 2 * alpha * r_signal
    safety_signal = _rand(rng)
    return float(alpha), float(beta), float(r_signal), float(danger_signal), float(safety_signal)


def apply_wo_movement(
    positions,
    lb,
    ub,
    dim,
    n_agents,
    male_count,
    female_count,
    child_count,
    best_pos,
    second_pos,
    gbest_x,
    alpha,
    beta,
    r_signal,
    danger_signal,
    safety_signal,
    rng=None,
):
    """Aplica el regimen del WO segun (danger_signal, safety_signal).

    Modifica ``positions`` in place y devuelve la version proyectada al dominio.
    Sigue exactamente las 4 ramas del paper:

    1. ``|danger| >= 1``               -> exploracion por diferencias (eq. 9).
    2. ``|danger| < 1 y safety >= 0.5``-> reproduccion: machos Halton, hembras
       hacia macho+gbest, crias con vuelo de Levy (eq. 12-14).
    3. ``safety < 0.5 y |danger| >= 0.5`` -> contraccion guiada (eq. 17).
    4. ``safety < 0.5 y |danger| < 0.5`` -> explotacion por dos lideres (eq. 18-21).
    """
    lb_arr = np.asarray(lb, dtype=float)
    ub_arr = np.asarray(ub, dtype=float)

    if abs(danger_signal) >= 1:
        r3 = _rand(rng)
        p1 = _permutation(n_agents, rng)
        p2 = _permutation(n_agents, rng)
        positions = positions + (beta * r3**2) * (positions[p1, :] - positions[p2, :])
        return enforce_bounds(positions, lb_arr, ub_arr)

    if safety_signal >= 0.5:
        # Machos: secuencia de Halton.
        for i in range(male_count):
            positions[i, :] = lb_arr + halton(i + 1, 7) * (ub_arr - lb_arr)

        # Hembras: combinacion macho + gbest.
        for j in range(male_count, male_count + female_count):
            male_index = min(j - male_count, male_count - 1)
            positions[j, :] = (
                positions[j, :]
                + alpha * (positions[male_index, :] - positions[j, :])
                + (1 - alpha) * (gbest_x[j, :] - positions[j, :])
            )

        # Crias: vuelo de Levy alrededor de gbest.
        for i in range(n_agents - child_count, n_agents):
            p = _rand(rng)
            o = gbest_x[i, :] + positions[i, :] * levy_flight(dim, rng)
            positions[i, :] = p * (o - positions[i, :])
        positions = enforce_bounds(positions, lb_arr, ub_arr)
        return positions

    if abs(danger_signal) >= 0.5:
        # Contraccion guiada por gbest (eq. 17).
        for i in range(n_agents):
            r4 = _rand(rng)
            positions[i, :] = (
                positions[i, :] * r_signal
                - np.abs(gbest_x[i, :] - positions[i, :]) * r4**2
            )
        return enforce_bounds(positions, lb_arr, ub_arr)

    # safety < 0.5 y |danger| < 0.5: explotacion por dos lideres (eq. 18-21).
    for i in range(n_agents):
        for j_dim in range(dim):
            theta1 = _rand(rng)
            a1 = beta * _rand(rng) - beta
            b1 = np.tan(theta1 * np.pi)
            x1 = best_pos[j_dim] - a1 * b1 * abs(best_pos[j_dim] - positions[i, j_dim])

            theta2 = _rand(rng)
            a2 = beta * _rand(rng) - beta
            b2 = np.tan(theta2 * np.pi)
            x2 = second_pos[j_dim] - a2 * b2 * abs(second_pos[j_dim] - positions[i, j_dim])
            positions[i, j_dim] = (x1 + x2) / 2
    return enforce_bounds(positions, lb_arr, ub_arr)


def r_signal_from_alpha_and_danger(alpha, danger_signal):
    """Despeja ``r_signal`` de ``danger_signal = 2 * alpha * r``.

    Usado por la value function de Shapley para reconstruir r_signal a partir
    de la coalicion (danger_signal puede venir de la coalicion o del baseline).
    """
    alpha = float(alpha)
    danger_signal = float(danger_signal)
    if not np.isfinite(alpha) or abs(alpha) <= 1e-12:
        return 0.0
    return float(np.clip(danger_signal / (2.0 * alpha), -1.0, 1.0))
