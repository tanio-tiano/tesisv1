"""Acciones de rescate del controlador (Tabla 2 del informe).

Dos acciones canonicas:

- ``partial_restart`` recoloca a la fraccion de peores agentes alrededor de
  soluciones elite, preservando la memoria estructural de la busqueda.
  Modos: ``elite_guided``, ``elite_guided_wide``, ``elite_guided_aggressive``.
- ``random_reinjection`` sustituye a la fraccion de peores por agentes
  generados uniformemente dentro del dominio. La variante ``random_aggressive``
  agrega un jitter gaussiano sobre el muestreo.

Ambas aceptan un ``rng`` opcional (``np.random.Generator``); si es None,
usan ``np.random`` global.
"""

import numpy as np


PARTIAL_RESTART_MODES = (
    "elite_guided",
    "elite_guided_wide",
    "elite_guided_aggressive",
)

RANDOM_REINJECTION_MODES = (
    "random",
    "random_aggressive",
)


def _draw_normal(rng, scale, size):
    if rng is None:
        return np.random.normal(loc=0.0, scale=scale, size=size)
    return rng.normal(loc=0.0, scale=scale, size=size)


def _draw_uniform(rng, size):
    if rng is None:
        return np.random.rand(*size) if isinstance(size, tuple) else np.random.rand(size)
    return rng.random(size=size)


def _randint(rng, high):
    if rng is None:
        return int(np.random.randint(0, high))
    return int(rng.integers(0, high))


def apply_partial_restart(
    positions,
    fitness_values,
    lb,
    ub,
    best_pos,
    second_pos,
    fraction,
    mode,
    rng=None,
):
    """Partial restart guiado por elites.

    Selecciona la ``fraction`` de peores agentes y los recoloca alrededor de
    X_best, X_second y otros lideres top-5, con dispersion gaussiana controlada
    por ``mode`` (``elite_guided`` < ``_wide`` < ``_aggressive``).
    """
    positions = positions.copy()
    n_agents, dim = positions.shape
    n_selected = max(1, int(round(fraction * n_agents)))
    n_selected = min(n_selected, n_agents - 1)
    worst_indices = np.argsort(fitness_values)[::-1][:n_selected]

    best_indices = np.argsort(fitness_values)[: max(2, min(5, n_agents))]
    elite_positions = [np.asarray(best_pos, dtype=float)]
    second = np.asarray(second_pos, dtype=float)
    if np.all(np.isfinite(second)):
        elite_positions.append(second)
    elite_positions.extend(positions[index].copy() for index in best_indices)

    spread_by_mode = {
        "elite_guided": 0.035,
        "elite_guided_wide": 0.075,
        "elite_guided_aggressive": 0.12,
    }
    spread = spread_by_mode.get(mode, spread_by_mode["elite_guided"])
    span = np.asarray(ub, dtype=float) - np.asarray(lb, dtype=float)

    for index in worst_indices:
        anchor = elite_positions[_randint(rng, len(elite_positions))]
        noise = _draw_normal(rng, spread, dim) * span
        positions[index, :] = anchor + noise

    return np.clip(positions, lb, ub), worst_indices


def apply_random_reinjection(
    positions,
    fitness_values,
    lb,
    ub,
    fraction,
    mode,
    rng=None,
):
    """Random reinjection dentro del dominio.

    Sustituye a la ``fraction`` de peores agentes por muestras uniformes en
    [lb, ub]. El modo ``random_aggressive`` agrega un jitter gaussiano de
    desviacion 0.05 sobre el span del dominio.
    """
    positions = positions.copy()
    n_agents, dim = positions.shape
    n_selected = max(1, int(round(fraction * n_agents)))
    n_selected = min(n_selected, n_agents - 1)
    worst_indices = np.argsort(fitness_values)[::-1][:n_selected]
    lb_arr = np.asarray(lb, dtype=float)
    ub_arr = np.asarray(ub, dtype=float)
    span = ub_arr - lb_arr

    uniform = _draw_uniform(rng, (n_selected, dim))
    candidates = uniform * span + lb_arr
    if mode == "random_aggressive":
        candidates = candidates + _draw_normal(rng, 0.05, (n_selected, dim)) * span

    positions[worst_indices, :] = candidates
    return np.clip(positions, lb_arr, ub_arr), worst_indices


def dispatch_rescue(
    action,
    mode,
    positions,
    fitness_values,
    lb,
    ub,
    best_pos,
    second_pos,
    fraction,
    rng=None,
):
    """Dispatcher unificado para ambas acciones canonicas.

    Acepta ``action in {'partial_restart', 'random_reinjection'}`` y enruta a
    la funcion correspondiente con un fallback si el ``mode`` no coincide.
    """
    if action == "partial_restart" or mode in PARTIAL_RESTART_MODES:
        chosen_mode = mode if mode in PARTIAL_RESTART_MODES else "elite_guided"
        return apply_partial_restart(
            positions,
            fitness_values,
            lb,
            ub,
            best_pos,
            second_pos,
            fraction,
            chosen_mode,
            rng=rng,
        )
    if action == "random_reinjection" or mode in RANDOM_REINJECTION_MODES:
        chosen_mode = mode if mode in RANDOM_REINJECTION_MODES else "random"
        return apply_random_reinjection(
            positions,
            fitness_values,
            lb,
            ub,
            fraction,
            chosen_mode,
            rng=rng,
        )
    raise ValueError(
        f"Accion/mode desconocidos: action={action!r}, mode={mode!r}. "
        f"Validas: partial_restart {PARTIAL_RESTART_MODES}, random_reinjection {RANDOM_REINJECTION_MODES}."
    )
