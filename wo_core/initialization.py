"""Inicializacion uniforme de la poblacion en un dominio rectangular.

Para problemas que requieren una inicializacion mas sofisticada (p.ej.
TMLAP con repair), la clase del problema deberia exponer su propio
``initial_population``.
"""

import numpy as np


def uniform_population(n_agents, dim, lb, ub, rng=None):
    """Muestreo uniforme en el rectangulo [lb, ub] de ``n_agents`` agentes."""
    lb_arr = np.asarray(lb, dtype=float)
    ub_arr = np.asarray(ub, dtype=float)

    if lb_arr.ndim == 0 and ub_arr.ndim == 0:
        if rng is None:
            return np.random.rand(n_agents, dim) * (float(ub_arr) - float(lb_arr)) + float(lb_arr)
        return rng.uniform(float(lb_arr), float(ub_arr), size=(n_agents, dim))

    # Vector lb/ub: muestrear por dimension.
    positions = np.zeros((n_agents, dim), dtype=float)
    for i in range(dim):
        lb_i = float(lb_arr[i] if lb_arr.ndim > 0 else lb_arr)
        ub_i = float(ub_arr[i] if ub_arr.ndim > 0 else ub_arr)
        if rng is None:
            positions[:, i] = np.random.rand(n_agents) * (ub_i - lb_i) + lb_i
        else:
            positions[:, i] = rng.uniform(lb_i, ub_i, size=n_agents)
    return positions
