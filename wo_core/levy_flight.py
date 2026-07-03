"""Vuelo de Levy (eq. 15-16 del paper de Walrus Optimizer).

Usado por el WO en la fase reproduccion para las crias.
"""

import math

import numpy as np


def levy_flight(dim, rng=None):
    """Devuelve un vector ``dim``-dimensional con distribucion Levy alpha-estable.

    Si ``rng`` se omite, usa ``np.random`` global. Si se pasa un
    ``np.random.Generator``, lo usa para reproducibilidad por corrida.
    """
    beta = 1.5
    num = math.gamma(1 + beta) * math.sin(math.pi * beta / 2)
    den = math.gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2))
    sigma = (num / den) ** (1 / beta)

    if rng is None:
        u = np.random.randn(dim) * sigma
        v = np.random.randn(dim)
    else:
        u = rng.standard_normal(dim) * sigma
        v = rng.standard_normal(dim)
    return u / (np.abs(v) ** (1 / beta))
