"""Secuencia de Halton (1D, base configurable).

Usada por el WO en la fase reproduccion para redistribuir machos de forma
quasi-uniforme dentro del dominio del problema.
"""

import math


def halton(index, base):
    """Devuelve el ``index``-esimo elemento de la secuencia de Halton en la ``base`` dada."""
    result = 0.0
    factor = 1.0 / base
    value = int(index)
    while value > 0:
        result += factor * (value % base)
        value = math.floor(value / base)
        factor = factor / base
    return result
