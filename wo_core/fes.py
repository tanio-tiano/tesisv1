"""Contador de evaluaciones de la funcion objetivo (FES) para el regimen MaxFES.

Bajo el setup experimental adoptado, el criterio de parada es **MaxFES** (numero
maximo de evaluaciones de la funcion objetivo) y el FES es la moneda unica de
presupuesto. Este modulo provee la contabilidad de ese presupuesto:

- ``FESBudget``: contador unico (``total``) + desglose por categoria
  (``search`` / ``shap`` / ``intervention`` / ``init``). Permite reportar en la
  tesis cuanto presupuesto consume cada componente (p. ej. el costo de SHAP).
- ``counting_objective``: envuelve una funcion de evaluacion (``problem.evaluate``)
  para que cada llamada gaste 1 FES en el bucket indicado. Como TODAS las
  evaluaciones del WO pasan por ``problem.evaluate``, envolverla captura el gasto
  de forma automatica (busqueda, simulacion SHAP, candidatos de intervencion).

La parada **exacta** en MaxFES se logra evaluando la poblacion agente por agente
y cortando cuando ``budget.remaining() == 0`` (ver
``wo_core.walrus.evaluate_and_update_leaders`` con ``budget=...``).
"""

from __future__ import annotations


class FESBudget:
    """Presupuesto de evaluaciones (FES) con desglose por categoria.

    ``total`` es la suma de los buckets y es lo que se compara contra ``max_fes``.
    Invariante: ``total == search + shap + intervention + init``.
    """

    BUCKETS = ("search", "shap", "intervention", "init")

    def __init__(self, max_fes):
        self.max_fes = int(max_fes)
        if self.max_fes <= 0:
            raise ValueError(f"max_fes debe ser > 0, recibido: {self.max_fes}")
        self.total = 0
        self.search = 0
        self.shap = 0
        self.intervention = 0
        self.init = 0

    def spend(self, bucket, n=1):
        """Gasta ``n`` evaluaciones en ``bucket`` (y en ``total``)."""
        if bucket not in self.BUCKETS:
            raise ValueError(f"Bucket desconocido: {bucket!r}. Validos: {self.BUCKETS}.")
        n = int(n)
        self.total += n
        setattr(self, bucket, getattr(self, bucket) + n)

    def remaining(self):
        return max(0, self.max_fes - self.total)

    def exhausted(self):
        return self.total >= self.max_fes

    def can_afford(self, n):
        """True si quedan al menos ``n`` evaluaciones de presupuesto."""
        return self.remaining() >= int(n)

    def as_dict(self):
        """Telemetria lista para una fila de ``summary.csv``."""
        return {
            "max_fes": int(self.max_fes),
            "fes_total": int(self.total),
            "fes_search": int(self.search),
            "fes_shap": int(self.shap),
            "fes_intervention": int(self.intervention),
            "fes_init": int(self.init),
        }

    def __repr__(self):
        return (
            f"FESBudget(max_fes={self.max_fes}, total={self.total}, "
            f"search={self.search}, shap={self.shap}, "
            f"intervention={self.intervention}, init={self.init})"
        )


def counting_objective(evaluate, budget, bucket):
    """Envuelve ``evaluate`` para que cada llamada gaste 1 FES en ``bucket``.

    Devuelve una funcion ``np.ndarray -> float`` con la misma firma que
    ``problem.evaluate``, apta para pasarse como ``objective`` a las primitivas
    del WO. El gasto ocurre *antes* de evaluar, de modo que el contador refleja
    la intencion de evaluar aunque ``evaluate`` lance.
    """

    def wrapped(x):
        budget.spend(bucket, 1)
        return float(evaluate(x))

    return wrapped
