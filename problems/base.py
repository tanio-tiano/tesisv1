"""Contrato comun de un problema usable por los runners de WO.

El protocolo describe lo minimo que un problema debe exponer para que el
runner pueda ejecutar el WO (base o SHAP) sin saber nada del dominio.

Atributos requeridos:
- ``name`` (str): identificador legible (aparece en CSVs/reportes).
- ``family`` (str): familia logica ("cec2022", "tmlap", etc.).
- ``dim`` (int): dimensionalidad del vector de decision.
- ``lb`` (float | ndarray): cota inferior (escalar o por dimension).
- ``ub`` (float | ndarray): cota superior.
- ``optimum`` (float | None): valor del optimo conocido (si se sabe).

Metodos requeridos:
- ``evaluate(x: ndarray) -> float``
- ``initial_population(n_agents, rng, **kwargs) -> ndarray``
- ``make_value_function_for_shapley(state, positions, max_iter, steps, rng)
   -> Callable[[dict], float]``  (devuelve una closure usable como
   ``value_function`` en ``SHAPFitnessController.explain_fitness``).

El protocolo se valida via ``runtime_checkable`` para que un duck-typed
problema pueda usarse sin heredar formalmente.
"""

from typing import Callable, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class WOProblem(Protocol):
    name: str
    family: str
    dim: int
    lb: object
    ub: object
    optimum: object  # float | None

    def evaluate(self, x: np.ndarray) -> float: ...

    def initial_population(
        self, n_agents: int, rng: np.random.Generator | None, **kwargs
    ) -> np.ndarray: ...

    def make_value_function_for_shapley(
        self,
        state: dict,
        positions: np.ndarray,
        max_iter: int,
        steps: int,
        rng: np.random.Generator | None,
    ) -> Callable[[dict], float]: ...
