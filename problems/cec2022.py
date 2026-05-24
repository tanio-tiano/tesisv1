"""Adaptador del benchmark CEC 2022 (envuelve ``opfunu.cec_based.cec2022``).

La ``value function`` para Shapley es **por agente** y vive en
``wo_core.agent_sim`` (generica); este adaptador solo aporta ``evaluate`` y la
geometria del problema (``lb/ub/dim``).
"""

import re

import numpy as np
from opfunu.cec_based import cec2022

from wo_core.initialization import uniform_population


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


def get_function_metadata(function_id, dim=10):
    """Devuelve {name, family, optimum} sin crear el objeto opfunu pesado dos veces."""
    p = CECProblem(function_id, dim=dim)
    return {"name": p.name, "family": p.family, "optimum": p.optimum}
