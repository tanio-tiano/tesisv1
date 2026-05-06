import re

import numpy as np
from opfunu.cec_based import cec2022


BENCHMARK_MODULES = {
    "cec2022": {"module": cec2022, "year": "2022", "function_ids": tuple(range(1, 13))},
}


def normalize_benchmark(benchmark):
    normalized = str(benchmark).strip().lower()
    if normalized not in BENCHMARK_MODULES:
        valid = ", ".join(sorted(BENCHMARK_MODULES))
        raise ValueError(f"Benchmark no soportado: {benchmark}. Usa: {valid}.")
    return normalized


def benchmark_function_ids(benchmark):
    benchmark = normalize_benchmark(benchmark)
    return list(BENCHMARK_MODULES[benchmark]["function_ids"])


def parse_function_id(function_label, benchmark="cec2022"):
    benchmark = normalize_benchmark(benchmark)
    if isinstance(function_label, str):
        normalized = function_label.strip().upper()
        if not normalized.startswith("F"):
            raise ValueError("La funcion debe tener formato F1, F2, etc.")
        try:
            function_id = int(normalized[1:])
        except ValueError as exc:
            raise ValueError(f"Funcion invalida: {function_label}") from exc
    else:
        function_id = int(function_label)

    if function_id not in BENCHMARK_MODULES[benchmark]["function_ids"]:
        max_id = max(BENCHMARK_MODULES[benchmark]["function_ids"])
        raise ValueError(f"{benchmark.upper()} define funciones F1-F{max_id}.")
    return function_id


def _function_class(benchmark, function_id):
    benchmark = normalize_benchmark(benchmark)
    function_id = parse_function_id(function_id, benchmark)
    config = BENCHMARK_MODULES[benchmark]
    class_name = f"F{function_id}{config['year']}"
    try:
        return getattr(config["module"], class_name)
    except AttributeError as exc:
        raise ValueError(f"No existe {class_name} en opfunu para {benchmark}.") from exc


def _family_for(benchmark, function_id):
    if function_id <= 5:
        return "basic"
    if function_id <= 8:
        return "hybrid"
    return "composition"


def _clean_name(raw_name, function_id):
    text = str(raw_name or f"F{function_id}").strip()
    return re.sub(r"^F\d+:\s*", "", text).strip() or f"F{function_id}"


class OpfunuCECProblem:
    def __init__(self, benchmark, function_id, dim=10):
        self.benchmark = normalize_benchmark(benchmark)
        self.function_id = parse_function_id(function_id, self.benchmark)
        self.dim = int(dim)
        cls = _function_class(self.benchmark, self.function_id)
        self._problem = cls(ndim=self.dim)

        self.lb = self._normalize_bounds(getattr(self._problem, "lb", -100.0))
        self.ub = self._normalize_bounds(getattr(self._problem, "ub", 100.0))
        self.f_global = float(getattr(self._problem, "f_global", np.nan))
        self.name = _clean_name(getattr(self._problem, "name", ""), self.function_id)
        self.family = _family_for(self.benchmark, self.function_id)

    def _normalize_bounds(self, values):
        array = np.asarray(values, dtype=float).reshape(-1)
        if array.size == 1:
            return float(array[0])
        if array.size != self.dim:
            return np.resize(array, self.dim).astype(float)
        return array.astype(float)

    def evaluate(self, x):
        return float(self._problem.evaluate(np.asarray(x, dtype=float)))


def get_function_metadata(benchmark, function_id, dim=10):
    problem = OpfunuCECProblem(benchmark, function_id, dim=dim)
    return {
        "name": problem.name,
        "family": problem.family,
        "optimum": problem.f_global,
    }


def get_benchmark_metadata(benchmark, dim=10):
    benchmark = normalize_benchmark(benchmark)
    return {
        function_id: get_function_metadata(benchmark, function_id, dim=dim)
        for function_id in benchmark_function_ids(benchmark)
    }
