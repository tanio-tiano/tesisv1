from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


MODULE_DIR = Path(__file__).resolve().parent
INPUT_DATA_DIR = MODULE_DIR / "input_data"
HELPERS_MODULE_PATH = MODULE_DIR.parent / "cec2022_official" / "CEC2022.py"

BASIC_FUNCTION_IDS = tuple(range(1, 11))
BIAS_OFFSETS = {
    1: 100.0,
    2: 1100.0,
    3: 700.0,
    4: 1900.0,
    5: 1700.0,
    6: 1600.0,
    7: 2100.0,
    8: 2200.0,
    9: 2400.0,
    10: 2500.0,
}

# CF1-CF10: Basic transformation functions
# CF11-CF20: Bias transformation functions (same base function + offset)
FUNCTION_BIASES = {
    **{function_id: 0.0 for function_id in BASIC_FUNCTION_IDS},
    **{function_id + 10: BIAS_OFFSETS[function_id] for function_id in BASIC_FUNCTION_IDS},
}


def _load_helpers_module():
    spec = importlib.util.spec_from_file_location(
        "_official_cec2021_helpers",
        HELPERS_MODULE_PATH,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"No se pudo cargar CEC2022.py desde {HELPERS_MODULE_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HELPERS = _load_helpers_module()


def _hybrid_partition(nx: int, proportions: list[float]) -> np.ndarray:
    counts = np.zeros(len(proportions), dtype=int)
    used = 0
    for i in range(1, len(proportions)):
        counts[i] = int(np.ceil(proportions[i] * nx))
        used += counts[i]
    counts[0] = nx - used
    return counts


def _matrix_block(matrix: np.ndarray, block_index: int, nx: int) -> np.ndarray:
    start = block_index * nx
    end = start + nx
    return matrix[start:end, :nx]


def bi_rastrigin_func(x, nx, Os, Mr, s_flag, r_flag):
    mu0 = 2.5
    d = 1.0
    s = 1.0 - 1.0 / (2.0 * np.sqrt(nx + 20.0) - 8.2)
    mu1 = -np.sqrt((mu0 * mu0 - d) / s)

    if s_flag == 1:
        y = np.asarray(HELPERS.shiftfunc(x, nx, Os), dtype=float)
    else:
        y = np.asarray(x, dtype=float).copy()

    y = y * (10.0 / 100.0)
    tmpx = 2.0 * y
    os_local = np.asarray(Os[:nx], dtype=float)
    tmpx = np.where(os_local < 0.0, -tmpx, tmpx)
    z = tmpx.copy()
    tmpx = tmpx + mu0

    tmp1 = np.sum((tmpx - mu0) ** 2)
    tmp2 = np.sum((tmpx - mu1) ** 2)
    tmp2 = tmp2 * s + d * nx

    if r_flag == 1:
        y_rot = np.asarray(HELPERS.rotatefunc(z, nx, Mr), dtype=float)
        cosine_sum = np.sum(np.cos(2.0 * np.pi * y_rot))
    else:
        cosine_sum = np.sum(np.cos(2.0 * np.pi * z))

    return float(min(tmp1, tmp2) + 10.0 * (nx - cosine_sum))


def hf01(x, nx, Os, Mr, S, s_flag, r_flag):
    counts = _hybrid_partition(nx, [0.3, 0.3, 0.4])
    starts = np.cumsum(np.concatenate(([0], counts[:-1])))

    z = np.asarray(HELPERS.sr_func(x, nx, Os, Mr, 1.0, s_flag, r_flag), dtype=float)
    y = z[np.asarray(S[:nx], dtype=int) - 1]

    fit = [
        HELPERS.schwefel_func(y[starts[0]:starts[0] + counts[0]], counts[0], Os, Mr, 0, 0),
        HELPERS.rastrigin_func(y[starts[1]:starts[1] + counts[1]], counts[1], Os, Mr, 0, 0),
        HELPERS.ellips_func(y[starts[2]:starts[2] + counts[2]], counts[2], Os, Mr, 0, 0),
    ]
    return float(sum(fit))


def hf05(x, nx, Os, Mr, S, s_flag, r_flag):
    if nx == 5:
        counts = np.asarray([1, 1, 1, 1, 1], dtype=int)
    else:
        counts = _hybrid_partition(nx, [0.1, 0.2, 0.2, 0.2, 0.3])
    starts = np.cumsum(np.concatenate(([0], counts[:-1])))

    z = np.asarray(HELPERS.sr_func(x, nx, Os, Mr, 1.0, s_flag, r_flag), dtype=float)
    y = z[np.asarray(S[:nx], dtype=int) - 1]

    fit = [
        HELPERS.escaffer6_func(y[starts[0]:starts[0] + counts[0]], counts[0], Os, Mr, 0, 0),
        HELPERS.hgbat_func(y[starts[1]:starts[1] + counts[1]], counts[1], Os, Mr, 0, 0),
        HELPERS.rosenbrock_func(y[starts[2]:starts[2] + counts[2]], counts[2], Os, Mr, 0, 0),
        HELPERS.schwefel_func(y[starts[3]:starts[3] + counts[3]], counts[3], Os, Mr, 0, 0),
        HELPERS.ellips_func(y[starts[4]:starts[4] + counts[4]], counts[4], Os, Mr, 0, 0),
    ]
    return float(sum(fit))


def hf06(x, nx, Os, Mr, S, s_flag, r_flag):
    if nx == 5:
        counts = np.asarray([1, 1, 1, 2], dtype=int)
    else:
        counts = _hybrid_partition(nx, [0.2, 0.2, 0.3, 0.3])
    starts = np.cumsum(np.concatenate(([0], counts[:-1])))

    z = np.asarray(HELPERS.sr_func(x, nx, Os, Mr, 1.0, s_flag, r_flag), dtype=float)
    y = z[np.asarray(S[:nx], dtype=int) - 1]

    fit = [
        HELPERS.escaffer6_func(y[starts[0]:starts[0] + counts[0]], counts[0], Os, Mr, 0, 0),
        HELPERS.hgbat_func(y[starts[1]:starts[1] + counts[1]], counts[1], Os, Mr, 0, 0),
        HELPERS.rosenbrock_func(y[starts[2]:starts[2] + counts[2]], counts[2], Os, Mr, 0, 0),
        HELPERS.schwefel_func(y[starts[3]:starts[3] + counts[3]], counts[3], Os, Mr, 0, 0),
    ]
    return float(sum(fit))


def cf02(x, nx, Os, Mr, r_flag):
    delta = [10, 20, 30]
    bias = [0, 0, 0]
    fit = [None] * 3

    fit[0] = HELPERS.rastrigin_func(x, nx, Os[0:nx], _matrix_block(Mr, 0, nx), 1, r_flag)
    fit[1] = HELPERS.griewank_func(x, nx, Os[nx:2 * nx], _matrix_block(Mr, 1, nx), 1, r_flag)
    fit[1] = 1000.0 * fit[1] / 100.0
    fit[2] = HELPERS.schwefel_func(x, nx, Os[2 * nx:3 * nx], _matrix_block(Mr, 2, nx), 1, r_flag)

    return float(HELPERS.cf_cal(x, nx, Os, delta, bias, fit, 3))


def cf04(x, nx, Os, Mr, r_flag):
    delta = [10, 20, 30, 40]
    bias = [0, 0, 0, 0]
    fit = [None] * 4

    fit[0] = HELPERS.ackley_func(x, nx, Os[0:nx], _matrix_block(Mr, 0, nx), 1, r_flag)
    fit[0] = 1000.0 * fit[0] / 100.0
    fit[1] = HELPERS.ellips_func(x, nx, Os[nx:2 * nx], _matrix_block(Mr, 1, nx), 1, r_flag)
    fit[1] = 10000.0 * fit[1] / 1e10
    fit[2] = HELPERS.griewank_func(x, nx, Os[2 * nx:3 * nx], _matrix_block(Mr, 2, nx), 1, r_flag)
    fit[2] = 1000.0 * fit[2] / 100.0
    fit[3] = HELPERS.rastrigin_func(x, nx, Os[3 * nx:4 * nx], _matrix_block(Mr, 3, nx), 1, r_flag)

    return float(HELPERS.cf_cal(x, nx, Os, delta, bias, fit, 4))


def cf05(x, nx, Os, Mr, r_flag):
    delta = [10, 20, 30, 40, 50]
    bias = [0, 0, 0, 0, 0]
    fit = [None] * 5

    fit[0] = HELPERS.rastrigin_func(x, nx, Os[0:nx], _matrix_block(Mr, 0, nx), 1, r_flag)
    fit[0] = 10000.0 * fit[0] / 1e3
    fit[1] = HELPERS.happycat_func(x, nx, Os[nx:2 * nx], _matrix_block(Mr, 1, nx), 1, r_flag)
    fit[1] = 1000.0 * fit[1] / 1e3
    fit[2] = HELPERS.ackley_func(x, nx, Os[2 * nx:3 * nx], _matrix_block(Mr, 2, nx), 1, r_flag)
    fit[2] = 1000.0 * fit[2] / 100.0
    fit[3] = HELPERS.discus_func(x, nx, Os[3 * nx:4 * nx], _matrix_block(Mr, 3, nx), 1, r_flag)
    fit[3] = 10000.0 * fit[3] / 1e10
    fit[4] = HELPERS.rosenbrock_func(x, nx, Os[4 * nx:5 * nx], _matrix_block(Mr, 4, nx), 1, r_flag)

    return float(HELPERS.cf_cal(x, nx, Os, delta, bias, fit, 5))


class OfficialCEC2021Problem:
    """CEC 2021 evaluator using official data files and direct Python translation of the official formulas."""

    def __init__(self, function_id: int, dim: int = 10):
        self.function_id = int(function_id)
        self.base_function_id = ((self.function_id - 1) % 10) + 1
        self.dim = int(dim)
        self.lb = -100.0
        self.ub = 100.0
        self.f_global = FUNCTION_BIASES[self.function_id]

        self._validate()
        self._matrix = self._load_matrix()
        self._shift = self._load_shift()
        self._shuffle = self._load_shuffle()

    def _validate(self):
        if self.function_id not in FUNCTION_BIASES:
            raise ValueError("La funcion CEC 2021 debe estar entre F1 y F20.")
        if self.dim not in (2, 10, 20):
            raise ValueError("CEC 2021 solo define D=2, D=10 y D=20.")
        if self.dim == 2 and self.base_function_id in (5, 6, 7):
            raise ValueError("CEC 2021 no define F5-F7 para D=2.")

    def _load_matrix(self):
        path = INPUT_DATA_DIR / f"M_{self.base_function_id}_D{self.dim}_nr.txt"
        matrix = np.loadtxt(path)
        return np.atleast_2d(np.asarray(matrix, dtype=float))

    def _load_shift(self):
        path = INPUT_DATA_DIR / f"shift_data_{self.base_function_id}_ns.txt"
        raw_shift = np.loadtxt(path)

        if self.base_function_id < 7:
            return np.asarray(raw_shift, dtype=float).reshape(-1)[: self.dim]

        shift = np.atleast_2d(np.asarray(raw_shift, dtype=float))
        return shift[:, : self.dim].reshape(-1)

    def _load_shuffle(self):
        if self.base_function_id < 5 or self.base_function_id > 7:
            return None

        path = INPUT_DATA_DIR / f"shuffle_data_{self.base_function_id}_D{self.dim}.txt"
        return np.asarray(np.loadtxt(path), dtype=int).reshape(-1)

    def evaluate(self, x):
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.size != self.dim:
            raise ValueError(f"Se esperaba un vector de dimension {self.dim}.")

        HELPERS.y = [0.0] * self.dim
        HELPERS.z = [0.0] * self.dim

        if self.base_function_id == 1:
            value = HELPERS.bent_cigar_func(x, self.dim, self._shift, self._matrix, 1, 1)
        elif self.base_function_id == 2:
            value = HELPERS.schwefel_func(x, self.dim, self._shift, self._matrix, 1, 1)
        elif self.base_function_id == 3:
            value = bi_rastrigin_func(x, self.dim, self._shift, self._matrix, 1, 1)
        elif self.base_function_id == 4:
            value = HELPERS.grie_rosen_func(x, self.dim, self._shift, self._matrix, 1, 1)
        elif self.base_function_id == 5:
            value = hf01(x, self.dim, self._shift, self._matrix, self._shuffle, 1, 1)
        elif self.base_function_id == 6:
            value = hf06(x, self.dim, self._shift, self._matrix, self._shuffle, 1, 1)
        elif self.base_function_id == 7:
            value = hf05(x, self.dim, self._shift, self._matrix, self._shuffle, 1, 1)
        elif self.base_function_id == 8:
            value = cf02(x, self.dim, self._shift, self._matrix, 1)
        elif self.base_function_id == 9:
            value = cf04(x, self.dim, self._shift, self._matrix, 1)
        elif self.base_function_id == 10:
            value = cf05(x, self.dim, self._shift, self._matrix, 1)
        else:
            raise ValueError("Funcion CEC 2021 no soportada.")

        return float(value + self.f_global)
