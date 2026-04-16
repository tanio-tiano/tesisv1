from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


MODULE_DIR = Path(__file__).resolve().parent
INPUT_DATA_DIR = MODULE_DIR / "input_data"
OFFICIAL_MODULE_PATH = MODULE_DIR / "CEC2022.py"

FUNCTION_BIASES = {
    1: 300.0,
    2: 400.0,
    3: 600.0,
    4: 800.0,
    5: 900.0,
    6: 1800.0,
    7: 2000.0,
    8: 2200.0,
    9: 2300.0,
    10: 2400.0,
    11: 2600.0,
    12: 2700.0,
}


def _load_official_module():
    spec = importlib.util.spec_from_file_location(
        "_official_cec2022_module",
        OFFICIAL_MODULE_PATH,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"No se pudo cargar CEC2022.py desde {OFFICIAL_MODULE_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_CEC2022 = _load_official_module()


class OfficialCEC2022Problem:
    """CEC 2022 official Python benchmark with cached input data."""

    def __init__(self, function_id: int, dim: int = 10):
        self.function_id = int(function_id)
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
            raise ValueError("La funcion CEC 2022 debe estar entre F1 y F12.")
        if self.dim not in (2, 10, 20):
            raise ValueError("CEC 2022 solo define D=2, D=10 y D=20.")
        if self.dim == 2 and self.function_id in (6, 7, 8):
            raise ValueError("CEC 2022 no define F6-F8 para D=2.")

    def _load_matrix(self):
        path = INPUT_DATA_DIR / f"M_{self.function_id}_D{self.dim}.txt"
        return np.loadtxt(path)

    def _load_shift(self):
        path = INPUT_DATA_DIR / f"shift_data_{self.function_id}.txt"
        raw_shift = np.loadtxt(path)

        if self.function_id < 9:
            return np.asarray(raw_shift[: self.dim], dtype=float)

        shift = np.zeros((9, self.dim), dtype=float)
        for i in range(9):
            for j in range(self.dim):
                shift[i, j] = raw_shift[i, j]
        return shift.reshape(9 * self.dim)

    def _load_shuffle(self):
        if self.function_id < 6 or self.function_id > 8:
            return None

        path = INPUT_DATA_DIR / f"shuffle_data_{self.function_id}_D{self.dim}.txt"
        return np.loadtxt(path)

    def evaluate(self, x):
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.size != self.dim:
            raise ValueError(f"Se esperaba un vector de dimension {self.dim}.")

        # The official functions use module-level buffers y/z.
        _CEC2022.y = [0] * self.dim
        _CEC2022.z = [None] * self.dim

        if self.function_id == 1:
            value = _CEC2022.zakharov_func(x, self.dim, self._shift, self._matrix, 1, 1)
        elif self.function_id == 2:
            value = _CEC2022.rosenbrock_func(x, self.dim, self._shift, self._matrix, 1, 1)
        elif self.function_id == 3:
            value = _CEC2022.schaffer_F7_func(x, self.dim, self._shift, self._matrix, 1, 1)
        elif self.function_id == 4:
            value = _CEC2022.step_rastrigin_func(
                x, self.dim, self._shift, self._matrix, 1, 1
            )
        elif self.function_id == 5:
            value = _CEC2022.levy_func(x, self.dim, self._shift, self._matrix, 1, 1)
        elif self.function_id == 6:
            value = _CEC2022.hf02(
                x, self.dim, self._shift, self._matrix, self._shuffle, 1, 1
            )
        elif self.function_id == 7:
            value = _CEC2022.hf10(
                x, self.dim, self._shift, self._matrix, self._shuffle, 1, 1
            )
        elif self.function_id == 8:
            value = _CEC2022.hf06(
                x, self.dim, self._shift, self._matrix, self._shuffle, 1, 1
            )
        elif self.function_id == 9:
            value = _CEC2022.cf01(x, self.dim, self._shift, self._matrix, 1, 1)
        elif self.function_id == 10:
            value = _CEC2022.cf02(x, self.dim, self._shift, self._matrix, 1, 1)
        elif self.function_id == 11:
            value = _CEC2022.cf06(x, self.dim, self._shift, self._matrix, 1, 1)
        elif self.function_id == 12:
            value = _CEC2022.cf07(x, self.dim, self._shift, self._matrix, 1, 1)
        else:
            raise ValueError("Funcion CEC 2022 no soportada.")

        return float(value + self.f_global)
