from __future__ import annotations

import math

import numpy as np


def _ufun(x, a, k, m):
    x = np.asarray(x, dtype=float)
    return k * ((x - a) ** m) * (x > a) + k * ((-x - a) ** m) * (x < -a)


def _f1(x):
    return float(np.sum(x**2))


def _f2(x):
    return float(np.sum(np.abs(x)) + np.prod(np.abs(x)))


def _f3(x):
    cumsum = np.cumsum(x)
    return float(np.sum(cumsum**2))


def _f4(x):
    return float(np.max(np.abs(x)))


def _f5(x):
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1.0) ** 2))


def _f6(x):
    return float(np.sum(np.abs(x + 0.5) ** 2))


def _f7(x):
    idx = np.arange(1, x.size + 1, dtype=float)
    return float(np.sum(idx * (x**4)) + np.random.rand())


def _f8(x):
    return float(np.sum(-x * np.sin(np.sqrt(np.abs(x)))))


def _f9(x):
    dim = x.size
    return float(np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x)) + 10.0 * dim)


def _f10(x):
    dim = x.size
    return float(
        -20.0 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / dim))
        - np.exp(np.sum(np.cos(2.0 * np.pi * x)) / dim)
        + 20.0
        + math.e
    )


def _f11(x):
    dim = x.size
    idx = np.arange(1, dim + 1, dtype=float)
    return float(np.sum(x**2) / 4000.0 - np.prod(np.cos(x / np.sqrt(idx))) + 1.0)


def _f12(x):
    dim = x.size
    term1 = 10.0 * np.sin(np.pi * (1.0 + (x[0] + 1.0) / 4.0)) ** 2
    term2 = np.sum(
        (((x[:-1] + 1.0) / 4.0) ** 2)
        * (1.0 + 10.0 * np.sin(np.pi * (1.0 + (x[1:] + 1.0) / 4.0)) ** 2)
    )
    term3 = ((x[-1] + 1.0) / 4.0) ** 2
    penalty = np.sum(_ufun(x, 10.0, 100.0, 4.0))
    return float((np.pi / dim) * (term1 + term2 + term3) + penalty)


def _f13(x):
    penalty = np.sum(_ufun(x, 5.0, 100.0, 4.0))
    term = (
        np.sin(3.0 * np.pi * x[0]) ** 2
        + np.sum((x[:-1] - 1.0) ** 2 * (1.0 + np.sin(3.0 * np.pi * x[1:]) ** 2))
        + (x[-1] - 1.0) ** 2 * (1.0 + np.sin(2.0 * np.pi * x[-1]) ** 2)
    )
    return float(0.1 * term + penalty)


def _f14(x):
    a_s = np.array(
        [
            [-32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32],
            [-32, -32, -32, -32, -32, -16, -16, -16, -16, -16, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32],
        ],
        dtype=float,
    )
    b_s = np.sum((x.reshape(-1, 1) - a_s) ** 6, axis=0)
    return float((1.0 / 500.0 + np.sum(1.0 / (np.arange(1, 26, dtype=float) + b_s))) ** -1)


def _f15(x):
    a_k = np.array([0.1957, 0.1947, 0.1735, 0.16, 0.0844, 0.0627, 0.0456, 0.0342, 0.0323, 0.0235, 0.0246])
    b_k = 1.0 / np.array([0.25, 0.5, 1, 2, 4, 6, 8, 10, 12, 14, 16], dtype=float)
    num = x[0] * (b_k**2 + x[1] * b_k)
    den = b_k**2 + x[2] * b_k + x[3]
    return float(np.sum((a_k - num / den) ** 2))


def _f16(x):
    return float(
        4.0 * x[0] ** 2
        - 2.1 * x[0] ** 4
        + (x[0] ** 6) / 3.0
        + x[0] * x[1]
        - 4.0 * x[1] ** 2
        + 4.0 * x[1] ** 4
    )


def _f17(x):
    return float(
        (x[1] - (x[0] ** 2) * 5.1 / (4.0 * np.pi**2) + 5.0 / np.pi * x[0] - 6.0) ** 2
        + 10.0 * (1.0 - 1.0 / (8.0 * np.pi)) * np.cos(x[0])
        + 10.0
    )


def _f18(x):
    return float(
        (1.0 + (x[0] + x[1] + 1.0) ** 2 * (19.0 - 14.0 * x[0] + 3.0 * x[0] ** 2 - 14.0 * x[1] + 6.0 * x[0] * x[1] + 3.0 * x[1] ** 2))
        * (30.0 + (2.0 * x[0] - 3.0 * x[1]) ** 2 * (18.0 - 32.0 * x[0] + 12.0 * x[0] ** 2 + 48.0 * x[1] - 36.0 * x[0] * x[1] + 27.0 * x[1] ** 2))
    )


def _f19(x):
    a_h = np.array([[3, 10, 30], [0.1, 10, 35], [3, 10, 30], [0.1, 10, 35]], dtype=float)
    c_h = np.array([1, 1.2, 3, 3.2], dtype=float)
    p_h = np.array(
        [
            [0.3689, 0.1170, 0.2673],
            [0.4699, 0.4387, 0.7470],
            [0.1091, 0.8732, 0.5547],
            [0.03815, 0.5743, 0.8828],
        ],
        dtype=float,
    )
    values = -c_h * np.exp(-np.sum(a_h * (x - p_h) ** 2, axis=1))
    return float(np.sum(values))


def _f20(x):
    a_h = np.array(
        [
            [10, 3, 17, 3.5, 1.7, 8],
            [0.05, 10, 17, 0.1, 8, 14],
            [3, 3.5, 1.7, 10, 17, 8],
            [17, 8, 0.05, 10, 0.1, 14],
        ],
        dtype=float,
    )
    c_h = np.array([1, 1.2, 3, 3.2], dtype=float)
    p_h = np.array(
        [
            [0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
            [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
            [0.2348, 0.1415, 0.3522, 0.2883, 0.3047, 0.6650],
            [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381],
        ],
        dtype=float,
    )
    values = -c_h * np.exp(-np.sum(a_h * (x - p_h) ** 2, axis=1))
    return float(np.sum(values))


def _shekel(x, n_terms):
    a_sh = np.array(
        [
            [4, 4, 4, 4],
            [1, 1, 1, 1],
            [8, 8, 8, 8],
            [6, 6, 6, 6],
            [3, 7, 3, 7],
            [2, 9, 2, 9],
            [5, 5, 3, 3],
            [8, 1, 8, 1],
            [6, 2, 6, 2],
            [7, 3.6, 7, 3.6],
        ],
        dtype=float,
    )
    c_sh = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5], dtype=float)
    total = 0.0
    for i in range(n_terms):
        diff = x - a_sh[i]
        total -= 1.0 / (np.dot(diff, diff) + c_sh[i])
    return float(total)


def _f21(x):
    return _shekel(x, 5)


def _f22(x):
    return _shekel(x, 7)


def _f23(x):
    return _shekel(x, 10)


FUNCTION_SPECS = {
    1: {"lb": -100.0, "ub": 100.0, "dim": 30, "func": _f1, "optimum": 0.0},
    2: {"lb": -10.0, "ub": 10.0, "dim": 30, "func": _f2, "optimum": 0.0},
    3: {"lb": -100.0, "ub": 100.0, "dim": 30, "func": _f3, "optimum": 0.0},
    4: {"lb": -100.0, "ub": 100.0, "dim": 30, "func": _f4, "optimum": 0.0},
    5: {"lb": -30.0, "ub": 30.0, "dim": 30, "func": _f5, "optimum": 0.0},
    6: {"lb": -100.0, "ub": 100.0, "dim": 30, "func": _f6, "optimum": 0.0},
    7: {"lb": -1.28, "ub": 1.28, "dim": 30, "func": _f7, "optimum": np.nan},
    8: {"lb": -500.0, "ub": 500.0, "dim": 30, "func": _f8, "optimum": -418.9828872724338 * 30.0},
    9: {"lb": -5.12, "ub": 5.12, "dim": 30, "func": _f9, "optimum": 0.0},
    10: {"lb": -32.0, "ub": 32.0, "dim": 30, "func": _f10, "optimum": 0.0},
    11: {"lb": -600.0, "ub": 600.0, "dim": 30, "func": _f11, "optimum": 0.0},
    12: {"lb": -50.0, "ub": 50.0, "dim": 30, "func": _f12, "optimum": 0.0},
    13: {"lb": -50.0, "ub": 50.0, "dim": 30, "func": _f13, "optimum": 0.0},
    14: {"lb": -65.536, "ub": 65.536, "dim": 2, "func": _f14, "optimum": 0.9980038377944493},
    15: {"lb": -5.0, "ub": 5.0, "dim": 4, "func": _f15, "optimum": 0.00030748610},
    16: {"lb": -5.0, "ub": 5.0, "dim": 2, "func": _f16, "optimum": -1.031628453489877},
    17: {"lb": np.array([-5.0, 0.0]), "ub": np.array([10.0, 15.0]), "dim": 2, "func": _f17, "optimum": 0.39788735772973816},
    18: {"lb": -2.0, "ub": 2.0, "dim": 2, "func": _f18, "optimum": 3.0},
    19: {"lb": 0.0, "ub": 1.0, "dim": 3, "func": _f19, "optimum": -3.86278214782076},
    20: {"lb": 0.0, "ub": 1.0, "dim": 6, "func": _f20, "optimum": -3.322368011415515},
    21: {"lb": 0.0, "ub": 10.0, "dim": 4, "func": _f21, "optimum": -10.15319967905823},
    22: {"lb": 0.0, "ub": 10.0, "dim": 4, "func": _f22, "optimum": -10.402940566818662},
    23: {"lb": 0.0, "ub": 10.0, "dim": 4, "func": _f23, "optimum": -10.536409816692049},
}


class WOPaper23Problem:
    """23-function benchmark used in the original Walrus Optimizer MATLAB package."""

    def __init__(self, function_id: int, dim: int | None = None):
        self.function_id = int(function_id)
        if self.function_id not in FUNCTION_SPECS:
            raise ValueError("La funcion del benchmark del paper WO debe estar entre F1 y F23.")

        spec = FUNCTION_SPECS[self.function_id]
        self.dim = int(spec["dim"])
        if dim is not None and int(dim) != self.dim:
            raise ValueError(
                f"F{self.function_id} usa dimension fija {self.dim} segun Get_Functions_details.m."
            )

        self.lb = spec["lb"]
        self.ub = spec["ub"]
        self.optimum = float(spec["optimum"]) if not np.isnan(spec["optimum"]) else np.nan
        self._func = spec["func"]

    def evaluate(self, x):
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.size != self.dim:
            raise ValueError(f"Se esperaba un vector de dimension {self.dim}.")
        return float(self._func(x))
