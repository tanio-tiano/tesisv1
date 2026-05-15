"""Features observados por el controlador y sus baselines neutros.

El controlador SHAP explica el fitness predicho como funcion de estos 6 features.
Si el historial esta vacio al momento de calcular Shapley, se usa el baseline
neutro de cada feature; de otro modo se usa la mediana del historial.
"""

import numpy as np


FEATURE_COLUMNS = [
    "alpha",
    "beta",
    "danger_signal",
    "safety_signal",
    "diversity",
    "iteration",
]

FEATURE_BASELINE_DEFAULTS = {
    "alpha": 0.5,
    "beta": 0.5,
    "danger_signal": 0.0,
    "safety_signal": 0.5,
    "diversity": np.nan,
    "iteration": 0.0,
}
