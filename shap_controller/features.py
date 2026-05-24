"""Features observados por el controlador y sus baselines neutros.

El set son las 6 senales producidas por ``wo_core.walrus.iteration_signals``
(Eq. 4-8, 11 del paper Walrus Optimizer Han et al. 2024, seccion 3.2.2):

  alpha         = 1 - t/T                                  (Eq. 5)
  beta          = 1 - 1 / (1 + exp((T/2 - t)/T * 10))      (Eq. 11)
  A             = 2 * alpha                                (Eq. 6)
  R             = 2 * r1 - 1   con r1 ~ U(0, 1)            (Eq. 7)
  Danger_signal = A * R                                    (Eq. 4)
  Safety_signal = r2           con r2 ~ U(0, 1)            (Eq. 8)

R y Danger_signal son SIGNADOS (R en [-1, 1], Danger en [-2*alpha, 2*alpha]),
no valores absolutos. La convencion signada coincide con el ``WO.m`` oficial.

El controlador SHAP atribuye el fitness predicho como funcion de estos 6
features. Para el calculo de Shapley se enumeran 2^6 = 64 coaliciones.
Si el historial esta vacio al momento de calcular Shapley, se usa el
baseline neutro de cada feature; de otro modo se usa la mediana del
historial.
"""

import numpy as np


FEATURE_COLUMNS = [
    "alpha",
    "beta",
    "A",
    "R",
    "danger_signal",
    "safety_signal",
]

# Baselines neutros (usados cuando el historial todavia esta vacio).
# Las features del paper Han 2024 son SIGNADAS (vease MATLAB oficial):
# - alpha: 0.5 -> punto medio del decaimiento lineal.
# - beta : 0.5 -> punto medio del sigmoide.
# - A    : 1.0 -> exactamente 2 * alpha_baseline = 2 * 0.5.
# - R    : 0.0 -> esperanza E[2 * r1 - 1] = 0 con r1 ~ U(0,1).
# - danger_signal: 0.0 -> A * R con R_baseline = 0.
# - safety_signal: 0.5 -> esperanza de r2 ~ U(0,1).
FEATURE_BASELINE_DEFAULTS = {
    "alpha": 0.5,
    "beta": 0.5,
    "A": 1.0,
    "R": 0.0,
    "danger_signal": 0.0,
    "safety_signal": 0.5,
}
