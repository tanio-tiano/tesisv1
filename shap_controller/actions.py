"""Acciones de rescate del controlador SHAP — **una sola accion, bifurcada por SHAP**.

Diseno fijado con el guia (reunion 2026-05-19): cuando un agente se estanca, la
unica accion es **reinicializar su posicion**, con una bifurcacion segun la
explicabilidad:

- **Rama A (`reinit_random`)** — sin contribucion dominante: reinit uniforme en
  ``[lb, ub]`` con la formula clasica ``lb + (ub - lb)*rand`` (descarta la
  posicion). Es el principio clasico de des-estancamiento aleatorio.
- **Rama B (`reinit_guided`)** — con contribucion dominante (>= umbral): un paso
  del movimiento WO **desde la posicion actual** con la **senal dominante
  amplificada** (mutacion guiada por la variable que mas contribuye).

Aceptan ``rng`` opcional; devuelven el nuevo **vector** de posicion del agente.
"""

import numpy as np

from wo_core.walrus import apply_wo_movement_single

from .features import FEATURE_BASELINE_DEFAULTS

REINIT_RANDOM = "reinit_random"
REINIT_GUIDED = "reinit_guided"

# Rangos validos de cada senal, para acotar la amplificacion de la Rama B.
_SIGNAL_RANGES = {
    "alpha": (0.0, 1.0),
    "beta": (0.0, 1.0),
    "A": (0.0, 2.0),
    "R": (-1.0, 1.0),
    "danger_signal": (-2.0, 2.0),
    "safety_signal": (0.0, 1.0),
}


def _draw_normal(rng, scale, size):
    if rng is None:
        return np.random.normal(loc=0.0, scale=scale, size=size)
    return rng.normal(loc=0.0, scale=scale, size=size)


def _draw_uniform(rng, size):
    if rng is None:
        return np.random.rand(*size) if isinstance(size, tuple) else np.random.rand(size)
    return rng.random(size=size)


def reinit_random_agent(positions, index, lb, ub, rng=None):
    """Rama A: reinit uniforme del agente ``index`` en ``[lb, ub]``.

    Es la formula clasica de inicializacion del WO: ``lb + (ub - lb)*rand``.
    Descarta la posicion previa (des-estancamiento puramente aleatorio).
    """
    lb_arr = np.asarray(lb, dtype=float)
    ub_arr = np.asarray(ub, dtype=float)
    dim = positions.shape[1]
    candidate = _draw_uniform(rng, dim) * (ub_arr - lb_arr) + lb_arr
    return np.clip(candidate, lb_arr, ub_arr)


def reinit_guided_agent(
    positions,
    index,
    lb,
    ub,
    signals,
    dominant_feature,
    role_counts,
    best_pos,
    second_pos,
    amplification_factor,
    rng=None,
    dominant_value=0.0,
):
    """Rama B: un paso WO desde la posicion actual con la senal dominante amplificada,
    usando el SIGNO del SHAP para elegir la direccion (Lundberg & Lee 2017).

    En minimizacion, el signo de Shapley es informativo:
    - ``dominant_value < 0`` (senal BENEFICIOSA: baja el fitness) -> amplificar
      en su direccion actual (profundizar el efecto positivo).
    - ``dominant_value > 0`` (senal PERJUDICIAL: sube el fitness) -> invertir
      la direccion (llevar la senal a la zona opuesta del baseline, que el
      SHAP predice como mas favorable).
    - ``dominant_value == 0`` (default backward-compatible) -> direccion +1
      (comportamiento anterior).

    Formula: ``sig' = base + direction * factor * (sig - base)``, donde
    ``direction = -1 if dominant_value > 0 else +1``. Se mantiene el clip al
    rango valido de la senal.
    """
    amplified = {key: float(value) for key, value in signals.items()}
    if dominant_feature in amplified and dominant_feature in FEATURE_BASELINE_DEFAULTS:
        base = FEATURE_BASELINE_DEFAULTS[dominant_feature]
        direction = -1.0 if float(dominant_value) > 0.0 else +1.0
        value = base + direction * float(amplification_factor) * (
            amplified[dominant_feature] - base
        )
        lo, hi = _SIGNAL_RANGES.get(dominant_feature, (-np.inf, np.inf))
        amplified[dominant_feature] = float(np.clip(value, lo, hi))

    n_agents, dim = positions.shape
    male_count, female_count, child_count = role_counts
    gbest_row = np.asarray(best_pos, dtype=float)
    return apply_wo_movement_single(
        positions, index, lb, ub, dim, n_agents,
        male_count, female_count, child_count,
        best_pos, second_pos, gbest_row,
        amplified["alpha"], amplified["beta"], amplified["R"],
        amplified["danger_signal"], amplified["safety_signal"], rng=rng,
    )


def dispatch_rescue_single(
    action,
    positions,
    index,
    lb,
    ub,
    *,
    signals=None,
    dominant_feature=None,
    role_counts=None,
    best_pos=None,
    second_pos=None,
    amplification_factor=2.0,
    rng=None,
    dominant_value=0.0,
):
    """Enruta a la rama A (``reinit_random``) o B (``reinit_guided``).

    Devuelve el nuevo vector de posicion del agente ``index``. La Rama B requiere
    ``signals``, ``dominant_feature``, ``role_counts``, ``best_pos``, ``second_pos``.
    ``dominant_value`` (signed SHAP de la dominante) define la direccion de la
    amplificacion en la Rama B; opcional para backward compat.
    """
    if action == REINIT_GUIDED:
        return reinit_guided_agent(
            positions, index, lb, ub, signals, dominant_feature,
            role_counts, best_pos, second_pos, amplification_factor, rng=rng,
            dominant_value=dominant_value,
        )
    if action == REINIT_RANDOM:
        return reinit_random_agent(positions, index, lb, ub, rng=rng)
    raise ValueError(f"Accion de rescate desconocida: {action!r}.")
