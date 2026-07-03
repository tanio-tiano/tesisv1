"""Controlador SHAP on-line para metaheuristicas poblacionales.

Importacion tipica:

    from shap_controller import (
        SHAPFitnessController,
        improvement_threshold,
        FEATURE_COLUMNS,
        FEATURE_BASELINE_DEFAULTS,
        dispatch_rescue_single,
        restart_single_agent,
        reinject_single_agent,
    )

El controlador tiene una configuracion UNICA (``DEFAULT_CONTROLLER`` en
profiles.py); no hay perfiles intercambiables ni mecanismo de override.
"""

from .controller import SHAPFitnessController, improvement_threshold
from .features import FEATURE_BASELINE_DEFAULTS, FEATURE_COLUMNS
from .profiles import DEFAULT_CONTROLLER, ControllerProfile, ResolvedProfile
from .actions import (
    REINIT_GUIDED,
    REINIT_RANDOM,
    dispatch_rescue_single,
    reinit_guided_agent,
    reinit_random_agent,
)

__all__ = [
    "ControllerProfile",
    "DEFAULT_CONTROLLER",
    "FEATURE_BASELINE_DEFAULTS",
    "FEATURE_COLUMNS",
    "REINIT_GUIDED",
    "REINIT_RANDOM",
    "ResolvedProfile",
    "SHAPFitnessController",
    "dispatch_rescue_single",
    "reinit_guided_agent",
    "reinit_random_agent",
    "improvement_threshold",
]
