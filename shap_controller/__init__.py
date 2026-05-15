"""Controlador SHAP on-line para metaheuristicas poblacionales.

Importacion tipica:

    from shap_controller import (
        SHAPFitnessController,
        get_controller_profile,
        improvement_threshold,
        FEATURE_COLUMNS,
        FEATURE_BASELINE_DEFAULTS,
        PARTIAL_RESTART_MODES,
        RANDOM_REINJECTION_MODES,
        apply_partial_restart,
        apply_random_reinjection,
        dispatch_rescue,
    )
"""

from .controller import SHAPFitnessController, improvement_threshold
from .features import FEATURE_BASELINE_DEFAULTS, FEATURE_COLUMNS
from .profiles import ControllerProfile, PROFILE_DEFAULTS, get_controller_profile
from .actions import (
    PARTIAL_RESTART_MODES,
    RANDOM_REINJECTION_MODES,
    apply_partial_restart,
    apply_random_reinjection,
    dispatch_rescue,
)

__all__ = [
    "ControllerProfile",
    "FEATURE_BASELINE_DEFAULTS",
    "FEATURE_COLUMNS",
    "PARTIAL_RESTART_MODES",
    "PROFILE_DEFAULTS",
    "RANDOM_REINJECTION_MODES",
    "SHAPFitnessController",
    "apply_partial_restart",
    "apply_random_reinjection",
    "dispatch_rescue",
    "get_controller_profile",
    "improvement_threshold",
]
