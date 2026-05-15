"""Perfiles del controlador (soft / medium / hard).

Cada perfil define las ventanas de estancamiento, cooldowns, guard window,
late_fraction y maximo de intervenciones permitidas. Valores alineados con
la Tabla 4 del informe.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ControllerProfile:
    name: str
    stagnation_window: int = 75
    action_cooldown: int = 90
    effective_cooldown: int = 120
    guard_window: int = 180
    late_fraction: float = 0.84
    max_interventions: int = 6
    recent_improvement_window: int = 75
    max_recent_improvement_ratio: float = 1e-5


PROFILE_DEFAULTS = {
    "soft": ControllerProfile(
        name="soft",
        stagnation_window=75,
        action_cooldown=90,
        effective_cooldown=120,
        guard_window=180,
        late_fraction=0.84,
        max_interventions=6,
        recent_improvement_window=75,
        max_recent_improvement_ratio=1e-5,
    ),
    "medium": ControllerProfile(
        name="medium",
        stagnation_window=50,
        action_cooldown=32,
        effective_cooldown=60,
        guard_window=130,
        late_fraction=0.92,
        max_interventions=8,
        recent_improvement_window=50,
        max_recent_improvement_ratio=2e-5,
    ),
    "hard": ControllerProfile(
        name="hard",
        stagnation_window=30,
        action_cooldown=20,
        effective_cooldown=40,
        guard_window=90,
        late_fraction=0.95,
        max_interventions=10,
        recent_improvement_window=30,
        max_recent_improvement_ratio=5e-5,
    ),
}


def get_controller_profile(
    name,
    stagnation_window=None,
    action_cooldown=None,
    effective_cooldown=None,
    guard_window=None,
    late_fraction=None,
    max_interventions=None,
    recent_improvement_window=None,
    max_recent_improvement_ratio=None,
):
    """Devuelve un ``ControllerProfile`` con overrides opcionales."""
    if name not in PROFILE_DEFAULTS:
        valid = ", ".join(sorted(PROFILE_DEFAULTS))
        raise ValueError(f"Perfil desconocido: {name}. Perfiles validos: {valid}")

    base = PROFILE_DEFAULTS[name]
    return ControllerProfile(
        name=name,
        stagnation_window=(
            int(stagnation_window)
            if stagnation_window is not None
            else base.stagnation_window
        ),
        action_cooldown=(
            int(action_cooldown) if action_cooldown is not None else base.action_cooldown
        ),
        effective_cooldown=(
            int(effective_cooldown)
            if effective_cooldown is not None
            else base.effective_cooldown
        ),
        guard_window=int(guard_window) if guard_window is not None else base.guard_window,
        late_fraction=(
            float(late_fraction) if late_fraction is not None else base.late_fraction
        ),
        max_interventions=(
            int(max_interventions)
            if max_interventions is not None
            else base.max_interventions
        ),
        recent_improvement_window=(
            int(recent_improvement_window)
            if recent_improvement_window is not None
            else base.recent_improvement_window
        ),
        max_recent_improvement_ratio=(
            float(max_recent_improvement_ratio)
            if max_recent_improvement_ratio is not None
            else base.max_recent_improvement_ratio
        ),
    )
