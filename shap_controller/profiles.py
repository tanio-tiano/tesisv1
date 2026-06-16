"""Configuracion UNICA del controlador (regimen FES).

El setup experimental exige **una sola configuracion de parametros** para el
algoritmo propuesto, valida en los 4 presupuestos (5e3..5e6) **sin re-tuning por
MaxFES ni por problema**. Por eso aqui NO hay perfiles intercambiables ni
mecanismo de override: existe una unica ``DEFAULT_CONTROLLER`` cuyos parametros
temporales se expresan como **fracciones de MaxFES** y se resuelven a valores
absolutos en FES via ``ControllerProfile.resolve(max_fes)``.

Regla estadistica de los parametros:
- **Ventana de estancamiento = 10% de MaxFES** (90% de confianza / 10% de error,
  alpha=0.10). Decision explicita; se mantiene distinta del resto.
- **El resto de las acciones = 5% de MaxFES** (95% de confianza / 5% de error,
  alpha=0.05): ``guard_window``, ``action_cooldown``, ``effective_cooldown`` y
  ``shap_budget``. El limite tardio ``late = 95% de MaxFES`` (nivel de confianza
  1-alpha).

Si en algun momento hiciera falta cambiar la configuracion, se edita
directamente ``DEFAULT_CONTROLLER`` / las constantes de abajo (no hay flags ni
perfiles que la alteren en tiempo de ejecucion).
"""

from dataclasses import dataclass


# --- Reglas estadisticas que gobiernan los parametros del controlador. ---
# Ventana de estancamiento: 90% de confianza / 10% de error (alpha=0.10).
# Se mantiene distinta del resto por decision explicita.
STAGNATION_WINDOW_FRACTION = 0.10
# El RESTO de las acciones del controlador sigue 95% de confianza / 5% de error:
ERROR_MARGIN_FRACTION = 0.05   # margen de error (alpha=0.05) -> ventanas y topes
CONFIDENCE_FRACTION = 0.95     # nivel de confianza (1-alpha) -> limite tardio

# --- Bifurcacion de la accion segun la explicabilidad (reunion guia 2026-05-19). ---
# Una sola accion (reinicializar el agente estancado) que se bifurca:
#   - share de la feature dominante >= CONTRIBUTION_THRESHOLD -> Rama B (reinit guiado:
#     un paso WO con la senal dominante amplificada).
#   - en otro caso                                            -> Rama A (reinit aleatorio).
# 0.90 era el inicial del guia (90% de importancia). Analisis de 8205 intervenciones
# (CEC 4 budgets + TMLAP dura): con 0.90 solo el 44% va a Rama B (guiado, 57% efectivo)
# y el 56% cae en Rama A (random, 29% efectivo). Bajar a 0.50 hace que ~75% de las
# intervenciones usen la senal SHAP dominante (cualquier mayoria simple), aprovechando
# que guided es ~2x mas efectivo que random.
CONTRIBUTION_THRESHOLD = 0.50
# Factor de amplificacion de la senal dominante en la Rama B (tunable).
AMPLIFICATION_FACTOR = 2.0

# --- Constantes fijas (no estadisticas) de la configuracion unica. ---
SHAPLEY_STEPS = 3                   # pasos de simulacion por coalicion (costo SHAP = 2^6 * steps)
RESCUE_SCALE = 1.0                  # escala de las acciones de rescate
NEUTRAL_COOLDOWN_MULTIPLIER = 1.5   # cooldown extra si la intervencion fue neutral
REJECTED_COOLDOWN_MULTIPLIER = 2.5  # cooldown extra si la intervencion fue rechazada


@dataclass(frozen=True)
class ResolvedProfile:
    """Configuracion con valores absolutos en **FES**, lista para el controlador."""

    name: str
    stagnation_window: int      # FES sin mejora (por agente) para declarar estancamiento
    action_cooldown: int        # FES minimos entre intervenciones sobre el MISMO agente
    effective_cooldown: int     # FES minimos tras una intervencion efectiva (global)
    guard_window: int           # no intervenir mientras total_fes < guard_window
    late_fes: int               # no intervenir cuando total_fes > late_fes
    shap_budget: int            # tope de FES totales gastables en explicaciones SHAP
    late_fraction: float        # late_fes / MaxFES (se guarda para telemetria)


@dataclass(frozen=True)
class ControllerProfile:
    """Configuracion parametrizada en fracciones de **MaxFES**."""

    name: str
    stagnation_window_fraction: float = STAGNATION_WINDOW_FRACTION
    action_cooldown_fraction: float = ERROR_MARGIN_FRACTION
    effective_cooldown_fraction: float = ERROR_MARGIN_FRACTION
    guard_window_fraction: float = ERROR_MARGIN_FRACTION
    late_fraction: float = CONFIDENCE_FRACTION
    shap_budget_fraction: float = ERROR_MARGIN_FRACTION

    def resolve(self, max_fes):
        """Convierte fracciones a enteros absolutos en FES para un ``max_fes`` dado."""
        max_fes = int(max_fes)
        if max_fes <= 0:
            raise ValueError(f"max_fes debe ser > 0, recibido: {max_fes}")

        def _round_to_int(fraction, minimum=1):
            return max(minimum, int(round(float(fraction) * max_fes)))

        return ResolvedProfile(
            name=self.name,
            stagnation_window=_round_to_int(self.stagnation_window_fraction, minimum=1),
            action_cooldown=_round_to_int(self.action_cooldown_fraction, minimum=1),
            effective_cooldown=_round_to_int(self.effective_cooldown_fraction, minimum=1),
            guard_window=_round_to_int(self.guard_window_fraction, minimum=0),
            late_fes=int(round(float(self.late_fraction) * max_fes)),
            shap_budget=_round_to_int(self.shap_budget_fraction, minimum=1),
            late_fraction=float(self.late_fraction),
        )


# Unica configuracion del controlador. No hay perfiles alternativos ni overrides.
# Regla estadistica:
#   - estancamiento     -> 10% de MaxFES (90% conf. / 10% error)
#   - guard / cooldowns / shap_budget -> 5% de MaxFES (95% conf. / 5% error)
#   - late              -> 95% de MaxFES (nivel de confianza 1-alpha)
DEFAULT_CONTROLLER = ControllerProfile(
    name="default",
    stagnation_window_fraction=STAGNATION_WINDOW_FRACTION,  # 10% (90% conf / 10% error)
    action_cooldown_fraction=ERROR_MARGIN_FRACTION,         # 5%
    effective_cooldown_fraction=ERROR_MARGIN_FRACTION,      # 5%
    guard_window_fraction=ERROR_MARGIN_FRACTION,            # 5%
    late_fraction=CONFIDENCE_FRACTION,                      # 95%
    shap_budget_fraction=ERROR_MARGIN_FRACTION,             # 5%
)
