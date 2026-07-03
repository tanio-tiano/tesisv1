"""SHAPFitnessController: controlador SHAP exacto **por agente** en regimen FES.

Independiente del problema. El flujo por iteracion del WO es:

1. Se detecta estancamiento **por agente**: el agente i es candidato cuando lleva
   ``>= 10% de MaxFES`` evaluaciones (reloj de FES global) sin mejorar su mejor
   fitness personal. Esto NO usa SHAP (es barato).
2. Solo entonces se aplica SHAP **sobre ese agente**: ``explain_fitness`` calcula
   los 6 valores de Shapley exactos (64 coaliciones) atribuyendo el fitness del
   agente a las 6 senales de control globales del WO, usando una ``value
   function`` que simula SOLO a ese agente (``wo_core.agent_sim``).
3. ``decide`` traduce el SHAP en una accion sobre el agente (reinjeccion /
   reinicio guiado por elite).

Todas las compuertas temporales (guard, cooldowns, late, ventana de
estancamiento, tope de gasto de SHAP) viven en **FES** y se resuelven como
fracciones de MaxFES (ver ``profiles.py``), de modo que una unica configuracion
sirve en los 4 presupuestos del setup.
"""

from math import factorial

import numpy as np
import pandas as pd

from wo_core.agent_sim import shap_cost_estimate

from .actions import REINIT_GUIDED, REINIT_RANDOM
from .features import FEATURE_BASELINE_DEFAULTS, FEATURE_COLUMNS
from .profiles import (
    AMPLIFICATION_FACTOR,
    CONTRIBUTION_THRESHOLD,
    DEFAULT_CONTROLLER,
    NEUTRAL_COOLDOWN_MULTIPLIER,
    REJECTED_COOLDOWN_MULTIPLIER,
    RESCUE_SCALE,
    SHAPLEY_STEPS,
)


def improvement_threshold(fitness, abs_eps=1e-10, rel_eps=1e-4):
    """Tolerancia adaptativa de mejora: max(absoluta, relativa·|fitness|).

    rel_eps=1e-4 (0.01% relativo): ruido numerico microscopico ya NO se cuenta
    como mejora. Antes (1e-6) cualquier oscilacion del orden de 1e-6*|f| pateaba
    `last_improve_fes` y mantenia al agente fuera de la ventana de estancamiento.
    """
    return max(abs_eps, rel_eps * max(abs(float(fitness)), 1.0))


class SHAPFitnessController:
    """Controlador SHAP por agente; explica el fitness desde las 6 senales del WO."""

    def __init__(self, max_fes, n_agents, random_state=1234):
        self.max_fes = int(max_fes)
        self.n_agents = int(n_agents)
        # Configuracion UNICA (sin perfiles ni overrides); ver profiles.py.
        self.profile = DEFAULT_CONTROLLER.resolve(self.max_fes)
        self.shapley_steps = SHAPLEY_STEPS
        self.contribution_threshold = CONTRIBUTION_THRESHOLD
        self.amplification_factor = AMPLIFICATION_FACTOR
        self.shap_cost_estimate = shap_cost_estimate(self.shapley_steps)
        # Tope de explicaciones derivado del presupuesto de SHAP (coherente con
        # shap_budget): a lo sumo cuantas explicaciones caben en shap_budget.
        self.max_interventions = max(
            1, self.profile.shap_budget // max(1, self.shap_cost_estimate)
        )
        self.random_state = int(random_state)
        self.rescue_scale = RESCUE_SCALE
        self.neutral_cooldown_multiplier = NEUTRAL_COOLDOWN_MULTIPLIER
        self.rejected_cooldown_multiplier = REJECTED_COOLDOWN_MULTIPLIER

        self.history = []          # historial de las 6 senales (para baseline Shapley)
        self.events = []
        self.non_events = []
        self.shap_rows = []
        self.intervention_count = 0
        self.last_action_fes = {}          # por agente: FES de su ultima intervencion
        self.last_effective_fes = None     # global: FES de la ultima intervencion efectiva
        self.adaptive_cooldown_until_fes = None
        self.adaptive_cooldown_reason = ""

    # ------------------------------------------------------------------
    # Historial y baseline para Shapley.
    # ------------------------------------------------------------------
    def record_state(self, state, fitness):
        row = {key: float(state[key]) for key in FEATURE_COLUMNS}
        row["fitness"] = float(fitness)
        self.history.append(row)

    def baseline_state(self, state):
        """Baseline para Shapley: mediana del historial (fallback a defaults neutros)."""
        baseline = dict(FEATURE_BASELINE_DEFAULTS)
        history_df = pd.DataFrame(self.history)
        if not history_df.empty:
            for feature in FEATURE_COLUMNS:
                values = pd.to_numeric(history_df.get(feature), errors="coerce")
                values = values.replace([np.inf, -np.inf], np.nan).dropna()
                if not values.empty:
                    baseline[feature] = float(values.median())
        for feature in FEATURE_COLUMNS:
            value = baseline.get(feature, np.nan)
            if not np.isfinite(value):
                baseline[feature] = float(state.get(feature, FEATURE_BASELINE_DEFAULTS[feature]))
        return baseline

    def explain_fitness(self, state, value_function=None):
        """Calcula los 6 valores SHAP exactos enumerando las 64 coaliciones.

        Identico al regimen global: solo cambia la ``value_function`` inyectada,
        que ahora simula UN agente (``wo_core.agent_sim.make_value_function_for_agent``).
        """
        if value_function is None:
            return None

        current_state = {feature: float(state[feature]) for feature in FEATURE_COLUMNS}
        baseline_state = self.baseline_state(state)
        n_features = len(FEATURE_COLUMNS)
        full_mask = (1 << n_features) - 1
        denominator = factorial(n_features)
        values_by_mask = {}

        def coalition_state(mask):
            return {
                feature: (
                    current_state[feature]
                    if mask & (1 << index)
                    else baseline_state[feature]
                )
                for index, feature in enumerate(FEATURE_COLUMNS)
            }

        def coalition_value(mask):
            if mask not in values_by_mask:
                value = float(value_function(coalition_state(mask)))
                if not np.isfinite(value):
                    raise ValueError("Non-finite exact Shapley coalition value")
                values_by_mask[mask] = value
            return values_by_mask[mask]

        shap_values = np.zeros(n_features, dtype=float)
        for feature_index in range(n_features):
            feature_bit = 1 << feature_index
            for mask in range(1 << n_features):
                if mask & feature_bit:
                    continue
                subset_size = int(mask.bit_count())
                weight = (
                    factorial(subset_size)
                    * factorial(n_features - subset_size - 1)
                    / denominator
                )
                shap_values[feature_index] += weight * (
                    coalition_value(mask | feature_bit) - coalition_value(mask)
                )

        expected_value = coalition_value(0)
        prediction = coalition_value(full_mask)
        shap_map = {
            feature: float(value) for feature, value in zip(FEATURE_COLUMNS, shap_values)
        }
        abs_sum = float(np.sum(np.abs(shap_values)))
        positive_pressure = float(np.sum(np.maximum(shap_values, 0.0)))
        dominant_feature = max(shap_map, key=lambda feature: abs(shap_map[feature]))

        return {
            "target": "agent_fitness",
            "method": "exact_wo_shapley_per_agent",
            "expected_fitness": float(expected_value),
            "predicted_fitness": float(prediction),
            "values": shap_map,
            "positive_pressure": positive_pressure,
            "absolute_pressure": abs_sum,
            "dominant_feature": dominant_feature,
            "dominant_value": float(shap_map[dominant_feature]),
            "baseline": baseline_state,
            "n_features": n_features,
            "n_coalitions": int(1 << n_features),
        }

    # ------------------------------------------------------------------
    # Compuertas (todas en FES).
    # ------------------------------------------------------------------
    def should_consider_intervention(self, state, budget):
        """Devuelve (bool, reason) para el agente ``state['agent_index']``."""
        fes = int(budget.total)
        agent = int(state["agent_index"])
        fes_since_improve = int(state["fes_since_improve"])
        diversity_norm = float(state["diversity_norm"])
        window = self.profile.stagnation_window

        # --- Compuertas globales (presupuesto y horizonte). ---
        if self.intervention_count >= self.max_interventions:
            return False, "max_interventions"
        if budget.shap >= self.profile.shap_budget:
            return False, "shap_budget_exhausted"
        if not budget.can_afford(self.shap_cost_estimate):
            return False, "insufficient_budget_for_shap"
        if fes < self.profile.guard_window:
            return False, "guard_window"
        if fes > self.profile.late_fes:
            return False, "late_fraction"
        if (
            self.adaptive_cooldown_until_fes is not None
            and fes < self.adaptive_cooldown_until_fes
        ):
            return False, "adaptive_outcome_cooldown"

        # --- Compuertas por agente. ---
        if fes_since_improve < window:
            return False, "stagnation_window"
        last_a = self.last_action_fes.get(agent)
        if last_a is not None and fes - last_a < self.profile.action_cooldown:
            return False, "action_cooldown"
        if (
            self.last_effective_fes is not None
            and fes - self.last_effective_fes < self.profile.effective_cooldown
        ):
            return False, "effective_cooldown"
        if diversity_norm >= 0.75 and fes_since_improve < 2 * window:
            return False, "weak_diversity_evidence"
        return True, "candidate"

    # ------------------------------------------------------------------
    # Decision por agente (politica que traduce SHAP -> accion).
    # ------------------------------------------------------------------
    def decide(self, state, shap_info):
        """Bifurca la UNICA accion (reinicializar el agente) segun la contribucion
        SHAP **conjunta** (todas las features). Asume que
        ``should_consider_intervention`` ya paso (el runner la chequea ANTES de
        gastar el SHAP); por eso aqui SIEMPRE se interviene. SHAP solo elige rama:

        - share de la feature dominante >= ``contribution_threshold`` -> Rama B
          (``reinit_guided``: un paso WO con la senal dominante amplificada).
        - en otro caso -> Rama A (``reinit_random``: reinit uniforme).
        """
        shap_values = (shap_info or {}).get("values", {})
        abs_values = {f: abs(float(shap_values.get(f, 0.0))) for f in FEATURE_COLUMNS}
        total = sum(abs_values.values())
        if total > 0:
            dominant_feature = max(abs_values, key=abs_values.get)
            dominant_share = abs_values[dominant_feature] / total
            # Valor SIGNED del SHAP de la dominante: en minimizacion, SHAP<0
            # es benefico (la senal baja el fitness) y SHAP>0 es perjudicial.
            # Lo usa reinit_guided_agent para decidir direccion de amplificacion.
            dominant_value = float(shap_values.get(dominant_feature, 0.0))
        else:
            dominant_feature = ""
            dominant_share = 0.0
            dominant_value = 0.0

        if dominant_share >= self.contribution_threshold:
            action = REINIT_GUIDED
            policy_signal = "shap_dominant_contribution"
        else:
            action = REINIT_RANDOM
            policy_signal = "shap_no_dominant_contribution"

        return {
            "intervene": True,
            "reason": f"{self.profile.name}_controller_{policy_signal}",
            "action": action,
            "agent_index": int(state["agent_index"]),
            "shap_info": shap_info,
            "policy_signal": policy_signal,
            "dominant_feature": dominant_feature,
            "dominant_share": float(dominant_share),
            "dominant_value": float(dominant_value),
        }

    # ------------------------------------------------------------------
    # Cooldowns adaptativos (en FES) y registro de outcomes.
    # ------------------------------------------------------------------
    def extend_adaptive_cooldown(self, fes, multiplier, reason):
        if multiplier <= 1.0:
            return
        cooldown = int(round(self.profile.action_cooldown * multiplier))
        until = int(min(self.max_fes - 1, fes + max(cooldown, 0)))
        if (
            self.adaptive_cooldown_until_fes is None
            or until > self.adaptive_cooldown_until_fes
        ):
            self.adaptive_cooldown_until_fes = until
            self.adaptive_cooldown_reason = str(reason)

    def register_decision(
        self,
        fes,
        decision,
        agent_index,
        pre_agent_fitness,
        post_agent_fitness,
        diversity,
        fes_since_improve,
        accepted,
        improved,
        acceptance_reason,
    ):
        """Registra el resultado de una decision (intervencion o no) en telemetria."""
        shap_info = decision.get("shap_info")
        if shap_info is not None:
            shap_row = {
                "fes": int(fes),
                "agent_index": int(agent_index),
                "controller_profile": self.profile.name,
                "target": shap_info.get("target", "agent_fitness"),
                "action": decision["action"],
                "decision_reason": decision["reason"],
                "policy_signal": decision.get("policy_signal", ""),
                "dominant_share": float(decision.get("dominant_share", np.nan)),
                "intervened": bool(decision["intervene"] and accepted),
                "accepted": bool(accepted),
                "improved": bool(improved),
                "acceptance_reason": acceptance_reason,
                "shap_method": shap_info.get("method", ""),
                "shap_n_coalitions": int(shap_info.get("n_coalitions", 0)),
                "expected_fitness": shap_info["expected_fitness"],
                "predicted_fitness": shap_info["predicted_fitness"],
                "dominant_feature": shap_info["dominant_feature"],
                "dominant_value": shap_info["dominant_value"],
                "positive_pressure": shap_info["positive_pressure"],
                "absolute_pressure": shap_info["absolute_pressure"],
            }
            shap_row.update(
                {f"shap_{feature}": value for feature, value in shap_info["values"].items()}
            )
            self.shap_rows.append(shap_row)

        if not decision["intervene"]:
            self.non_events.append(
                {
                    "fes": int(fes),
                    "agent_index": int(agent_index),
                    "controller_profile": self.profile.name,
                    "reason": str(decision.get("reason", "")),
                    "policy_signal": str(decision.get("policy_signal", "")),
                    "shap_was_computed": shap_info is not None,
                    "pre_agent_fitness": float(pre_agent_fitness),
                    "pre_diversity": float(diversity),
                    "fes_since_improve": int(fes_since_improve),
                }
            )
            return

        # Es una intervencion propuesta.
        self.last_action_fes[int(agent_index)] = int(fes)
        if accepted:
            self.intervention_count += 1
            if improved:
                self.last_effective_fes = int(fes)
            else:
                self.extend_adaptive_cooldown(
                    fes, self.neutral_cooldown_multiplier, "neutral"
                )
        else:
            self.extend_adaptive_cooldown(
                fes, self.rejected_cooldown_multiplier, "rejected"
            )

        self.events.append(
            {
                "event_id": len(self.events) + 1,
                "fes": int(fes),
                "agent_index": int(agent_index),
                "controller_profile": self.profile.name,
                "stagnation_window": self.profile.stagnation_window,
                "action_cooldown": self.profile.action_cooldown,
                "effective_cooldown": self.profile.effective_cooldown,
                "guard_window": self.profile.guard_window,
                "late_fes": self.profile.late_fes,
                "shap_budget": self.profile.shap_budget,
                "max_interventions": self.max_interventions,
                "action": decision["action"],
                "policy_signal": decision.get("policy_signal", ""),
                "dominant_share": float(decision.get("dominant_share", np.nan)),
                "accepted": bool(accepted),
                "improved": bool(improved),
                "acceptance_reason": acceptance_reason,
                "pre_agent_fitness": float(pre_agent_fitness),
                "post_agent_fitness": float(post_agent_fitness),
                "pre_diversity": float(diversity),
                "fes_since_improve": int(fes_since_improve),
                "adaptive_cooldown_until_fes": (
                    int(self.adaptive_cooldown_until_fes)
                    if self.adaptive_cooldown_until_fes is not None
                    else np.nan
                ),
                "adaptive_cooldown_reason": self.adaptive_cooldown_reason,
                "shap_dominant_feature": (shap_info or {}).get("dominant_feature", ""),
                "shap_dominant_value": (shap_info or {}).get("dominant_value", np.nan),
            }
        )

    def events_dataframe(self):
        return pd.DataFrame(self.events)

    def non_events_dataframe(self):
        return pd.DataFrame(self.non_events)

    def shap_dataframe(self):
        return pd.DataFrame(self.shap_rows)
