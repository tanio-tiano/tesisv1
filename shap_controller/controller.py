"""SHAPFitnessController: controlador SHAP exacto + politica + cooldowns.

Independiente del problema. Recibe el ``state`` (alpha, beta, danger, safety,
diversity, iteration, stagnation_length, diversity_norm, ...) y devuelve
una decision: intervenir o no, y con que accion/modo/fraccion.

Si se le pasa una ``value_function(coalition_state) -> float``, calcula los
6 valores SHAP exactos (64 coaliciones) sobre el fitness predicho y los usa
para orientar la decision. La construccion de esa value function vive en
``problems/<problema>.py`` porque depende del problema.
"""

from math import factorial

import numpy as np
import pandas as pd

from .features import FEATURE_BASELINE_DEFAULTS, FEATURE_COLUMNS
from .profiles import PROFILE_DEFAULTS


def improvement_threshold(fitness, abs_eps=1e-10, rel_eps=1e-6):
    """Tolerancia adaptativa de mejora: max(absoluta, relativa·|fitness|)."""
    return max(abs_eps, rel_eps * max(abs(float(fitness)), 1.0))


class SHAPFitnessController:
    """Controlador configurable: SHAP explica fitness desde senales internas."""

    def __init__(
        self,
        max_iter,
        random_state=1234,
        profile=None,
        rescue_scale=1.0,
        neutral_cooldown_multiplier=1.0,
        rejected_cooldown_multiplier=1.0,
    ):
        self.max_iter = int(max_iter)
        self.profile = profile or PROFILE_DEFAULTS["soft"]
        self.random_state = int(random_state)
        self.rescue_scale = float(rescue_scale)
        self.neutral_cooldown_multiplier = float(neutral_cooldown_multiplier)
        self.rejected_cooldown_multiplier = float(rejected_cooldown_multiplier)
        self.history = []
        self.events = []
        self.shap_rows = []
        self.last_action_iteration = None
        self.last_effective_iteration = None
        self.adaptive_cooldown_until_iteration = None
        self.adaptive_cooldown_reason = ""
        self.intervention_count = 0

    def record_state(self, state, fitness):
        row = {key: float(state[key]) for key in FEATURE_COLUMNS}
        row["fitness"] = float(fitness)
        self.history.append(row)

    def update_pending_events(self, iteration, current_fitness, diversity, stagnation_length):
        for event in self.events:
            if event["post_status"] != "pending":
                continue
            if iteration < event["post_iteration"]:
                continue

            delta = float(event["pre_fitness"] - current_fitness)
            threshold = improvement_threshold(event["pre_fitness"])
            if delta > threshold:
                status = "improved"
                self.last_effective_iteration = int(event["iteration"])
            elif delta < -threshold:
                status = "worsened"
                self.extend_adaptive_cooldown(
                    iteration,
                    max(self.neutral_cooldown_multiplier, 3.0),
                    "worsened",
                )
            else:
                status = "neutral"
                self.extend_adaptive_cooldown(
                    iteration,
                    self.neutral_cooldown_multiplier,
                    "neutral",
                )

            event["post_status"] = status
            event["post_fitness"] = float(current_fitness)
            event["post_delta_fitness"] = delta
            event["post_diversity"] = float(diversity)
            event["post_stagnation_length"] = int(stagnation_length)
            event["adaptive_cooldown_until_iteration"] = (
                int(self.adaptive_cooldown_until_iteration)
                if self.adaptive_cooldown_until_iteration is not None
                else np.nan
            )
            event["adaptive_cooldown_reason"] = self.adaptive_cooldown_reason

    def extend_adaptive_cooldown(self, iteration, multiplier, reason):
        if multiplier <= 1.0:
            return
        cooldown = int(round(self.profile.action_cooldown * multiplier))
        until = int(min(self.max_iter - 1, iteration + max(cooldown, 0)))
        if (
            self.adaptive_cooldown_until_iteration is None
            or until > self.adaptive_cooldown_until_iteration
        ):
            self.adaptive_cooldown_until_iteration = until
            self.adaptive_cooldown_reason = str(reason)

    def should_consider_intervention(self, state):
        iteration = int(state["iteration"])
        stagnation_length = int(state["stagnation_length"])
        diversity_norm = float(state["diversity_norm"])
        recent_improvement_ratio = float(state.get("recent_improvement_ratio", 0.0))

        if self.intervention_count >= self.profile.max_interventions:
            return False, "max_interventions"
        if iteration < self.profile.guard_window:
            return False, "guard_window"
        if iteration > self.profile.late_fraction * self.max_iter:
            return False, "late_fraction"
        if self.adaptive_cooldown_until_iteration is not None:
            if iteration < self.adaptive_cooldown_until_iteration:
                return False, "adaptive_outcome_cooldown"
        if stagnation_length < self.profile.stagnation_window:
            return False, "stagnation_window"
        if recent_improvement_ratio > self.profile.max_recent_improvement_ratio:
            return False, "recent_improvement_active"
        if self.last_action_iteration is not None:
            if iteration - self.last_action_iteration < self.profile.action_cooldown:
                return False, "action_cooldown"
        if self.last_effective_iteration is not None:
            if iteration - self.last_effective_iteration < self.profile.effective_cooldown:
                return False, "effective_cooldown"
        if diversity_norm >= 0.75 and stagnation_length < 2 * self.profile.stagnation_window:
            return False, "weak_diversity_evidence"
        return True, "candidate"

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
                value = float(state.get(feature, FEATURE_BASELINE_DEFAULTS[feature]))
            baseline[feature] = float(value)

        baseline["iteration"] = float(
            np.clip(round(baseline["iteration"]), 0, max(self.max_iter - 1, 0))
        )
        return baseline

    def explain_fitness(self, state, value_function=None):
        """Calcula los 6 valores SHAP exactos enumerando las 64 coaliciones."""
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
            "target": "fitness",
            "method": "exact_wo_shapley",
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

    def decide(self, state, shap_info=None):
        """Devuelve la decision de intervencion (action, mode, fraction) o None.

        Taxonomia segun Tabla 2 del informe:
        - ``partial_restart`` (15% base): elite_guided / _wide / _aggressive.
        - ``random_reinjection`` (25% base): random / random_aggressive.
        """
        should_intervene, reason = self.should_consider_intervention(state)
        if not should_intervene:
            return {
                "intervene": False,
                "reason": reason,
                "action": "none",
                "mode": "none",
                "fraction": 0.0,
                "shap_info": None,
                "policy_signal": reason,
                "policy_confidence": 0.0,
                "diversity_pressure": 0.0,
                "stagnation_pressure": 0.0,
                "temporal_pressure": 0.0,
            }

        if shap_info is None:
            shap_info = self.explain_fitness(state)
        diversity_norm = float(state["diversity_norm"])
        stagnation_length = int(state["stagnation_length"])
        severity = min(
            1.0,
            max(
                0.0,
                (stagnation_length - self.profile.stagnation_window)
                / max(self.profile.stagnation_window, 1),
            ),
        )

        shap_values = shap_info["values"] if shap_info is not None else {}
        diversity_pressure = max(shap_values.get("diversity", 0.0), 0.0)
        stagnation_pressure = 0.0
        temporal_features = (
            "iteration",
            "alpha",
            "beta",
            "danger_signal",
            "safety_signal",
        )
        temporal_pressure = sum(
            max(shap_values.get(feature, 0.0), 0.0) for feature in temporal_features
        )
        pressure_den = (shap_info or {}).get("absolute_pressure", 0.0)
        pressure_num = (shap_info or {}).get("positive_pressure", 0.0)
        bad_share = pressure_num / pressure_den if pressure_den > 0 else 0.5
        dominant_feature = (shap_info or {}).get("dominant_feature", "")
        positive_group_pressure = diversity_pressure + stagnation_pressure + temporal_pressure

        def pressure_share(value):
            if positive_group_pressure <= 0:
                return 0.0
            return float(value / positive_group_pressure)

        if (
            shap_info is not None
            and diversity_norm >= 0.75
            and stagnation_length < 2 * self.profile.stagnation_window
            and dominant_feature in temporal_features
            and pressure_share(temporal_pressure) >= 0.55
            and pressure_share(diversity_pressure) < 0.20
        ):
            return {
                "intervene": False,
                "reason": "shap_cancelled_non_diversity_pressure",
                "action": "none",
                "mode": "none",
                "fraction": 0.0,
                "shap_info": shap_info,
                "policy_signal": "temporal_pressure_with_sufficient_diversity",
                "policy_confidence": pressure_share(temporal_pressure),
                "diversity_pressure": diversity_pressure,
                "stagnation_pressure": stagnation_pressure,
                "temporal_pressure": temporal_pressure,
            }

        if diversity_norm < 0.25:
            policy_signal = (
                "shap_diversity_collapse"
                if shap_info is not None and diversity_pressure > 0
                else "state_diversity_collapse"
            )
        elif diversity_norm < 0.45:
            policy_signal = (
                "shap_diversity_risk"
                if shap_info is not None and diversity_pressure > 0
                else "state_diversity_risk"
            )
        elif shap_info is None:
            if stagnation_length >= 2 * self.profile.stagnation_window:
                policy_signal = "heuristic_long_stagnation"
            else:
                policy_signal = "heuristic_stagnation"
        elif pressure_share(temporal_pressure) >= 0.50:
            policy_signal = "shap_temporal_pressure"
        else:
            policy_signal = "shap_ambiguous_pressure"

        temporal_share = pressure_share(temporal_pressure)
        diversity_share = pressure_share(diversity_pressure)

        if policy_signal in {
            "shap_diversity_collapse",
            "state_diversity_collapse",
            "heuristic_diversity_collapse",
        }:
            action = "random_reinjection"
            mode = "random_aggressive"
            fraction = 0.25 + 0.08 * severity + 0.03 * bad_share
            fraction = float(np.clip(fraction, 0.25, 0.36))
            policy_confidence = max(diversity_share, 0.50)
        elif policy_signal in {
            "shap_diversity_risk",
            "state_diversity_risk",
            "heuristic_diversity_risk",
        }:
            action = "random_reinjection"
            mode = "random"
            fraction = 0.20 + 0.05 * severity + 0.03 * bad_share
            fraction = float(np.clip(fraction, 0.20, 0.30))
            policy_confidence = max(diversity_share, 0.45)
        elif policy_signal in {"shap_stagnation_pressure", "heuristic_long_stagnation"}:
            action = "partial_restart"
            mode = (
                "elite_guided_aggressive" if severity >= 0.75 else "elite_guided_wide"
            )
            fraction = 0.15 + 0.06 * severity + 0.03 * bad_share
            fraction = float(np.clip(fraction, 0.15, 0.26))
            policy_confidence = max(pressure_share(stagnation_pressure), 0.45)
        else:
            action = "partial_restart"
            mode = "elite_guided"
            fraction = 0.12 + 0.04 * severity + 0.02 * bad_share
            fraction = float(np.clip(fraction, 0.12, 0.20))
            policy_confidence = max(
                temporal_share,
                pressure_share(stagnation_pressure),
                diversity_share,
                0.25,
            )

        fraction = float(np.clip(fraction * self.rescue_scale, 0.01, 0.40))

        return {
            "intervene": True,
            "reason": f"{self.profile.name}_controller_{policy_signal}",
            "action": action,
            "mode": mode,
            "fraction": fraction,
            "shap_info": shap_info,
            "policy_signal": policy_signal,
            "policy_confidence": float(policy_confidence),
            "diversity_pressure": float(diversity_pressure),
            "stagnation_pressure": float(stagnation_pressure),
            "temporal_pressure": float(temporal_pressure),
        }

    def register_decision(
        self,
        iteration,
        decision,
        pre_fitness,
        diversity,
        stagnation_length,
        indices,
        accepted=True,
        acceptance_reason="accepted_without_gate",
        current_population_best=np.nan,
        candidate_population_best=np.nan,
        candidate_diversity=np.nan,
        lookahead_steps=0,
        current_lookahead_best=np.nan,
        candidate_lookahead_best=np.nan,
        lookahead_delta=np.nan,
    ):
        if decision["shap_info"] is not None:
            shap_info = decision["shap_info"]
            shap_row = {
                "iteration": int(iteration),
                "controller_profile": self.profile.name,
                "target": "fitness",
                "action": decision["action"],
                "mode": decision["mode"],
                "decision_reason": decision["reason"],
                "policy_signal": decision.get("policy_signal", ""),
                "policy_confidence": float(decision.get("policy_confidence", np.nan)),
                "diversity_pressure": float(decision.get("diversity_pressure", np.nan)),
                "stagnation_pressure": float(decision.get("stagnation_pressure", np.nan)),
                "temporal_pressure": float(decision.get("temporal_pressure", np.nan)),
                "pre_diversity": float(diversity),
                "pre_stagnation_length": int(stagnation_length),
                "greedy_candidates": int(decision.get("greedy_candidates", 1)),
                "greedy_improved_agents": float(
                    decision.get("greedy_improved_agents", np.nan)
                ),
                "greedy_candidate_evaluations": int(
                    decision.get("greedy_candidate_evaluations", 0)
                ),
                "proposed_intervention": bool(decision["intervene"]),
                "intervened": bool(decision["intervene"] and accepted),
                "accepted_intervention": bool(accepted),
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
            return

        self.last_action_iteration = int(iteration)
        if accepted:
            self.intervention_count += 1
        else:
            self.extend_adaptive_cooldown(
                iteration,
                self.rejected_cooldown_multiplier,
                "rejected",
            )
        self.events.append(
            {
                "event_id": len(self.events) + 1,
                "iteration": int(iteration),
                "controller_profile": self.profile.name,
                "stagnation_window": self.profile.stagnation_window,
                "action_cooldown": self.profile.action_cooldown,
                "effective_cooldown": self.profile.effective_cooldown,
                "guard_window": self.profile.guard_window,
                "late_fraction": self.profile.late_fraction,
                "max_interventions": self.profile.max_interventions,
                "rescue_scale": self.rescue_scale,
                "neutral_cooldown_multiplier": self.neutral_cooldown_multiplier,
                "rejected_cooldown_multiplier": self.rejected_cooldown_multiplier,
                "action": decision["action"],
                "mode": decision["mode"],
                "proposed_intervention": bool(decision["intervene"]),
                "policy_signal": decision.get("policy_signal", ""),
                "policy_confidence": float(decision.get("policy_confidence", np.nan)),
                "diversity_pressure": float(decision.get("diversity_pressure", np.nan)),
                "stagnation_pressure": float(decision.get("stagnation_pressure", np.nan)),
                "temporal_pressure": float(decision.get("temporal_pressure", np.nan)),
                "greedy_candidates": int(decision.get("greedy_candidates", 1)),
                "greedy_improved_agents": float(
                    decision.get("greedy_improved_agents", np.nan)
                ),
                "greedy_candidate_evaluations": int(
                    decision.get("greedy_candidate_evaluations", 0)
                ),
                "fraction": float(decision["fraction"]),
                "accepted_intervention": bool(accepted),
                "acceptance_reason": acceptance_reason,
                "lookahead_steps": int(lookahead_steps),
                "current_lookahead_best": float(current_lookahead_best),
                "candidate_lookahead_best": float(candidate_lookahead_best),
                "lookahead_delta": float(lookahead_delta),
                "selected_agents": " ".join(str(int(index)) for index in indices),
                "n_selected_agents": int(len(indices)),
                "pre_fitness": float(pre_fitness),
                "current_population_best": float(current_population_best),
                "candidate_population_best": float(candidate_population_best),
                "pre_diversity": float(diversity),
                "candidate_diversity": float(candidate_diversity),
                "pre_stagnation_length": int(stagnation_length),
                "post_iteration": int(min(iteration + 10, self.max_iter - 1)),
                "post_status": "pending" if accepted else "rejected_by_acceptance",
                "post_fitness": np.nan,
                "post_delta_fitness": np.nan,
                "post_diversity": np.nan,
                "post_stagnation_length": np.nan,
                "adaptive_cooldown_until_iteration": (
                    int(self.adaptive_cooldown_until_iteration)
                    if self.adaptive_cooldown_until_iteration is not None
                    else np.nan
                ),
                "adaptive_cooldown_reason": self.adaptive_cooldown_reason,
                "shap_target": "fitness",
                "shap_method": (decision["shap_info"] or {}).get("method", ""),
                "shap_n_coalitions": int(
                    (decision["shap_info"] or {}).get("n_coalitions", 0)
                ),
                "shap_dominant_feature": (decision["shap_info"] or {}).get(
                    "dominant_feature", ""
                ),
                "shap_dominant_value": (decision["shap_info"] or {}).get(
                    "dominant_value", np.nan
                ),
            }
        )

    def events_dataframe(self):
        return pd.DataFrame(self.events)

    def shap_dataframe(self):
        return pd.DataFrame(self.shap_rows)
