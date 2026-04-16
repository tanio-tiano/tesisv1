import hashlib
import math
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.special import gamma

from halton import hal


class OnlineXAIController:
    def __init__(
        self,
        baseline_window=15,
        delta_window=50,
        shap_post_window=10,
        epsilon_abs=1e-10,
        epsilon_rel=1e-6,
        diversity_low_threshold=0.25,
        diversity_high_threshold=0.55,
        alpha_adjust_scale=1.10,
        beta_adjust_scale=0.90,
        partial_restart_fraction=0.15,
        random_reinjection_fraction=0.25,
        action_cooldown=40,
        late_intervention_fraction=0.90,
        effective_action_cooldown=None,
        max_shap_episodes=None,
    ):
        self.feature_names = [
            "alpha",
            "beta",
            "danger_signal",
            "safety_signal",
            "diversity",
            "diversity_norm",
            "iteration",
        ]
        self.shap_output_name = "fitness"
        self.baseline_window = baseline_window
        self.delta_window = delta_window
        self.shap_post_window = int(shap_post_window)
        self.epsilon_abs = epsilon_abs
        self.epsilon_rel = epsilon_rel
        self.diversity_low_threshold = diversity_low_threshold
        self.diversity_high_threshold = diversity_high_threshold
        self.alpha_adjust_scale = alpha_adjust_scale
        self.beta_adjust_scale = beta_adjust_scale
        self.partial_restart_fraction = partial_restart_fraction
        self.random_reinjection_fraction = random_reinjection_fraction
        self.action_cooldown = action_cooldown
        self.late_intervention_fraction = float(late_intervention_fraction)
        self.effective_action_cooldown = (
            None if effective_action_cooldown is None else int(effective_action_cooldown)
        )
        self.max_shap_episodes = None if max_shap_episodes is None else int(max_shap_episodes)
        self.runtime = None
        self.reset()

    def set_runtime(
        self,
        objective,
        lb,
        ub,
        dim,
        search_agents_no,
        female_count,
        male_count,
        child_count,
        max_iter,
    ):
        self.runtime = {
            "objective": objective,
            "lb": lb,
            "ub": ub,
            "dim": dim,
            "search_agents_no": search_agents_no,
            "female_count": female_count,
            "male_count": male_count,
            "child_count": child_count,
            "max_iter": max_iter,
        }

    def reset(self):
        self.records = []
        self.state_log = []
        self.shap_log = []
        self.event_log = []
        self.episode_log = []
        self.output_history = []
        self.diversity_ref = None
        self.last_action_iteration = -10**9
        self.last_action_stagnation_length = None
        self.last_effective_action_iteration = -10**9
        self.pending_feedback_events = []
        self.next_event_id = 1

    def plan(self, metrics):
        if self.runtime is None:
            raise RuntimeError("Controller runtime not configured.")

        metrics = dict(metrics)
        metrics["positions_context"] = np.array(metrics["positions_context"], dtype=float, copy=True)
        iteration = int(metrics["iteration"])
        current_fitness = float(metrics["current_fitness"])
        threshold = self._improvement_threshold(current_fitness)
        self._update_runtime_feedback(iteration, current_fitness)

        diversity = self._population_diversity(metrics["positions_context"])
        if self.diversity_ref is None:
            self.diversity_ref = max(diversity, self.epsilon_abs)
        diversity_norm = diversity / (self.diversity_ref + self.epsilon_abs)

        fitness_trace = list(self.output_history) + [current_fitness]
        stagnation_length = self._non_improvement_count(fitness_trace, threshold)
        stagnation = stagnation_length >= self.delta_window
        diagnosis, action = self._diagnose_and_build_action(
            stagnation,
            diversity_norm,
            iteration,
            stagnation_length,
        )
        shap_requested = False
        shap_reason = "none"
        delta_fitness = self._window_improvement(fitness_trace, self.delta_window)
        decision_shap = self._empty_decision_shap()
        features = {
            "alpha": float(metrics["alpha"]),
            "beta": float(metrics["beta"]),
            "danger_signal": float(metrics["danger_signal"]),
            "safety_signal": float(metrics["safety_signal"]),
            "diversity": float(diversity),
            "diversity_norm": float(diversity_norm),
            "iteration": float(iteration),
        }

        if action["event_active"]:
            shap_requested = True
            shap_reason = "decision_intervention"
            decision_shap = self._decision_shap_analysis(
                features,
                metrics["positions_context"],
                float(metrics["best_score_context"]),
                np.array(metrics["best_pos_context"], dtype=float, copy=True),
                np.array(metrics["second_pos_context"], dtype=float, copy=True),
            )
            action = self._apply_shap_to_action(action, diagnosis, decision_shap)

        if action["event_active"]:
            self.last_action_iteration = iteration
            self.last_action_stagnation_length = int(stagnation_length)

        alpha = min(1.0, float(features["alpha"]) * float(action["alpha_scale"]))
        beta = float(features["beta"]) * float(action["beta_scale"])

        return {
            "iteration": iteration,
            "current_fitness": current_fitness,
            "stagnation": bool(stagnation),
            "stagnation_length": int(stagnation_length),
            "diversity": float(diversity),
            "diversity_norm": float(diversity_norm),
            "diagnosis": diagnosis,
            "s_t": int(stagnation_length),
            "delta_fitness": float(delta_fitness),
            "threshold": float(threshold),
            "stagnation_state": diagnosis,
            "shap_requested": bool(shap_requested),
            "shap_reason": str(shap_reason),
            "decision_shap": dict(decision_shap),
            "action": action,
            "features": {
                "alpha": alpha,
                "beta": beta,
                "danger_signal": features["danger_signal"],
                "safety_signal": features["safety_signal"],
                "diversity": features["diversity"],
                "diversity_norm": features["diversity_norm"],
                "iteration": features["iteration"],
            },
            "positions_context": metrics["positions_context"],
            "best_score_context": float(metrics["best_score_context"]),
            "best_pos_context": np.array(metrics["best_pos_context"], dtype=float, copy=True),
            "second_pos_context": np.array(metrics["second_pos_context"], dtype=float, copy=True),
        }

    def commit(self, plan_data, outcome):
        event_id = 0
        if plan_data["action"]["event_active"]:
            event_id = self.next_event_id
            self.next_event_id += 1

        record = {
            "iteration": int(plan_data["iteration"]),
            "event_id": int(event_id),
            "features": dict(plan_data["features"]),
            "positions_context": np.array(plan_data["positions_context"], dtype=float, copy=True),
            "best_score_context": float(plan_data["best_score_context"]),
            "best_pos_context": np.array(plan_data["best_pos_context"], dtype=float, copy=True),
            "second_pos_context": np.array(plan_data["second_pos_context"], dtype=float, copy=True),
            "current_fitness": float(plan_data["current_fitness"]),
            "output_fitness": float(outcome["output_fitness"]),
            "stagnation": bool(plan_data["stagnation"]),
            "stagnation_length": int(plan_data["stagnation_length"]),
            "diversity": float(plan_data["diversity"]),
            "diversity_norm": float(plan_data["diversity_norm"]),
            "diagnosis": str(plan_data["diagnosis"]),
            "s_t": int(plan_data["s_t"]),
            "delta_fitness": float(plan_data["delta_fitness"]),
            "threshold": float(plan_data["threshold"]),
            "stagnation_state": str(plan_data["stagnation_state"]),
            "shap_requested": bool(plan_data["shap_requested"]),
            "shap_reason": str(plan_data["shap_reason"]),
            "event_active": int(plan_data["action"]["event_active"]),
            "control_mode": str(plan_data["action"]["mode"]),
            "action_taken": str(plan_data["action"]["action_taken"]),
            "action_justification": str(plan_data["action"]["justification"]),
            "alpha_scale": float(plan_data["action"]["alpha_scale"]),
            "beta_scale": float(plan_data["action"]["beta_scale"]),
            "rescue_fraction": float(plan_data["action"]["rescue_fraction"]),
            "rescue_mode": str(plan_data["action"]["rescue_mode"]),
            "rescued_agents": int(outcome["rescued_agents"]),
            "decision_shap_used": int(plan_data["decision_shap"]["used"]),
            "decision_dominant_feature": str(plan_data["decision_shap"]["dominant_feature"]),
            "decision_harmful_feature": str(plan_data["decision_shap"]["harmful_feature"]),
            "decision_harmful_value": float(plan_data["decision_shap"]["harmful_value"]),
            "decision_harmful_mass": float(plan_data["decision_shap"]["harmful_mass"]),
            "decision_policy": str(plan_data["decision_shap"]["policy"]),
        }
        self.records.append(record)
        self.output_history.append(record["output_fitness"])

        self.state_log.append(
            {
                "iteration": record["iteration"],
                "event_id": int(record["event_id"]),
                "alpha": record["features"]["alpha"],
                "beta": record["features"]["beta"],
                "danger_signal": record["features"]["danger_signal"],
                "safety_signal": record["features"]["safety_signal"],
                "diversity": record["features"]["diversity"],
                "diversity_norm": record["features"]["diversity_norm"],
                "current_fitness": record["current_fitness"],
                "output_fitness": record["output_fitness"],
                "stagnation": int(record["stagnation"]),
                "stagnation_length": record["stagnation_length"],
                "diversity": record["diversity"],
                "diversity_norm": record["diversity_norm"],
                "diagnosis": record["diagnosis"],
                "s_t": record["s_t"],
                "delta_fitness": record["delta_fitness"],
                "threshold": record["threshold"],
                "stagnation_state": record["stagnation_state"],
                "event_active": record["event_active"],
                "control_mode": record["control_mode"],
                "action_taken": record["action_taken"],
                "action_justification": record["action_justification"],
                "alpha_scale": record["alpha_scale"],
                "beta_scale": record["beta_scale"],
                "rescue_fraction": record["rescue_fraction"],
                "rescue_mode": record["rescue_mode"],
                "rescued_agents": record["rescued_agents"],
                "shap_requested": int(record["shap_requested"]),
                "shap_reason": record["shap_reason"],
                "decision_shap_used": record["decision_shap_used"],
                "decision_dominant_feature": record["decision_dominant_feature"],
                "decision_harmful_feature": record["decision_harmful_feature"],
                "decision_harmful_value": record["decision_harmful_value"],
                "decision_harmful_mass": record["decision_harmful_mass"],
                "decision_policy": record["decision_policy"],
                "dominant_feature": "none",
                "trace_role": "none",
                "effect_label": "not_applicable",
                "delta_fitness_post": 0.0,
            }
        )

        if record["event_active"]:
            self.pending_feedback_events.append(
                {
                    "iteration": int(record["iteration"]),
                    "fitness_pre": float(record["output_fitness"]),
                    "evaluate_at": int(
                        min(
                            record["iteration"] + self.shap_post_window,
                            self.runtime["max_iter"] - 1,
                        )
                    ),
                }
            )
            self.event_log.append(
                {
                    "event_id": int(record["event_id"]),
                    "iteration": record["iteration"],
                    "t_pre": int(record["iteration"]),
                    "t_action": int(record["iteration"]),
                    "t_post": int(
                        min(
                            record["iteration"] + self.shap_post_window,
                            self.runtime["max_iter"] - 1,
                        )
                    ),
                    "stagnation_state": record["stagnation_state"],
                    "action_taken": record["action_taken"],
                    "action_justification": record["action_justification"],
                    "stagnation": int(record["stagnation"]),
                    "stagnation_length": record["stagnation_length"],
                    "diversity": record["diversity"],
                    "diversity_norm": record["diversity_norm"],
                    "diagnosis": record["diagnosis"],
                    "alpha_scale": record["alpha_scale"],
                    "beta_scale": record["beta_scale"],
                    "rescue_fraction": record["rescue_fraction"],
                    "rescue_mode": record["rescue_mode"],
                    "rescued_agents": record["rescued_agents"],
                    "decision_shap_used": record["decision_shap_used"],
                    "decision_dominant_feature": record["decision_dominant_feature"],
                    "decision_harmful_feature": record["decision_harmful_feature"],
                    "decision_harmful_value": record["decision_harmful_value"],
                    "decision_harmful_mass": record["decision_harmful_mass"],
                    "decision_policy": record["decision_policy"],
                    "dominant_feature": "none",
                    "dominant_feature_pre": "none",
                    "dominant_feature_post": "none",
                    "fitness_pre": record["output_fitness"],
                    "fitness_post": record["output_fitness"],
                    "delta_fitness_post": 0.0,
                    "stagnation_length_pre": record["stagnation_length"],
                    "stagnation_length_post": record["stagnation_length"],
                    "diversity_pre": record["diversity"],
                    "diversity_post": record["diversity"],
                    "diversity_norm_pre": record["diversity_norm"],
                    "diversity_norm_post": record["diversity_norm"],
                    "effective_intervention": 0,
                    "effect_label": "pending",
                }
            )

    def finalize(self):
        self.shap_log = []
        if not self.records:
            return

        self.episode_log = self._extract_stagnation_episodes()
        record_index_by_iteration = {
            int(record["iteration"]): idx for idx, record in enumerate(self.records)
        }

        for event_row in self.event_log:
            pre_iteration = int(event_row["t_pre"])
            post_iteration = int(event_row["t_post"])
            pre_idx = record_index_by_iteration[pre_iteration]
            post_idx = record_index_by_iteration[post_iteration]
            pre_record = self.records[pre_idx]
            post_record = self.records[post_idx]

            pre_shap = self._build_shap_row(
                pre_record,
                pre_idx,
                event_row["event_id"],
                "pre",
                "pre_intervention",
                event_row["t_pre"],
                event_row["t_post"],
            )
            post_shap = self._build_shap_row(
                post_record,
                post_idx,
                event_row["event_id"],
                "post",
                "post_intervention",
                event_row["t_pre"],
                event_row["t_post"],
            )
            self.shap_log.extend([pre_shap, post_shap])

            effect_threshold = self._improvement_threshold(pre_record["output_fitness"])
            delta_fitness_post = float(pre_record["output_fitness"]) - float(post_record["output_fitness"])
            if delta_fitness_post > effect_threshold:
                effect_label = "improved"
                effective_intervention = 1
            elif abs(delta_fitness_post) <= effect_threshold:
                effect_label = "neutral"
                effective_intervention = 0
            else:
                effect_label = "worsened"
                effective_intervention = 0

            event_row["dominant_feature"] = pre_shap["dominant_feature"]
            event_row["dominant_feature_pre"] = pre_shap["dominant_feature"]
            event_row["dominant_feature_post"] = post_shap["dominant_feature"]
            event_row["fitness_pre"] = float(pre_record["output_fitness"])
            event_row["fitness_post"] = float(post_record["output_fitness"])
            event_row["delta_fitness_post"] = float(delta_fitness_post)
            event_row["stagnation_length_pre"] = int(pre_record["stagnation_length"])
            event_row["stagnation_length_post"] = int(post_record["stagnation_length"])
            event_row["diversity_pre"] = float(pre_record["diversity"])
            event_row["diversity_post"] = float(post_record["diversity"])
            event_row["diversity_norm_pre"] = float(pre_record["diversity_norm"])
            event_row["diversity_norm_post"] = float(post_record["diversity_norm"])
            event_row["effective_intervention"] = int(effective_intervention)
            event_row["effect_label"] = effect_label

            self._annotate_trace_state(pre_idx, event_row["event_id"], "pre", pre_shap["dominant_feature"], delta_fitness_post, effect_label)
            self._annotate_trace_state(post_idx, event_row["event_id"], "post", post_shap["dominant_feature"], delta_fitness_post, effect_label)

    def save_logs(self, output_dir, function_id):
        pd.DataFrame(self.state_log).to_csv(output_dir / f"controller_state_F{function_id}.csv", index=False)
        shap_columns = [
            "event_id",
            "phase",
            "iteration",
            "t_pre",
            "t_post",
            "event_active",
            "shap_reason",
            "selected_output",
            "BASE_fitness",
            "OUTPUT_fitness",
            "OBSERVED_fitness",
            "stagnation_state",
            "action_taken",
            "dominant_feature",
        ] + [f"SHAP_{feature}" for feature in self.feature_names]
        pd.DataFrame(self.shap_log, columns=shap_columns).to_csv(
            output_dir / f"shap_values_F{function_id}.csv", index=False
        )
        event_columns = [
            "event_id",
            "iteration",
            "t_pre",
            "t_action",
            "t_post",
            "stagnation_state",
            "action_taken",
            "action_justification",
            "stagnation",
            "stagnation_length",
            "diversity",
            "diversity_norm",
            "diagnosis",
            "alpha_scale",
            "beta_scale",
            "rescue_fraction",
            "rescue_mode",
            "rescued_agents",
            "decision_shap_used",
            "decision_dominant_feature",
            "decision_harmful_feature",
            "decision_harmful_value",
            "decision_harmful_mass",
            "decision_policy",
            "dominant_feature",
            "dominant_feature_pre",
            "dominant_feature_post",
            "fitness_pre",
            "fitness_post",
            "delta_fitness_post",
            "stagnation_length_pre",
            "stagnation_length_post",
            "diversity_pre",
            "diversity_post",
            "diversity_norm_pre",
            "diversity_norm_post",
            "effective_intervention",
            "effect_label",
        ]
        pd.DataFrame(self.event_log, columns=event_columns).to_csv(
            output_dir / f"controller_events_F{function_id}.csv", index=False
        )
        episode_columns = [
            "episode_id",
            "start_iteration",
            "end_iteration",
            "duration",
            "max_stagnation_length",
            "selected_iteration",
            "selected_for_shap",
        ]
        pd.DataFrame(self.episode_log, columns=episode_columns).to_csv(
            output_dir / f"controller_episodes_F{function_id}.csv", index=False
        )

    def event_summary(self):
        if not self.event_log:
            return {
                "event_count": 0,
                "episode_count": 0,
                "actions": {},
                "stagnation_states": {},
                "effects": {},
            }
        event_df = pd.DataFrame(self.event_log)
        return {
            "event_count": int(len(event_df)),
            "episode_count": int(len(self.episode_log)),
            "actions": event_df["action_taken"].value_counts().to_dict(),
            "stagnation_states": event_df["stagnation_state"].value_counts().to_dict(),
            "effects": event_df["effect_label"].value_counts().to_dict(),
        }

    def _build_shap_row(self, record, record_index, event_id, phase, shap_reason, t_pre, t_post):
        baseline_metrics = self._baseline_metrics_at(record_index)
        metrics = dict(record["features"])
        metrics["positions_context"] = record["positions_context"]
        metrics["best_score_context"] = record["best_score_context"]
        metrics["best_pos_context"] = record["best_pos_context"]
        metrics["second_pos_context"] = record["second_pos_context"]
        output_value = self._coalition_value(set(self.feature_names), metrics, baseline_metrics)
        baseline_output = self._coalition_value(set(), metrics, baseline_metrics)
        shap_values = self._compute_shapley_output(metrics, baseline_metrics)
        dominant_feature = self._dominant_feature(shap_values)

        shap_row = {
            "event_id": int(event_id),
            "phase": str(phase),
            "iteration": int(record["iteration"]),
            "t_pre": int(t_pre),
            "t_post": int(t_post),
            "event_active": int(record["event_active"]),
            "shap_reason": shap_reason,
            "selected_output": self.shap_output_name,
            "BASE_fitness": float(baseline_output),
            "OUTPUT_fitness": float(output_value),
            "OBSERVED_fitness": float(record["output_fitness"]),
            "stagnation_state": record["stagnation_state"],
            "action_taken": record["action_taken"],
            "dominant_feature": dominant_feature,
        }
        for feature in self.feature_names:
            shap_row[f"SHAP_{feature}"] = shap_values[feature]
        return shap_row

    def _annotate_trace_state(self, state_index, event_id, trace_role, dominant_feature, delta_fitness_post, effect_label):
        self.records[state_index]["shap_requested"] = True
        self.records[state_index]["shap_reason"] = f"{trace_role}_intervention"
        state_row = self.state_log[state_index]
        if state_row["trace_role"] == "none":
            state_row["trace_role"] = trace_role
            state_row["event_id"] = int(event_id)
        elif trace_role not in state_row["trace_role"].split("|"):
            state_row["trace_role"] = f"{state_row['trace_role']}|{trace_role}"
            state_row["event_id"] = f"{state_row['event_id']}|{int(event_id)}"
        state_row["shap_requested"] = 1
        state_row["shap_reason"] = f"{trace_role}_intervention"
        state_row["dominant_feature"] = dominant_feature
        state_row["delta_fitness_post"] = float(delta_fitness_post)
        state_row["effect_label"] = str(effect_label)

    def _improvement_threshold(self, fitness_value):
        return max(self.epsilon_abs, self.epsilon_rel * max(abs(float(fitness_value)), 1.0))

    def _non_improvement_count(self, fitness_trace, threshold):
        if len(fitness_trace) < 2:
            return 0
        count = 0
        for idx in range(len(fitness_trace) - 1, 0, -1):
            improvement = float(fitness_trace[idx - 1]) - float(fitness_trace[idx])
            if improvement <= threshold:
                count += 1
            else:
                break
        return count

    def _window_improvement(self, fitness_trace, window):
        if len(fitness_trace) <= window:
            return float("inf")
        return float(fitness_trace[-window - 1]) - float(fitness_trace[-1])

    def _population_diversity(self, positions):
        positions = np.array(positions, dtype=float, copy=False)
        if positions.size == 0:
            return 0.0
        centroid = np.mean(positions, axis=0)
        distances = np.linalg.norm(positions - centroid, axis=1)
        return float(np.mean(distances))

    def _diagnose_and_build_action(self, stagnation, diversity_norm, iteration, stagnation_length):
        if not stagnation:
            self.last_action_stagnation_length = None
            return "no_stagnation", self._action("none")

        late_iteration_cutoff = int(
            math.floor(self.runtime["max_iter"] * self.late_intervention_fraction)
        )
        if iteration >= late_iteration_cutoff:
            return self._diagnosis_from_diversity(diversity_norm), self._action("none")

        if (
            self.last_action_stagnation_length is not None
            and stagnation_length < self.last_action_stagnation_length
        ):
            self.last_action_stagnation_length = None

        if iteration - self.last_action_iteration < self.action_cooldown:
            return self._diagnosis_from_diversity(diversity_norm), self._action("none")

        diagnosis = self._diagnosis_from_diversity(diversity_norm)
        if (
            diagnosis != "baja"
            and iteration - self.last_effective_action_iteration < self._effective_cooldown()
        ):
            return diagnosis, self._action("none")

        if (
            diagnosis != "baja"
            and iteration - self.last_effective_action_iteration < self._post_effective_guard_window()
            and diversity_norm > self.diversity_low_threshold
        ):
            return diagnosis, self._action("none")

        if (
            diagnosis != "baja"
            and self.last_effective_action_iteration > -10**8
            and iteration >= int(0.80 * self.runtime["max_iter"])
        ):
            return diagnosis, self._action("none")

        if (
            self.last_action_stagnation_length is not None
            and stagnation_length - self.last_action_stagnation_length < self.delta_window
        ):
            return diagnosis, self._action("none")

        min_stagnation = self._minimum_stagnation_for_intervention(diagnosis)
        if stagnation_length < min_stagnation:
            return diagnosis, self._action("none")

        severity = self._stagnation_severity(stagnation_length)
        if diagnosis == "alta":
            diagnosis, action = "alta", self._action(
                "partial_restart",
                rescue_fraction=self._adaptive_rescue_fraction(
                    self.partial_restart_fraction * self._severity_base_fraction("alta", severity),
                    stagnation_length,
                    self._severity_cap("alta", severity),
                ),
                rescue_mode=self._severity_rescue_mode("alta", severity),
            )
        elif diagnosis == "media":
            diagnosis, action = "media", self._action(
                "partial_restart",
                rescue_fraction=self._adaptive_rescue_fraction(
                    self.partial_restart_fraction * self._severity_base_fraction("media", severity),
                    stagnation_length,
                    self._severity_cap("media", severity),
                ),
                rescue_mode=self._severity_rescue_mode("media", severity),
            )
        else:
            diagnosis, action = "baja", self._action(
                "random_reinjection",
                rescue_fraction=self._adaptive_rescue_fraction(
                    self.random_reinjection_fraction * self._severity_base_fraction("baja", severity),
                    stagnation_length,
                    self._severity_cap("baja", severity),
                ),
                rescue_mode=self._severity_rescue_mode("baja", severity),
            )

        return diagnosis, action

    def _diagnosis_from_diversity(self, diversity_norm):
        if diversity_norm > self.diversity_high_threshold:
            return "alta"
        if diversity_norm > self.diversity_low_threshold:
            return "media"
        return "baja"

    def _minimum_stagnation_for_intervention(self, diagnosis):
        if diagnosis == "alta":
            return int(self.delta_window + max(20, round(0.60 * self.delta_window)))
        if diagnosis == "media":
            return int(self.delta_window + max(10, round(0.30 * self.delta_window)))
        return int(self.delta_window)

    def _action(self, action_taken, rescue_fraction=None, rescue_mode=None):
        alpha_scale = 1.0
        beta_scale = 1.0
        rescue_fraction = 0.0 if rescue_fraction is None else float(rescue_fraction)
        rescue_mode = "none" if rescue_mode is None else str(rescue_mode)

        if action_taken == "partial_restart":
            if rescue_mode == "none":
                rescue_fraction = self.partial_restart_fraction
                rescue_mode = "elite_guided"
        elif action_taken == "random_reinjection":
            if rescue_mode == "none":
                rescue_fraction = self.random_reinjection_fraction
                rescue_mode = "random"

        event_active = action_taken != "none"
        return {
            "event_active": event_active,
            "mode": "intervene" if event_active else "baseline",
            "action_taken": action_taken,
            "alpha_scale": float(alpha_scale),
            "beta_scale": float(beta_scale),
            "rescue_fraction": float(rescue_fraction),
            "rescue_mode": rescue_mode,
            "justification": self._action_justification(action_taken, rescue_mode),
        }

    def _empty_decision_shap(self):
        return {
            "used": 0,
            "dominant_feature": "none",
            "harmful_feature": "none",
            "harmful_value": 0.0,
            "harmful_mass": 0.0,
            "policy": "not_evaluated",
        }

    def _effective_cooldown(self):
        if self.effective_action_cooldown is not None:
            return int(self.effective_action_cooldown)
        return int(max(60, 2 * int(self.delta_window)))

    def _post_effective_guard_window(self):
        return int(max(160, 3 * int(self.delta_window)))

    def _update_runtime_feedback(self, iteration, current_fitness):
        if not self.pending_feedback_events:
            return

        pending = []
        for event in self.pending_feedback_events:
            if int(event["evaluate_at"]) > int(iteration):
                pending.append(event)
                continue

            effect_threshold = self._improvement_threshold(float(event["fitness_pre"]))
            delta = float(event["fitness_pre"]) - float(current_fitness)
            if delta > effect_threshold:
                self.last_effective_action_iteration = int(event["evaluate_at"])

        self.pending_feedback_events = pending

    def _current_baseline_metrics(self):
        history_slice = self.records[-self.baseline_window :]
        if not history_slice:
            return {name: 0.0 for name in self.feature_names}
        return {
            feature: float(np.mean([row["features"][feature] for row in history_slice]))
            for feature in self.feature_names
        }

    def _decision_shap_analysis(
        self,
        features,
        positions_context,
        best_score_context,
        best_pos_context,
        second_pos_context,
    ):
        baseline_metrics = self._current_baseline_metrics()
        metrics = dict(features)
        metrics["positions_context"] = np.array(positions_context, dtype=float, copy=True)
        metrics["best_score_context"] = float(best_score_context)
        metrics["best_pos_context"] = np.array(best_pos_context, dtype=float, copy=True)
        metrics["second_pos_context"] = np.array(second_pos_context, dtype=float, copy=True)
        shap_values = self._compute_shapley_output(metrics, baseline_metrics)
        harmful_feature, harmful_value = self._dominant_harmful_feature(shap_values)
        harmful_mass = float(sum(max(0.0, float(value)) for value in shap_values.values()))
        return {
            "used": 1,
            "dominant_feature": self._dominant_feature(shap_values),
            "harmful_feature": harmful_feature,
            "harmful_value": float(harmful_value),
            "harmful_mass": harmful_mass,
            "policy": "evaluated",
            "shap_values": shap_values,
        }

    def _apply_shap_to_action(self, action, diagnosis, decision_shap):
        adjusted = dict(action)
        shap_values = dict(decision_shap.get("shap_values", {}))
        harmful_feature = str(decision_shap.get("harmful_feature", "none"))
        harmful_mass = float(decision_shap.get("harmful_mass", 0.0))
        diversity_mass = max(0.0, float(shap_values.get("diversity", 0.0))) + max(
            0.0, float(shap_values.get("diversity_norm", 0.0))
        )

        if harmful_feature == "none" or harmful_mass <= self.epsilon_abs:
            decision_shap["policy"] = "skip_no_harmful_signal"
            return self._action("none")

        if harmful_feature == "iteration" and diversity_mass <= 0.25 * harmful_mass:
            decision_shap["policy"] = "skip_iteration_dominant"
            return self._action("none")

        if diagnosis == "media" and harmful_feature in {"alpha", "beta", "iteration"}:
            decision_shap["policy"] = "skip_schedule_signal_media"
            return self._action("none")

        if (
            diagnosis != "baja"
            and harmful_feature in {"alpha", "beta", "iteration"}
            and diversity_mass <= 0.40 * harmful_mass
        ):
            decision_shap["policy"] = "skip_schedule_signal"
            return self._action("none")

        if harmful_feature in {"diversity", "diversity_norm"}:
            adjusted["rescue_fraction"] = float(adjusted["rescue_fraction"]) * 1.10
            if adjusted["action_taken"] == "partial_restart":
                adjusted["rescue_mode"] = "elite_guided_wide"
            decision_shap["policy"] = "reinforce_diversity_signal"
        elif harmful_feature in {"danger_signal", "safety_signal"}:
            if adjusted["action_taken"] == "partial_restart":
                adjusted["rescue_mode"] = "elite_guided"
            decision_shap["policy"] = "respect_regime_signal"
        elif harmful_feature in {"alpha", "beta", "iteration"}:
            adjusted["rescue_fraction"] = float(adjusted["rescue_fraction"]) * 0.85
            if adjusted["action_taken"] == "partial_restart" and adjusted["rescue_mode"] == "elite_guided_wide":
                adjusted["rescue_mode"] = "elite_guided"
            decision_shap["policy"] = "attenuate_schedule_signal"
        else:
            decision_shap["policy"] = "keep_rule_action"

        adjusted["rescue_fraction"] = max(0.0, float(adjusted["rescue_fraction"]))
        adjusted["justification"] = (
            f"{adjusted['justification']} SHAP-decision: {decision_shap['policy']} "
            f"(feature={harmful_feature})."
        )
        return adjusted

    def _action_justification(self, action_taken, rescue_mode):
        if action_taken == "partial_restart":
            if rescue_mode == "elite_guided_wide":
                return "Estancamiento con diversidad alta; se recoloca la cola poblacional alrededor de los lideres con perturbacion amplia."
            if rescue_mode == "elite_guided_aggressive":
                return "Estancamiento persistente; se recoloca una fraccion mayor de la cola poblacional alrededor de los lideres con perturbacion reforzada."
            return "Estancamiento detectado; se recoloca parcialmente la cola poblacional alrededor de best_pos y second_pos."
        if action_taken == "random_reinjection":
            if rescue_mode == "random_aggressive":
                return "Estancamiento severo con diversidad baja; se reinyecta una fraccion mayor de poblacion aleatoria."
            return "Estancamiento con diversidad baja; se reinyecta poblacion aleatoria sin reinicio total."
        return "Sin intervencion."

    def _adaptive_rescue_fraction(self, base_fraction, stagnation_length, cap):
        excess = max(0, int(stagnation_length) - int(self.delta_window))
        severity = min(1.0, excess / max(1, int(self.delta_window)))
        return min(float(base_fraction) * (1.0 + 0.70 * severity), float(cap))

    def _stagnation_severity(self, stagnation_length):
        excess = max(0, int(stagnation_length) - int(self.delta_window))
        if excess >= max(18, int(0.35 * self.delta_window)):
            return "high"
        if excess >= max(8, int(0.15 * self.delta_window)):
            return "medium"
        return "low"

    def _severity_base_fraction(self, diagnosis, severity):
        table = {
            ("alta", "low"): 0.85,
            ("alta", "medium"): 1.00,
            ("alta", "high"): 1.10,
            ("media", "low"): 0.95,
            ("media", "medium"): 1.10,
            ("media", "high"): 1.20,
            ("baja", "low"): 1.00,
            ("baja", "medium"): 1.10,
            ("baja", "high"): 1.25,
        }
        return table[(diagnosis, severity)]

    def _severity_cap(self, diagnosis, severity):
        table = {
            ("alta", "low"): 0.18,
            ("alta", "medium"): 0.20,
            ("alta", "high"): 0.22,
            ("media", "low"): 0.18,
            ("media", "medium"): 0.22,
            ("media", "high"): 0.26,
            ("baja", "low"): 0.28,
            ("baja", "medium"): 0.32,
            ("baja", "high"): 0.36,
        }
        return table[(diagnosis, severity)]

    def _severity_rescue_mode(self, diagnosis, severity):
        if diagnosis == "baja":
            return "random_aggressive" if severity == "high" else "random"
        if diagnosis == "alta":
            return "elite_guided_wide"
        if severity == "high":
            return "elite_guided_wide"
        return "elite_guided"

    def _baseline_metrics_at(self, index):
        start = max(0, index - self.baseline_window + 1)
        history_slice = self.records[start : index + 1]
        if not history_slice:
            return {name: 0.0 for name in self.feature_names}
        return {
            feature: float(np.mean([row["features"][feature] for row in history_slice]))
            for feature in self.feature_names
        }

    def _compute_shapley_output(self, metrics, baseline_metrics):
        shap_values = {feature: 0.0 for feature in self.feature_names}
        feature_count = len(self.feature_names)
        factorial_den = math.factorial(feature_count)
        for feature in self.feature_names:
            other_features = [name for name in self.feature_names if name != feature]
            contribution = 0.0
            for subset_size in range(feature_count):
                for subset in combinations(other_features, subset_size):
                    subset = set(subset)
                    weight = (
                        math.factorial(len(subset))
                        * math.factorial(feature_count - len(subset) - 1)
                        / factorial_den
                    )
                    with_feature = self._coalition_value(subset | {feature}, metrics, baseline_metrics)
                    without_feature = self._coalition_value(subset, metrics, baseline_metrics)
                    contribution += weight * (with_feature - without_feature)
            shap_values[feature] = float(contribution)
        return shap_values

    def _coalition_value(self, coalition, metrics, baseline_metrics):
        mixed = {}
        for feature in self.feature_names:
            mixed[feature] = metrics[feature] if feature in coalition else baseline_metrics[feature]
        return self._direct_fitness_value(
            mixed,
            metrics["positions_context"],
            metrics["best_score_context"],
            metrics["best_pos_context"],
            metrics["second_pos_context"],
        )

    def _direct_fitness_value(
        self,
        state,
        positions_context,
        best_score_context,
        best_pos_context,
        second_pos_context,
    ):
        objective = self.runtime["objective"]
        lb = self.runtime["lb"]
        ub = self.runtime["ub"]
        dim = self.runtime["dim"]
        search_agents_no = int(self.runtime["search_agents_no"])
        female_count = int(self.runtime["female_count"])
        male_count = int(self.runtime["male_count"])
        child_count = int(self.runtime["child_count"])

        positions = np.array(positions_context, dtype=float, copy=True)[:search_agents_no]
        positions = self._match_target_diversity(
            positions,
            float(state["diversity"]),
            float(state["diversity_norm"]),
        )
        positions = np.clip(positions, lb, ub)
        current_best = float(best_score_context)
        best_pos = np.array(best_pos_context, dtype=float, copy=True)
        second_pos = np.array(second_pos_context, dtype=float, copy=True)
        gbest_x = np.zeros((search_agents_no, dim), dtype=float)

        alpha = float(state["alpha"])
        beta = float(np.clip(state["beta"], 0.0, 1.0))
        danger_signal = float(state["danger_signal"])
        safety_signal = float(np.clip(state["safety_signal"], 0.0, 1.0))
        rng = np.random.default_rng(self._state_seed(state))

        if abs(danger_signal) >= 1:
            r3 = rng.random()
            p1 = rng.permutation(search_agents_no)
            p2 = rng.permutation(search_agents_no)
            positions = positions + (beta * (r3**2)) * (
                positions[p1, :] - positions[p2, :]
            )
        else:
            if safety_signal >= 0.5:
                for i in range(male_count):
                    positions[i, :] = lb + hal(i + 1, 7) * (ub - lb)

                last_male_index = max(male_count - 1, 0)
                for j in range(male_count, male_count + female_count):
                    positions[j, :] = positions[j, :] + alpha * (
                        positions[last_male_index, :] - positions[j, :]
                    ) + (1 - alpha) * (gbest_x[j, :] - positions[j, :])

                for i in range(search_agents_no - child_count, search_agents_no):
                    levy_step = self._levy_flight(dim, rng)
                    o = gbest_x[i, :] + positions[i, :] * levy_step
                    positions[i, :] = rng.random() * (o - positions[i, :])

            if safety_signal < 0.5 and abs(danger_signal) >= 0.5:
                for i in range(search_agents_no):
                    r4 = rng.random()
                    positions[i, :] = positions[i, :] * danger_signal - np.abs(
                        gbest_x[i, :] - positions[i, :]
                    ) * r4**2

            if safety_signal < 0.5 and abs(danger_signal) < 0.5:
                for i in range(search_agents_no):
                    for j_dim in range(dim):
                        theta1 = rng.random()
                        a1 = beta * rng.random() - beta
                        b1 = np.tan(theta1 * np.pi)
                        x1 = best_pos[j_dim] - a1 * b1 * np.abs(
                            best_pos[j_dim] - positions[i, j_dim]
                        )

                        theta2 = rng.random()
                        a2 = beta * rng.random() - beta
                        b2 = np.tan(theta2 * np.pi)
                        x2 = second_pos[j_dim] - a2 * b2 * np.abs(
                            second_pos[j_dim] - positions[i, j_dim]
                        )
                        positions[i, j_dim] = (x1 + x2) / 2

        positions = np.clip(positions, lb, ub)
        updated_fitness = np.array([objective(agent) for agent in positions], dtype=float)
        candidate_best = float(np.min(updated_fitness))
        return float(min(current_best, candidate_best))

    def _match_target_diversity(self, positions, target_diversity, target_diversity_norm):
        positions = np.array(positions, dtype=float, copy=True)
        if positions.size == 0 or self.diversity_ref is None:
            return positions

        if np.isfinite(float(target_diversity)) and float(target_diversity) > 0.0:
            target_diversity_value = float(target_diversity)
        else:
            target_diversity_value = max(0.0, float(target_diversity_norm)) * float(self.diversity_ref)
        centroid = np.mean(positions, axis=0)
        centered = positions - centroid
        current_diversity = float(np.mean(np.linalg.norm(centered, axis=1)))

        if current_diversity <= self.epsilon_abs:
            return positions

        scale = target_diversity_value / current_diversity
        return centroid + centered * scale

    def _state_seed(self, state):
        payload = "|".join(f"{name}:{float(state[name]):.12f}" for name in self.feature_names)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return int(digest[:16], 16) % (2**32)

    def _levy_flight(self, dim, rng):
        beta = 1.5
        num = gamma(1 + beta) * np.sin(np.pi * beta / 2)
        den = gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2))
        sigma = (num / den) ** (1 / beta)
        u = rng.normal(0.0, sigma, dim)
        v = rng.normal(0.0, 1.0, dim)
        return u / (np.abs(v) ** (1 / beta))

    def _dominant_feature(self, shap_values):
        best_feature = "none"
        best_value = -float("inf")
        for feature, value in shap_values.items():
            if abs(value) > best_value:
                best_feature = feature
                best_value = abs(value)
        return best_feature

    def _dominant_harmful_feature(self, shap_values):
        best_feature = "none"
        best_value = 0.0
        for feature, value in shap_values.items():
            value = float(value)
            if value > best_value:
                best_feature = feature
                best_value = value
        return best_feature, float(best_value)

    def _extract_stagnation_episodes(self):
        episodes = []
        current_indices = []
        for idx, record in enumerate(self.records):
            if record["stagnation"]:
                current_indices.append(idx)
                continue
            if current_indices:
                episodes.append(self._build_episode(current_indices, len(episodes) + 1))
                current_indices = []
        if current_indices:
            episodes.append(self._build_episode(current_indices, len(episodes) + 1))

        ranked_episodes = sorted(
            episodes,
            key=lambda episode: (
                episode["max_stagnation_length"],
                episode["duration"],
                -episode["start_iteration"],
            ),
            reverse=True,
        )
        if self.max_shap_episodes is None:
            selected_ids = {episode["episode_id"] for episode in ranked_episodes}
        else:
            selected_ids = {
                episode["episode_id"] for episode in ranked_episodes[: self.max_shap_episodes]
            }
        for episode in episodes:
            episode["selected_for_shap"] = int(episode["episode_id"] in selected_ids)
        return episodes

    def _build_episode(self, indices, episode_id):
        rows = [self.records[idx] for idx in indices]
        representative = max(
            rows,
            key=lambda row: (
                row["stagnation_length"],
                row["event_active"],
                row["iteration"],
            ),
        )
        return {
            "episode_id": int(episode_id),
            "start_iteration": int(rows[0]["iteration"]),
            "end_iteration": int(rows[-1]["iteration"]),
            "duration": int(len(rows)),
            "max_stagnation_length": int(max(row["stagnation_length"] for row in rows)),
            "selected_iteration": int(representative["iteration"]),
            "selected_for_shap": 0,
        }
