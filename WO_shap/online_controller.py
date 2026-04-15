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
        shap_interval=25,
        baseline_window=15,
        delta_window=50,
        epsilon_abs=1e-10,
        epsilon_rel=1e-6,
        diversity_low_threshold=0.15,
        diversity_high_threshold=0.40,
        alpha_adjust_scale=1.10,
        beta_adjust_scale=0.90,
        partial_restart_fraction=0.15,
        random_reinjection_fraction=0.30,
        action_cooldown=0,
    ):
        self.feature_names = [
            "alpha",
            "beta",
            "danger_signal",
            "safety_signal",
            "pop_size",
            "iteration",
        ]
        self.shap_output_name = "fitness"
        self.shap_interval = shap_interval
        self.baseline_window = baseline_window
        self.delta_window = delta_window
        self.epsilon_abs = epsilon_abs
        self.epsilon_rel = epsilon_rel
        self.diversity_low_threshold = diversity_low_threshold
        self.diversity_high_threshold = diversity_high_threshold
        self.alpha_adjust_scale = alpha_adjust_scale
        self.beta_adjust_scale = beta_adjust_scale
        self.partial_restart_fraction = partial_restart_fraction
        self.random_reinjection_fraction = random_reinjection_fraction
        self.action_cooldown = action_cooldown
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
        self.output_history = []
        self.diversity_ref = None
        self.last_shap_iteration = -10**9
        self.last_action_iteration = -10**9

    def plan(self, metrics):
        if self.runtime is None:
            raise RuntimeError("Controller runtime not configured.")

        metrics = dict(metrics)
        metrics["positions_context"] = np.array(metrics["positions_context"], dtype=float, copy=True)
        iteration = int(metrics["iteration"])
        current_fitness = float(metrics["current_fitness"])
        threshold = self._improvement_threshold(current_fitness)

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
        )
        delta_fitness = self._window_improvement(fitness_trace, self.delta_window)
        shap_requested, shap_reason = self._should_compute_shap(
            iteration,
            stagnation_length,
            delta_fitness,
            threshold,
            action,
        )
        alpha = min(1.0, float(metrics["alpha"]) * float(action["alpha_scale"]))
        beta = float(metrics["beta"]) * float(action["beta_scale"])

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
            "shap_reason": shap_reason,
            "action": action,
            "features": {
                "alpha": alpha,
                "beta": beta,
                "danger_signal": float(metrics["danger_signal"]),
                "safety_signal": float(metrics["safety_signal"]),
                "pop_size": float(metrics["pop_size"]),
                "iteration": float(iteration),
            },
            "positions_context": metrics["positions_context"],
            "best_score_context": float(metrics["best_score_context"]),
            "best_pos_context": np.array(metrics["best_pos_context"], dtype=float, copy=True),
            "second_pos_context": np.array(metrics["second_pos_context"], dtype=float, copy=True),
        }

    def commit(self, plan_data, outcome):
        record = {
            "iteration": int(plan_data["iteration"]),
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
        }
        self.records.append(record)
        self.output_history.append(record["output_fitness"])

        self.state_log.append(
            {
                "iteration": record["iteration"],
                "alpha": record["features"]["alpha"],
                "beta": record["features"]["beta"],
                "danger_signal": record["features"]["danger_signal"],
                "safety_signal": record["features"]["safety_signal"],
                "pop_size": record["features"]["pop_size"],
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
                "dominant_feature": "none",
            }
        )

        if record["event_active"]:
            self.event_log.append(
                {
                    "iteration": record["iteration"],
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
                    "dominant_feature": "none",
                }
            )

    def finalize(self):
        self.shap_log = []
        if not self.records:
            return

        event_lookup = {int(row["iteration"]): row for row in self.event_log}
        for idx, record in enumerate(self.records):
            if not record["shap_requested"]:
                continue

            baseline_metrics = self._baseline_metrics_at(idx)
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
                "iteration": int(record["iteration"]),
                "event_active": int(record["event_active"]),
                "shap_reason": record["shap_reason"],
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
            self.shap_log.append(shap_row)

            self.state_log[idx]["dominant_feature"] = dominant_feature
            event_row = event_lookup.get(int(record["iteration"]))
            if event_row is not None:
                event_row["dominant_feature"] = dominant_feature

    def save_logs(self, output_dir, function_id):
        pd.DataFrame(self.state_log).to_csv(output_dir / f"controller_state_F{function_id}.csv", index=False)
        shap_columns = [
            "iteration",
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
            "iteration",
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
            "dominant_feature",
        ]
        pd.DataFrame(self.event_log, columns=event_columns).to_csv(
            output_dir / f"controller_events_F{function_id}.csv", index=False
        )

    def event_summary(self):
        if not self.event_log:
            return {"event_count": 0, "actions": {}, "stagnation_states": {}}
        event_df = pd.DataFrame(self.event_log)
        return {
            "event_count": int(len(event_df)),
            "actions": event_df["action_taken"].value_counts().to_dict(),
            "stagnation_states": event_df["stagnation_state"].value_counts().to_dict(),
        }

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

    def _diagnose_and_build_action(self, stagnation, diversity_norm, iteration):
        if not stagnation:
            return "no_stagnation", self._action("none")

        if iteration - self.last_action_iteration < self.action_cooldown:
            return self._diagnosis_from_diversity(diversity_norm), self._action("none")

        if diversity_norm > self.diversity_high_threshold:
            diagnosis, action = "alta", self._action("adjust_alpha_beta")
        elif diversity_norm > self.diversity_low_threshold:
            diagnosis, action = "media", self._action("partial_restart")
        else:
            diagnosis, action = "baja", self._action("random_reinjection")

        if action["event_active"]:
            self.last_action_iteration = iteration
        return diagnosis, action

    def _diagnosis_from_diversity(self, diversity_norm):
        if diversity_norm > self.diversity_high_threshold:
            return "alta"
        if diversity_norm > self.diversity_low_threshold:
            return "media"
        return "baja"

    def _action(self, action_taken):
        alpha_scale = 1.0
        beta_scale = 1.0
        rescue_fraction = 0.0
        rescue_mode = "none"

        if action_taken == "adjust_alpha_beta":
            alpha_scale = self.alpha_adjust_scale
            beta_scale = self.beta_adjust_scale
        elif action_taken == "partial_restart":
            rescue_fraction = self.partial_restart_fraction
            rescue_mode = "worst"
        elif action_taken == "random_reinjection":
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
            "justification": self._action_justification(action_taken),
        }

    def _action_justification(self, action_taken):
        if action_taken == "adjust_alpha_beta":
            return "Estancamiento con diversidad alta; se ajusta alpha/beta para corregir dinamica de busqueda."
        if action_taken == "partial_restart":
            return "Estancamiento con diversidad media; se reinicia parcialmente la cola poblacional."
        if action_taken == "random_reinjection":
            return "Estancamiento con diversidad baja; se reinyecta poblacion aleatoria sin reinicio total."
        return "Sin intervencion."

    def _should_compute_shap(self, iteration, stagnation_length, delta_fitness, threshold, action):
        if action["event_active"] and iteration - self.last_shap_iteration >= self.shap_interval:
            self.last_shap_iteration = iteration
            return True, "intervention"
        return False, "none"

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
        full_population = self.runtime["search_agents_no"]

        search_agents_no = int(np.clip(round(float(state["pop_size"])), 2, len(positions_context)))
        ratio = self.runtime["female_count"] / max(full_population, 1)
        female_count = round(search_agents_no * ratio)
        male_count = female_count
        child_count = search_agents_no - female_count - male_count

        positions = np.array(positions_context, dtype=float, copy=True)[:search_agents_no]
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
