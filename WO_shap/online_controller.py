import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier


class OnlineXAIController:
    def __init__(
        self,
        warmup_iters=20, # número de iteraciones de calentamiento antes de confiar en el modelo, se usarán solo heurísticas
        min_training_samples=12, # mínimo de muestras para entrenar el modelo, incluso si se alcanzó el riesgo
        retrain_interval=20, # cada tantas iteraciones se reentrena el modelo, incluso si no se alcanza el riesgo
        event_interval=20, # cada tantas iteraciones se fuerza un análisis, aunque no se alcance el riesgo
        stagnation_threshold=8, # si lleva tantas iteraciones sin mejorar, es un riesgo
        min_improvement=1e-8, # umbral para considerar que hubo mejora
        diversity_floor_ratio=0.15, #Diversidad bajo el 15% indica riesgo
        risk_trigger=0.60, #Disparador de riesgo
        random_state=42, #Semilla del random forest
    ):
        self.feature_names = [
            "delta_best_window", # cuánto ha mejorado el mejor resultado en la ventana reciente (puede ser negativo o muy pequeño si no mejora)
            "stagnation_length", # cuántas iteraciones lleva sin mejorar el mejor resultado
            "population_diversity", # medida de diversidad de la población (puede ser baja si todos los individuos son similares)
            "diversity_ratio", # relación entre la diversidad actual y la diversidad inicial (puede ser baja si la población se ha vuelto homogénea)
            "mean_distance_to_best", # distancia promedio de los individuos al mejor resultado (puede ser baja si todos están cerca del mismo punto)
            "success_rate_recent", # tasa de éxito en la ventana reciente (puede ser baja si no se encuentran soluciones mejores con frecuencia)
        ]
        self.warmup_iters = warmup_iters
        self.min_training_samples = min_training_samples
        self.retrain_interval = retrain_interval
        self.event_interval = event_interval
        self.stagnation_threshold = stagnation_threshold
        self.min_improvement = min_improvement
        self.diversity_floor_ratio = diversity_floor_ratio
        self.risk_trigger = risk_trigger
        self.random_state = random_state
        self.reset()

    def reset(self):
        self.model = None
        self.explainer = None
        self.last_metrics = None
        self.last_trained_iteration = -1
        self.last_event_iteration = -1
        self.training_X = []
        self.training_y = []
        self.state_log = []
        self.shap_log = []

    def update(self, metrics):
        metrics = dict(metrics)

        if self.last_metrics is not None:
            self.training_X.append(self._features_from_metrics(self.last_metrics))
            self.training_y.append(self._risk_label(metrics))

        current_features = self._features_from_metrics(metrics)
        heuristic_risk = self._heuristic_risk(metrics)
        event_active, event_reason = self._should_trigger_event(metrics, heuristic_risk)

        model_risk = heuristic_risk
        shap_values = {name: 0.0 for name in self.feature_names}
        if event_active:
            model_risk, shap_values = self._run_event_analysis(metrics, current_features)

        action = self._build_action(model_risk, shap_values, event_active)
        self._log_iteration(metrics, heuristic_risk, model_risk, shap_values, action, event_active, event_reason)
        self.last_metrics = metrics
        return action

    def finalize(self):
        return

    def save_logs(self, output_dir, function_id):
        pd.DataFrame(self.state_log).to_csv(
            output_dir / f"controller_state_F{function_id}.csv",
            index=False,
        )
        pd.DataFrame(self.shap_log).to_csv(
            output_dir / f"shap_values_F{function_id}.csv",
            index=False,
        )

    def _features_from_metrics(self, metrics):
        return np.array(
            [
                metrics["delta_best_window"],
                metrics["stagnation_length"],
                metrics["population_diversity"],
                metrics["diversity_ratio"],
                metrics["mean_distance_to_best"],
                metrics["success_rate_recent"],
            ],
            dtype=float,
        )

    def _risk_label(self, metrics):
        weak_progress = metrics["delta_best_window"] <= self.min_improvement
        low_diversity = metrics["diversity_ratio"] <= self.diversity_floor_ratio
        prolonged_stagnation = metrics["stagnation_length"] >= self.stagnation_threshold
        crowding = metrics["mean_distance_to_best"] <= 0.20
        low_success = metrics["success_rate_recent"] <= 0.20
        return int((weak_progress and low_diversity) or prolonged_stagnation or (crowding and low_success))

    def _heuristic_risk(self, metrics):
        low_progress = 1.0 if metrics["delta_best_window"] <= self.min_improvement else 0.0
        stagnation = min(metrics["stagnation_length"] / max(self.stagnation_threshold, 1), 1.0)
        low_diversity = float(np.clip(1.0 - metrics["diversity_ratio"], 0.0, 1.0))
        crowding = float(np.clip(1.0 - metrics["mean_distance_to_best"], 0.0, 1.0))
        low_success = float(np.clip(1.0 - metrics["success_rate_recent"], 0.0, 1.0))
        return float(
            np.clip(
                0.30 * low_progress
                + 0.25 * stagnation
                + 0.20 * low_diversity
                + 0.15 * crowding
                + 0.10 * low_success,
                0.0,
                1.0,
            )
        )

    def _should_trigger_event(self, metrics, heuristic_risk):
        reasons = []
        if metrics["stagnation_length"] >= self.stagnation_threshold:
            reasons.append("stagnation")
        if metrics["diversity_ratio"] <= self.diversity_floor_ratio:
            reasons.append("low_diversity")
        if metrics["delta_best_window"] <= self.min_improvement:
            reasons.append("low_progress")
        if heuristic_risk >= self.risk_trigger:
            reasons.append("high_risk")
        if (
            metrics["iteration"] > 0
            and metrics["iteration"] % self.event_interval == 0
        ):
            reasons.append("periodic_check")

        event_active = bool(reasons)
        event_reason = "|".join(reasons) if reasons else "none"
        return event_active, event_reason

    def _run_event_analysis(self, metrics, current_features):
        enough_history = (
            metrics["iteration"] >= self.warmup_iters
            and len(self.training_X) >= self.min_training_samples
            and len(set(self.training_y)) > 1
        )

        if not enough_history:
            return self._heuristic_risk(metrics), {name: 0.0 for name in self.feature_names}

        should_retrain = (
            self.model is None
            or metrics["iteration"] - self.last_trained_iteration >= self.retrain_interval
        )
        if should_retrain:
            self.model = RandomForestClassifier(
                n_estimators=60,
                max_depth=4,
                random_state=self.random_state,
                n_jobs=1,
            )
            self.model.fit(np.array(self.training_X), np.array(self.training_y))
            self.explainer = shap.TreeExplainer(self.model)
            self.last_trained_iteration = metrics["iteration"]

        risk_score = float(self.model.predict_proba(current_features.reshape(1, -1))[0, 1])
        shap_values = self._compute_shap_values(current_features)
        self.last_event_iteration = metrics["iteration"]
        return risk_score, shap_values

    def _compute_shap_values(self, current_features):
        if self.explainer is None:
            return {name: 0.0 for name in self.feature_names}

        shap_values = self.explainer.shap_values(current_features.reshape(1, -1))
        if isinstance(shap_values, list):
            class_values = shap_values[1]
        else:
            class_values = shap_values[:, :, 1] if shap_values.ndim == 3 else shap_values

        flattened = np.asarray(class_values).reshape(-1)
        return {
            feature: float(value)
            for feature, value in zip(self.feature_names, flattened)
        }

    def _build_action(self, risk_score, shap_values, event_active):
        dominant_feature = self._dominant_positive_feature(shap_values)
        action = {
            "mode": "baseline",
            "risk_score": risk_score,
            "dominant_feature": dominant_feature,
            "alpha_scale": 1.0,
            "beta_scale": 1.0,
            "danger_scale": 1.0,
            "safety_shift": 0.0,
            "exploration_weight": 0.0,
            "partial_reset_fraction": 0.0,
        }

        if risk_score >= 0.75:
            action.update(
                {
                    "mode": "rescue",
                    "alpha_scale": 1.18,
                    "beta_scale": 0.82,
                    "danger_scale": 1.35,
                    "safety_shift": -0.30,
                    "exploration_weight": 0.22,
                    "partial_reset_fraction": 0.15 if event_active else 0.0,
                }
            )
        elif risk_score >= 0.55:
            action.update(
                {
                    "mode": "preemptive",
                    "alpha_scale": 1.08,
                    "beta_scale": 0.92,
                    "danger_scale": 1.12,
                    "safety_shift": -0.10,
                    "exploration_weight": 0.10,
                }
            )

        if dominant_feature in {"population_diversity", "diversity_ratio", "mean_distance_to_best"}:
            action["exploration_weight"] += 0.08
        if dominant_feature in {"stagnation_length", "delta_best_window"}:
            action["danger_scale"] += 0.05

        return action

    def _dominant_positive_feature(self, shap_values):
        best_feature = "none"
        best_value = 0.0
        for feature, value in shap_values.items():
            if value > best_value:
                best_feature = feature
                best_value = value
        return best_feature

    def _log_iteration(
        self,
        metrics,
        heuristic_risk,
        model_risk,
        shap_values,
        action,
        event_active,
        event_reason,
    ):
        state_row = dict(metrics)
        state_row.update(
            {
                "heuristic_risk": heuristic_risk,
                "risk_score": model_risk,
                "event_active": int(event_active),
                "event_reason": event_reason,
                "control_mode": action["mode"],
                "dominant_feature": action["dominant_feature"],
                "alpha_scale": action["alpha_scale"],
                "beta_scale": action["beta_scale"],
                "danger_scale": action["danger_scale"],
                "safety_shift": action["safety_shift"],
                "exploration_weight": action["exploration_weight"],
                "partial_reset_fraction": action["partial_reset_fraction"],
            }
        )
        self.state_log.append(state_row)

        shap_row = {"iteration": metrics["iteration"], "event_active": int(event_active)}
        shap_row.update({f"SHAP_{name}": value for name, value in shap_values.items()})
        self.shap_log.append(shap_row)
