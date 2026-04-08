import numpy as np
from initialization import initialization
from halton import hal
from levy_flight import levy_flight


def run_wo_controlled(search_agents_no, max_iter, lb, ub, dim, objective, controller):
    best_pos = np.zeros(dim)
    second_pos = np.zeros(dim)
    best_score = float("inf")
    second_score = float("inf")

    positions = initialization(search_agents_no, dim, ub, lb)
    initial_diversity = max(float(np.mean(np.std(positions, axis=0))), 1e-12)
    convergence_curve = np.zeros(max_iter)

    ratio = 0.4
    female_count = round(search_agents_no * ratio)
    male_count = female_count
    child_count = search_agents_no - female_count - male_count

    best_history = []
    recent_successes = []
    stagnation_length = 0
    improvement_tolerance = controller.min_improvement
    progress_window = 10
    success_window = 10

    for iteration in range(max_iter):
        fitness_values = np.zeros(search_agents_no)

        for i in range(positions.shape[0]):
            positions[i, :] = np.clip(positions[i, :], lb, ub)
            fitness = objective(positions[i, :])
            fitness_values[i] = fitness

            if fitness < best_score:
                second_score = best_score
                second_pos = best_pos.copy()
                best_score = fitness
                best_pos = positions[i, :].copy()
            elif fitness < second_score:
                second_score = fitness
                second_pos = positions[i, :].copy()

        previous_best_score = best_history[-1] if best_history else None
        instantaneous_improvement = 0.0
        if previous_best_score is not None and np.isfinite(previous_best_score):
            instantaneous_improvement = max(previous_best_score - best_score, 0.0)

        if instantaneous_improvement > improvement_tolerance:
            stagnation_length = 0
        else:
            stagnation_length += 1

        best_history.append(float(best_score))
        start_idx = max(0, len(best_history) - progress_window - 1)
        delta_best_window = max(best_history[start_idx] - best_score, 0.0)

        recent_successes.append(1 if instantaneous_improvement > improvement_tolerance else 0)
        if len(recent_successes) > success_window:
            recent_successes.pop(0)

        population_diversity = float(np.mean(np.std(positions, axis=0)))
        diversity_ratio = population_diversity / initial_diversity
        mean_distance_to_best = float(
            np.mean(np.linalg.norm(positions - best_pos, axis=1)) / np.sqrt(dim)
        )
        success_rate_recent = float(np.mean(recent_successes)) if recent_successes else 0.0

        metrics = {
            "iteration": iteration,
            "best_score": float(best_score),
            "mean_fitness": float(np.mean(fitness_values)),
            "fitness_std": float(np.std(fitness_values)),
            "delta_best_window": float(delta_best_window),
            "stagnation_length": int(stagnation_length),
            "population_diversity": population_diversity,
            "diversity_ratio": float(diversity_ratio),
            "mean_distance_to_best": mean_distance_to_best,
            "success_rate_recent": success_rate_recent,
        }

        action = controller.update(metrics)

        alpha = (1 - iteration / max_iter) * action["alpha_scale"]
        beta = (
            1 - 1 / (1 + np.exp((0.5 * max_iter - iteration) / max_iter * 10))
        ) * action["beta_scale"]
        danger_signal = (
            2 * alpha * (2 * np.random.rand() - 1) * action["danger_scale"]
        )
        safety_signal = np.clip(np.random.rand() + action["safety_shift"], 0.0, 1.0)

        if abs(danger_signal) >= 1:
            p1 = np.random.permutation(search_agents_no)
            p2 = np.random.permutation(search_agents_no)
            positions = positions + (beta * np.random.rand() ** 2) * (
                positions[p1, :] - positions[p2, :]
            )
            if action["exploration_weight"] > 0:
                positions = positions + action["exploration_weight"] * np.random.randn(
                    *positions.shape
                ) * (ub - lb) / 10
        else:
            if safety_signal >= 0.5:
                for i in range(male_count):
                    positions[i, :] = lb + hal(i + 1, 7) * (ub - lb)

                last_male_index = male_count - 1
                for j in range(male_count, male_count + female_count):
                    positions[j, :] = positions[j, :] + alpha * (
                        positions[last_male_index, :] - positions[j, :]
                    ) + (1 - alpha) * (best_pos - positions[j, :])

                for i in range(search_agents_no - child_count, search_agents_no):
                    positions[i, :] = np.random.rand() * (
                        best_pos + positions[i, :] * levy_flight(dim) - positions[i, :]
                    )
            else:
                if abs(danger_signal) >= 0.5:
                    for i in range(search_agents_no):
                        positions[i, :] = positions[i, :] * (2 * np.random.rand() - 1) - np.abs(
                            best_pos - positions[i, :]
                        ) * (np.random.rand() ** 2)
                else:
                    for i in range(search_agents_no):
                        for j_dim in range(dim):
                            x1 = best_pos[j_dim] - (
                                beta * np.random.rand() - beta
                            ) * np.tan(np.random.rand() * np.pi) * np.abs(
                                best_pos[j_dim] - positions[i, j_dim]
                            )
                            x2 = second_pos[j_dim] - (
                                beta * np.random.rand() - beta
                            ) * np.tan(np.random.rand() * np.pi) * np.abs(
                                second_pos[j_dim] - positions[i, j_dim]
                            )
                            positions[i, j_dim] = (x1 + x2) / 2

                if action["exploration_weight"] > 0:
                    positions = positions + action["exploration_weight"] * levy_flight(dim)

        if action["partial_reset_fraction"] > 0:
            reset_count = max(1, int(search_agents_no * action["partial_reset_fraction"]))
            worst_indices = np.argsort(fitness_values)[-reset_count:]
            new_positions = initialization(reset_count, dim, ub, lb)
            positions[worst_indices, :] = new_positions

        convergence_curve[iteration] = best_score

    controller.finalize()
    return best_score, best_pos, convergence_curve
