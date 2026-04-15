import numpy as np
from initialization import initialization
from halton import hal
from levy_flight import levy_flight


def _evaluate_and_update_leaders(
    positions,
    lb,
    ub,
    objective,
    best_score,
    best_pos,
    second_score,
    second_pos,
):
    positions = np.clip(positions, lb, ub)
    fitness_values = np.zeros(positions.shape[0], dtype=float)

    for i in range(positions.shape[0]):
        fitness = float(objective(positions[i, :]))
        fitness_values[i] = fitness

        if fitness < best_score:
            best_score = fitness
            best_pos = positions[i, :].copy()

        if fitness > best_score and fitness < second_score:
            second_score = fitness
            second_pos = positions[i, :].copy()

    return positions, fitness_values, best_score, best_pos, second_score, second_pos


def _apply_diversity_rescue(
    positions,
    fitness_values,
    rescued_agents,
    lb,
    ub,
    best_pos,
    second_pos,
    rescue_mode,
):
    if rescued_agents <= 0:
        return positions, 0

    positions = positions.copy()
    protected = {int(np.argmin(fitness_values))}
    if len(fitness_values) > 1:
        protected.add(int(np.argsort(fitness_values)[1]))

    available = [idx for idx in range(len(fitness_values)) if idx not in protected]
    if rescue_mode == "random" and available:
        replace_count = min(rescued_agents, len(available))
        candidate_indices = np.random.choice(available, size=replace_count, replace=False)
    else:
        candidate_indices = [
            int(idx) for idx in np.argsort(fitness_values)[::-1] if int(idx) not in protected
        ]

    rescued = 0
    for idx in candidate_indices:
        idx = int(idx)
        positions[idx, :] = np.random.rand(positions.shape[1]) * (ub - lb) + lb
        rescued += 1
        if rescued >= rescued_agents:
            break
    return positions, rescued


def run_wo_controlled(search_agents_no, max_iter, lb, ub, dim, objective, controller):
    positions = initialization(search_agents_no, dim, ub, lb)
    convergence_curve = np.zeros(max_iter)

    ratio = 0.4
    female_count = round(search_agents_no * ratio)
    male_count = female_count
    child_count = search_agents_no - female_count - male_count
    controller.set_runtime(
        objective=objective,
        lb=lb,
        ub=ub,
        dim=dim,
        search_agents_no=search_agents_no,
        female_count=female_count,
        male_count=male_count,
        child_count=child_count,
        max_iter=max_iter,
    )

    best_score = float("inf")
    second_score = float("inf")
    best_pos = np.zeros(dim)
    second_pos = np.zeros(dim)
    gbest_x = np.tile(best_pos, (search_agents_no, 1))

    for iteration in range(max_iter):
        (
            positions,
            fitness_values,
            best_score,
            best_pos,
            second_score,
            second_pos,
        ) = _evaluate_and_update_leaders(
            positions,
            lb,
            ub,
            objective,
            best_score,
            best_pos,
            second_score,
            second_pos,
        )
        current_fitness = best_score

        base_alpha = 1 - iteration / max_iter
        base_beta = 1 - 1 / (1 + np.exp((0.5 * max_iter - iteration) / max_iter * 10))
        base_danger_signal = 2 * base_alpha * (2 * np.random.rand() - 1)
        base_safety_signal = np.random.rand()

        plan = controller.plan(
            {
                "iteration": iteration,
                "current_fitness": current_fitness,
                "alpha": float(base_alpha),
                "beta": float(base_beta),
                "danger_signal": float(base_danger_signal),
                "safety_signal": float(base_safety_signal),
                "pop_size": float(search_agents_no),
                "positions_context": positions.copy(),
                "best_score_context": float(best_score),
                "best_pos_context": best_pos.copy(),
                "second_pos_context": second_pos.copy(),
            }
        )

        rescued_agents = int(round(search_agents_no * plan["action"]["rescue_fraction"]))
        positions, rescued_count = _apply_diversity_rescue(
            positions,
            fitness_values,
            rescued_agents,
            lb,
            ub,
            best_pos,
            second_pos,
            plan["action"]["rescue_mode"],
        )

        alpha = plan["features"]["alpha"]
        beta = np.clip(plan["features"]["beta"], 0.0, 1.0)
        danger_signal = plan["features"]["danger_signal"]
        safety_signal = np.clip(plan["features"]["safety_signal"], 0.0, 1.0)

        if abs(danger_signal) >= 1:
            r3 = np.random.rand()
            p1 = np.random.permutation(search_agents_no)
            p2 = np.random.permutation(search_agents_no)
            positions = positions + (beta * r3**2) * (
                positions[p1, :] - positions[p2, :]
            )
        else:
            if safety_signal >= 0.5:
                for i in range(male_count):
                    positions[i, :] = lb + hal(i + 1, 7) * (ub - lb)

                last_male_index = male_count - 1
                for j in range(male_count, male_count + female_count):
                    positions[j, :] = positions[j, :] + alpha * (
                        positions[last_male_index, :] - positions[j, :]
                    ) + (1 - alpha) * (gbest_x[j, :] - positions[j, :])

                for i in range(search_agents_no - child_count, search_agents_no):
                    p = np.random.rand()
                    o = gbest_x[i, :] + positions[i, :] * levy_flight(dim)
                    positions[i, :] = p * (o - positions[i, :])

            if safety_signal < 0.5 and abs(danger_signal) >= 0.5:
                for i in range(search_agents_no):
                    r4 = np.random.rand()
                    positions[i, :] = positions[i, :] * danger_signal - np.abs(
                        gbest_x[i, :] - positions[i, :]
                    ) * r4**2

            if safety_signal < 0.5 and abs(danger_signal) < 0.5:
                for i in range(search_agents_no):
                    for j_dim in range(dim):
                        theta1 = np.random.rand()
                        a1 = beta * np.random.rand() - beta
                        b1 = np.tan(theta1 * np.pi)
                        x1 = best_pos[j_dim] - a1 * b1 * abs(
                            best_pos[j_dim] - positions[i, j_dim]
                        )

                        theta2 = np.random.rand()
                        a2 = beta * np.random.rand() - beta
                        b2 = np.tan(theta2 * np.pi)
                        x2 = second_pos[j_dim] - a2 * b2 * abs(
                            second_pos[j_dim] - positions[i, j_dim]
                        )
                        positions[i, j_dim] = (x1 + x2) / 2

        convergence_curve[iteration] = best_score

        controller.commit(
            plan,
            {
                "output_fitness": best_score,
                "rescued_agents": rescued_count,
            },
        )

    controller.finalize()
    return best_score, best_pos, convergence_curve
