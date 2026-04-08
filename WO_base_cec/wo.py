import numpy as np
from initialization import initialization
from halton import hal
from levy_flight import levy_flight


def run_wo(search_agents_no, max_iter, lb, ub, dim, objective):
    best_pos = np.zeros(dim)
    second_pos = np.zeros(dim)
    best_score = float("inf")
    second_score = float("inf")

    positions = initialization(search_agents_no, dim, ub, lb)
    convergence_curve = np.zeros(max_iter)

    ratio = 0.4
    female_count = round(search_agents_no * ratio)
    male_count = female_count
    child_count = search_agents_no - female_count - male_count

    for iteration in range(max_iter):
        for i in range(positions.shape[0]):
            positions[i, :] = np.clip(positions[i, :], lb, ub)
            fitness = objective(positions[i, :])

            if fitness < best_score:
                second_score = best_score
                second_pos = best_pos.copy()
                best_score = fitness
                best_pos = positions[i, :].copy()
            elif fitness < second_score:
                second_score = fitness
                second_pos = positions[i, :].copy()

        alpha = 1 - iteration / max_iter
        beta = 1 - 1 / (1 + np.exp((0.5 * max_iter - iteration) / max_iter * 10))
        danger_signal = 2 * alpha * (2 * np.random.rand() - 1)
        safety_signal = np.random.rand()

        if abs(danger_signal) >= 1:
            p1 = np.random.permutation(search_agents_no)
            p2 = np.random.permutation(search_agents_no)
            positions = positions + (beta * np.random.rand() ** 2) * (
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

        convergence_curve[iteration] = best_score

    return best_score, best_pos, convergence_curve
