import numpy as np


def initialization(search_agents_no, dim, ub, lb):
    boundary_no = 1
    if isinstance(ub, (list, np.ndarray)):
        boundary_no = len(ub)

    positions = np.zeros((search_agents_no, dim))

    if boundary_no == 1:
        positions = np.random.rand(search_agents_no, dim) * (ub - lb) + lb

    if boundary_no > 1:
        for i in range(dim):
            ub_i = ub[i]
            lb_i = lb[i]
            positions[:, i] = np.random.rand(search_agents_no) * (ub_i - lb_i) + lb_i

    return positions
