import numpy as np
from opfunu.cec_based import cec2022


class CECProblem:
    def __init__(self, function_id, dim=10):
        self.problem = getattr(cec2022, f"F{function_id}2022")(ndim=dim)
        self.lb = (
            self.problem.lb[0]
            if np.all(self.problem.lb == self.problem.lb[0])
            else self.problem.lb
        )
        self.ub = (
            self.problem.ub[0]
            if np.all(self.problem.ub == self.problem.ub[0])
            else self.problem.ub
        )
        self.dim = dim

    def evaluate(self, x):
        return self.problem.evaluate(x)
