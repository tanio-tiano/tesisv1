"""Adaptadores de problema para los runners de Walrus Optimizer.

Cada adaptador cumple el contrato ``WOProblem`` (ver ``problems/base.py``).
Construir uno desde la linea de comandos:

    from problems.factory import parse_problem_spec
    problem = parse_problem_spec("cec2022:F6", dim=10)
    problem = parse_problem_spec("tmlap:1.instancia_simple.txt")
"""

from .base import WOProblem
from .factory import parse_problem_spec

__all__ = ["WOProblem", "parse_problem_spec"]
