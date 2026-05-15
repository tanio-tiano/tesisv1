"""Construye una instancia ``WOProblem`` a partir de una spec corta.

Especificacion CLI: ``--problem <familia>:<target>``.

Familias soportadas:

- ``cec2022:Fk`` o ``cec2022:k`` -> ``CECProblem(k, dim)``.
- ``tmlap:<path>`` -> ``load_problem(path)``.

Parametros opcionales adicionales se pasan via kwargs del runner
(p.ej. ``dim`` para CEC, ``clients`` / ``hubs`` para TMLAP).
"""


def parse_problem_spec(spec, *, dim=10, clients=None, hubs=None):
    family, _, target = str(spec).partition(":")
    family = family.strip().lower()
    target = target.strip()
    if not family:
        raise ValueError("Spec invalida; usa 'cec2022:F6' o 'tmlap:path/instancia.txt'.")

    if family == "cec2022":
        if not target:
            raise ValueError("CEC: se requiere una funcion, p.ej. cec2022:F6.")
        from .cec2022 import CECProblem
        return CECProblem(target, dim=dim)

    if family == "tmlap":
        if not target:
            raise ValueError("TMLAP: se requiere una ruta, p.ej. tmlap:1.instancia_simple.txt.")
        from .tmlap import load_problem
        return load_problem(target, clients=clients, hubs=hubs)

    raise ValueError(f"Familia de problema desconocida: {family!r}.")
