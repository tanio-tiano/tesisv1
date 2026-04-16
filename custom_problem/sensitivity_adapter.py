from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SensitivityHubProblem:
    name: str
    n_clientes: int
    n_hubs: int
    distancias: tuple[tuple[int, ...], ...]
    costos_fijos: tuple[int, ...]
    capacidad: tuple[int, ...]
    d_max: int
    sensitivity_profile: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_clientes": self.n_clientes,
            "n_hubs": self.n_hubs,
            "distancias": [list(row) for row in self.distancias],
            "costos_fijos": list(self.costos_fijos),
            "capacidad": list(self.capacidad),
            "D_max": self.d_max,
            "sensitivity_profile": dict(self.sensitivity_profile),
            "total_capacidad": self.total_capacidad,
            "capacidad_por_cliente": self.capacidad_por_cliente,
        }

    @property
    def total_capacidad(self) -> int:
        return int(sum(self.capacidad))

    @property
    def capacidad_por_cliente(self) -> float:
        return float(self.total_capacidad / self.n_clientes)

    def validate(self) -> None:
        if len(self.distancias) != self.n_clientes:
            raise ValueError(
                f"{self.name}: numero de filas de distancias invalido "
                f"({len(self.distancias)} != {self.n_clientes})."
            )

        for row in self.distancias:
            if len(row) != self.n_hubs:
                raise ValueError(
                    f"{self.name}: numero de columnas de distancias invalido "
                    f"({len(row)} != {self.n_hubs})."
                )

        if len(self.costos_fijos) != self.n_hubs:
            raise ValueError(
                f"{self.name}: costos_fijos debe tener largo {self.n_hubs}."
            )

        if len(self.capacidad) != self.n_hubs:
            raise ValueError(
                f"{self.name}: capacidad debe tener largo {self.n_hubs}."
            )


def _build_soft_instance() -> SensitivityHubProblem:
    instance = SensitivityHubProblem(
        name="soft",
        n_clientes=6,
        n_hubs=3,
        distancias=(
            (4, 6, 5),
            (7, 5, 6),
            (6, 4, 7),
            (5, 7, 6),
            (6, 5, 4),
            (7, 6, 5),
        ),
        costos_fijos=(10, 12, 11),
        capacidad=(3, 3, 3),
        d_max=8,
        sensitivity_profile={
            "difficulty_rank": 1,
            "distance_pressure": "low",
            "capacity_pressure": "low",
            "fixed_cost_pressure": "low",
            "feasibility_margin": "high",
            "source": "user_defined_baseline_soft",
        },
    )
    instance.validate()
    return instance


def _build_medium_instance() -> SensitivityHubProblem:
    instance = SensitivityHubProblem(
        name="medium",
        n_clientes=12,
        n_hubs=5,
        distancias=(
            (6, 9, 7, 10, 8),
            (8, 5, 9, 7, 11),
            (7, 8, 6, 9, 10),
            (9, 6, 8, 10, 7),
            (10, 7, 9, 6, 8),
            (6, 10, 7, 8, 9),
            (8, 9, 10, 7, 6),
            (7, 6, 8, 9, 10),
            (9, 8, 6, 7, 11),
            (10, 7, 8, 9, 6),
            (6, 8, 9, 10, 7),
            (7, 9, 6, 8, 10),
        ),
        costos_fijos=(18, 22, 20, 19, 21),
        capacidad=(5, 5, 4, 5, 5),
        d_max=9,
        sensitivity_profile={
            "difficulty_rank": 2,
            "distance_pressure": "medium",
            "capacity_pressure": "medium",
            "fixed_cost_pressure": "medium",
            "feasibility_margin": "medium",
            "source": "user_defined_baseline_medium",
        },
    )
    instance.validate()
    return instance


def _expand_hard_distances(
    medium_distances: tuple[tuple[int, ...], ...],
    target_clients: int,
    target_hubs: int,
) -> tuple[tuple[int, ...], ...]:
    base = np.asarray(medium_distances, dtype=int)
    rows, cols = base.shape
    expanded = np.zeros((target_clients, target_hubs), dtype=int)

    for i in range(target_clients):
        base_row = base[i % rows]
        for j in range(target_hubs):
            anchor = int(base_row[j % cols])
            layer_penalty = (i // rows) + (j // cols)
            asymmetry = abs((i % rows) - (j % cols)) % 3
            congestion = 1 if (i + j) % 4 == 0 else 0
            expanded[i, j] = anchor + 1 + layer_penalty + asymmetry + congestion

    return tuple(tuple(int(value) for value in row) for row in expanded.tolist())


def _build_hard_instance() -> SensitivityHubProblem:
    medium = _build_medium_instance()
    hard_distances = _expand_hard_distances(
        medium_distances=medium.distancias,
        target_clients=24,
        target_hubs=8,
    )

    instance = SensitivityHubProblem(
        name="hard",
        n_clientes=24,
        n_hubs=8,
        distancias=hard_distances,
        costos_fijos=(26, 28, 27, 29, 30, 31, 28, 27),
        capacidad=(4, 4, 4, 4, 4, 4, 4, 4),
        d_max=8,
        sensitivity_profile={
            "difficulty_rank": 3,
            "distance_pressure": "high",
            "capacity_pressure": "high",
            "fixed_cost_pressure": "high",
            "feasibility_margin": "low",
            "source": "deterministic_sensitivity_escalation_from_medium",
            "construction_rule": (
                "Hard se genera elevando distancias, costos fijos y presion "
                "de capacidad a partir del perfil medium."
            ),
        },
    )
    instance.validate()
    return instance


PROBLEM_INSTANCES: dict[str, SensitivityHubProblem] = {
    "soft": _build_soft_instance(),
    "medium": _build_medium_instance(),
    "hard": _build_hard_instance(),
}


def get_problem_instance(level: str) -> SensitivityHubProblem:
    key = level.strip().lower()
    if key not in PROBLEM_INSTANCES:
        valid = ", ".join(PROBLEM_INSTANCES.keys())
        raise ValueError(f"Nivel no soportado: {level}. Use uno de: {valid}.")
    return PROBLEM_INSTANCES[key]


def list_problem_instances() -> list[dict[str, Any]]:
    return [instance.to_dict() for instance in PROBLEM_INSTANCES.values()]

