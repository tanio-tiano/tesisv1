"""Genera la instancia TMLAP "grande" usada para stress test de WO_base vs WO_shap.

Proporciones replicando la logica de las instancias pequenas (simple/mediana/dura)
pero a escala mayor:

- Ratio clientes:hubs = 2:1 (igual a mediana).
- Distancias enteras en [4, 12] (mismo rango que las chicas).
- Costos fijos enteros en [15, 25] (mismo rango).
- Capacidades enteras en [2, 3] => total ~ 1.25x clientes (slack como en chicas).
- D_max = 9 => mismo ratio D_max / mediana_distancia ~ 1.1x.

Compatible con WO_tmlap_shap/run_wo_tmlap_online_shap.py::load_problem
(`self.n_clientes`, `self.n_hubs`, `self.distancias`, `self.costos_fijos`,
`self.capacidad`, `self.D_max`).
"""

from pathlib import Path

import numpy as np


N_HUBS = 500
N_CLIENTS = 1000
SEED = 20260514
DISTANCE_RANGE = (4, 12)
COST_RANGE = (15, 25)
CAPACITY_RANGE = (2, 3)
D_MAX = 9
CAPACITY_SLACK_TARGET = 1.25

OUTPUT_PATH = Path(__file__).resolve().parent / "4.instancia_grande.txt"


def main():
    rng = np.random.default_rng(SEED)
    distances = rng.integers(DISTANCE_RANGE[0], DISTANCE_RANGE[1] + 1, size=(N_CLIENTS, N_HUBS))
    fixed_costs = rng.integers(COST_RANGE[0], COST_RANGE[1] + 1, size=N_HUBS)

    capacities = rng.integers(CAPACITY_RANGE[0], CAPACITY_RANGE[1] + 1, size=N_HUBS)
    target_total = int(np.ceil(N_CLIENTS * CAPACITY_SLACK_TARGET))
    if int(capacities.sum()) < target_total:
        deficit = target_total - int(capacities.sum())
        for _ in range(deficit):
            idx = int(rng.integers(0, N_HUBS))
            capacities[idx] += 1

    feasible_per_client = (distances <= D_MAX).sum(axis=1)
    if (feasible_per_client == 0).any():
        for client in np.where(feasible_per_client == 0)[0]:
            hub = int(np.argmin(distances[client, :]))
            distances[client, hub] = D_MAX

    def fmt_row(row):
        return "    [" + ", ".join(str(int(value)) for value in row) + "]"

    distance_lines = ",\n".join(fmt_row(row) for row in distances)
    cost_line = ", ".join(str(int(value)) for value in fixed_costs)
    capacity_line = ", ".join(str(int(value)) for value in capacities)

    text = (
        f"self.n_clientes = {N_CLIENTS}\n"
        f"self.n_hubs = {N_HUBS}\n\n"
        f"# Matriz de distancias cliente-hub ({N_CLIENTS} x {N_HUBS}), enteros en [{DISTANCE_RANGE[0]}, {DISTANCE_RANGE[1]}]\n"
        "self.distancias = [\n"
        f"{distance_lines}\n"
        "]\n\n"
        f"# Costos fijos por hub ({N_HUBS}), enteros en [{COST_RANGE[0]}, {COST_RANGE[1]}]\n"
        f"self.costos_fijos = [{cost_line}]\n\n"
        f"# Capacidad por hub ({N_HUBS}); suma total >= {target_total} (slack ~{CAPACITY_SLACK_TARGET}x sobre {N_CLIENTS} clientes)\n"
        f"self.capacidad = [{capacity_line}]\n\n"
        f"# Distancia maxima cliente-hub permitida (ratio ~1.1x sobre mediana de distancias)\n"
        f"self.D_max = {D_MAX}\n"
    )

    OUTPUT_PATH.write_text(text, encoding="utf-8")

    total_capacity = int(capacities.sum())
    median_feasible = int(np.median(feasible_per_client))
    print(f"Generada: {OUTPUT_PATH}")
    print(f"  n_clientes={N_CLIENTS}, n_hubs={N_HUBS}, ratio={N_CLIENTS / N_HUBS:.2f}:1")
    print(f"  suma_capacidades={total_capacity} -> slack={total_capacity / N_CLIENTS:.2f}x")
    print(f"  D_max={D_MAX}, mediana_hubs_factibles_por_cliente={median_feasible}/{N_HUBS}")


if __name__ == "__main__":
    main()
