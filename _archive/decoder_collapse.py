"""Auditoria del decoder TMLAP: cuantifica colapso continuo->discreto.

No corre WO. Genera N posiciones en el dominio del problema, las decodifica +
repara, y reporta:

1. Numero de asignaciones discretas unicas sobre N muestras.
2. Distribucion de frecuencias de las asignaciones unicas (top-k mostrado).
3. Sensibilidad a perturbaciones epsilon: para una muestra de M agentes y un
   conjunto de epsilons, fraccion de perturbaciones que cambian la asignacion
   reparada.

Uso:

    python -m analysis.decoder_collapse \\
        --instance 3.instancia_dura.txt \\
        --n-samples 500 --n-perturb-agents 30 --seed 1234 \\
        --output experiments/decoder_audit_dura.csv
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from problems.factory import parse_problem_spec


def _decode_repair(problem, position):
    return tuple(int(h) for h in problem.repair(problem.decode(position)))


def _sample_uniform(problem, n, rng):
    span = problem.ub - problem.lb
    return rng.random((n, problem.dim)) * span + problem.lb


def audit_population_uniqueness(problem, n_samples, rng):
    positions = _sample_uniform(problem, n_samples, rng)
    assignments = [_decode_repair(problem, p) for p in positions]
    counter = Counter(assignments)
    return {
        "n_samples": int(n_samples),
        "n_unique": int(len(counter)),
        "uniqueness_ratio": float(len(counter) / max(n_samples, 1)),
        "top_frequencies": counter.most_common(10),
    }


def audit_perturbation_sensitivity(problem, n_agents, epsilons, rng):
    base_positions = _sample_uniform(problem, n_agents, rng)
    base_assignments = [_decode_repair(problem, p) for p in base_positions]
    rows = []
    span = problem.ub - problem.lb
    for eps in epsilons:
        changed = 0
        total = 0
        for agent_idx in range(n_agents):
            base_pos = base_positions[agent_idx]
            base_asg = base_assignments[agent_idx]
            for _trial in range(10):
                noise = rng.normal(0.0, eps, size=problem.dim) * span
                perturbed = np.clip(base_pos + noise, problem.lb, problem.ub)
                new_asg = _decode_repair(problem, perturbed)
                total += 1
                if new_asg != base_asg:
                    changed += 1
        rows.append(
            {
                "epsilon": float(eps),
                "n_agents": int(n_agents),
                "n_perturbations": int(total),
                "n_changed": int(changed),
                "change_ratio": float(changed / max(total, 1)),
            }
        )
    return pd.DataFrame(rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Auditoria de colapso del decoder TMLAP (continuo -> discreto)."
    )
    parser.add_argument(
        "--instance",
        required=True,
        help="Ruta al .txt de la instancia TMLAP (ej. 3.instancia_dura.txt).",
    )
    parser.add_argument("--clients", type=int, default=None)
    parser.add_argument("--hubs", type=int, default=None)
    parser.add_argument("--n-samples", type=int, default=500,
                        help="Muestras uniformes para el conteo de asignaciones unicas.")
    parser.add_argument("--n-perturb-agents", type=int, default=30,
                        help="Agentes base para el test de sensibilidad.")
    parser.add_argument(
        "--epsilons",
        type=float,
        nargs="+",
        default=[0.01, 0.05, 0.1, 0.25, 0.5],
        help="Magnitudes (en fraccion del span) para el ruido gaussiano.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--output",
        default=None,
        help="CSV de salida con la sensibilidad por epsilon (opcional).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    spec = f"tmlap:{args.instance}"
    problem = parse_problem_spec(spec, clients=args.clients, hubs=args.hubs)
    rng = np.random.default_rng(args.seed)

    print(f"== Decoder audit: {problem.name} ({problem.n_clients}c x {problem.n_hubs}h) ==")
    uniqueness = audit_population_uniqueness(problem, args.n_samples, rng)
    print(f"\n[uniqueness]")
    print(f"  muestras totales = {uniqueness['n_samples']}")
    print(f"  asignaciones unicas = {uniqueness['n_unique']}")
    print(f"  ratio = {uniqueness['uniqueness_ratio']:.4f}")
    print(f"  top-10 frecuencias:")
    for asg, freq in uniqueness["top_frequencies"]:
        print(f"    {freq:4d}x  {' '.join(str(h) for h in asg)}")

    sensitivity_df = audit_perturbation_sensitivity(
        problem, args.n_perturb_agents, args.epsilons, rng
    )
    print(f"\n[sensibilidad a perturbaciones]")
    for _, row in sensitivity_df.iterrows():
        print(
            f"  eps={row['epsilon']:.4f}  cambios={row['n_changed']}/{row['n_perturbations']} "
            f"ratio={row['change_ratio']:.4f}"
        )

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        sensitivity_df.to_csv(out, index=False)
        print(f"\nCSV de sensibilidad: {out}")


if __name__ == "__main__":
    main()
