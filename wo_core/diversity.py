"""Metricas de diversidad poblacional usadas por el controlador SHAP."""

import numpy as np


def population_diversity(positions):
    """Distancia media (L2) de cada agente al centroide de la poblacion."""
    centroid = np.mean(positions, axis=0)
    distances = np.linalg.norm(positions - centroid, axis=1)
    return float(np.mean(distances))


def domain_diversity_scale(lb, ub):
    """Norma L2 del span del dominio. Usado para normalizar la diversidad."""
    scale = float(np.linalg.norm(np.asarray(ub, dtype=float) - np.asarray(lb, dtype=float)))
    return max(scale, 1e-12)


def normalize_diversity_by_domain(diversity, lb, ub):
    """Diversidad dividida por el span del dominio (escala invariante)."""
    return float(diversity) / domain_diversity_scale(lb, ub)
