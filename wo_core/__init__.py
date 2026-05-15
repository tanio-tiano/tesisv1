"""Walrus Optimizer core: dinamica del paper de Han et al. 2024.

Modulo independiente del problema. Importar las primitivas necesarias:

    from wo_core import (
        apply_wo_movement,
        evaluate_and_update_leaders,
        iteration_signals,
        walrus_role_counts,
        enforce_bounds,
        r_signal_from_alpha_and_danger,
    )
    from wo_core.halton import halton
    from wo_core.levy_flight import levy_flight
    from wo_core.initialization import uniform_population
    from wo_core.diversity import (
        population_diversity,
        domain_diversity_scale,
        normalize_diversity_by_domain,
    )
"""

from .walrus import (
    apply_wo_movement,
    enforce_bounds,
    evaluate_and_update_leaders,
    iteration_signals,
    r_signal_from_alpha_and_danger,
    walrus_role_counts,
)
from .halton import halton
from .levy_flight import levy_flight
from .initialization import uniform_population
from .diversity import (
    domain_diversity_scale,
    normalize_diversity_by_domain,
    population_diversity,
)

__all__ = [
    "apply_wo_movement",
    "domain_diversity_scale",
    "enforce_bounds",
    "evaluate_and_update_leaders",
    "halton",
    "iteration_signals",
    "levy_flight",
    "normalize_diversity_by_domain",
    "population_diversity",
    "r_signal_from_alpha_and_danger",
    "uniform_population",
    "walrus_role_counts",
]
