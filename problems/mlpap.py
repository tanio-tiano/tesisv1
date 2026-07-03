"""Adaptador MLPAP (Microhub Location and Pedestrian Assignment Problem).

Formalizado en el PDF de referencia (Modelo (1).pdf). Es una generalizacion
del CFLP que incorpora simultaneamente:

- costos fijos de apertura de hub (``f_j``) y costos operativos por unidad de
  demanda servida (``o_j``);
- capacidad maxima (``L_j``) y utilizacion minima (``mu_j``) por hub;
- distancia peatonal maxima (``D_max``) para la cobertura cliente-hub;
- cardinalidad de hubs abiertos con cotas ``P_min <= sum(y) <= P_max``;
- pesos de prioridad por cliente (``w_c``) y demanda variable (``q_c``).

Funcion objetivo (minimizacion, Ec. 1 del PDF)::

    f(z) = sum_j [f_j*y_j + o_j*sum_c q_c*x_cj]  +  sum_c sum_j w_c*d_cj*x_cj
           ^--------- costo de facilities ---------^   ^-- costo ponderado --^

Manejo de infactibilidad (Ec. 8 del PDF, penalizacion aditiva SIN repair)::

    f_tilde(z) = f(z) + pi * v(z)

donde ``v(z)`` es la suma agregada de las violaciones a (2)-(6):

- ``v_cap``: exceso sobre capacidad ``L_j``;
- ``v_util``: deficit bajo utilizacion minima ``mu_j`` (solo hubs abiertos);
- ``v_dist``: exceso sobre distancia maxima ``D_max`` en asignaciones activas;
- ``v_card``: violacion de las cotas ``P_min`` y ``P_max`` sobre ``sum(y)``.

Encoding continuo -> discreto (dim = n_clientes):

Cada posicion ``x`` es un vector real de largo ``n``, con ``x[c] in [0, m-1]``
codificando el hub asignado al cliente ``c``. El decoder redondea al entero
mas cercano y clipea al rango valido. Este encoding satisface por construccion
las Ec. (2) (asignacion unica) y (4) (coherencia con y). Las demas
restricciones se penalizan.

Kernel acelerado con Numba (``@njit`` + ``cache=True``): la primera evaluacion
compila el kernel (~1-2 s); las siguientes corren a velocidad de C. Si Numba
no esta instalado, se usa un fallback puro Python/numpy (funcionalmente
identico, mas lento).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Numba con fallback opcional. Si esta disponible, el kernel se JIT-compila.
# ---------------------------------------------------------------------------
try:  # pragma: no cover - depende del ambiente
    from numba import njit
    _NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover
    _NUMBA_AVAILABLE = False

    def njit(func=None, **_kw):  # type: ignore[no-redef]
        """Fallback: decorator sin efecto si Numba no esta instalado."""
        if func is None:
            return lambda f: f
        return func


@njit(cache=True, fastmath=False)
def _mlpap_kernel(x_idx, f, o, L, mu, q, w, d, D_max, P_min, P_max, pi):
    """Kernel: recibe asignacion entera y parametros; devuelve f + pi*v (Ec. 8).

    Tipos esperados:
    - ``x_idx``: int64[n]  (asignacion entera, ya decodificada)
    - ``f, o, L, mu``: float64[m]
    - ``q, w``: float64[n]
    - ``d``: float64[n, m]
    - escalares: ``D_max``, ``pi``: float; ``P_min``, ``P_max``: int
    """
    n = x_idx.shape[0]
    m = f.shape[0]

    load = np.zeros(m, dtype=np.float64)     # demanda agregada por hub
    assignment_cost = 0.0
    v_dist = 0.0

    # Paso 1: recorrer clientes -> agregar load, costo de asignacion y v_dist.
    for c in range(n):
        j = x_idx[c]
        load[j] += q[c]
        d_cj = d[c, j]
        assignment_cost += w[c] * d_cj
        if d_cj > D_max:
            v_dist += d_cj - D_max

    # Paso 2: recorrer hubs -> y_j, facility_cost, v_cap, v_util, cardinalidad.
    facility_cost = 0.0
    v_cap = 0.0
    v_util = 0.0
    n_open = 0
    for j in range(m):
        if load[j] > 0.0:
            n_open += 1
            facility_cost += f[j] + o[j] * load[j]
            if load[j] > L[j]:
                v_cap += load[j] - L[j]
            if load[j] < mu[j]:
                v_util += mu[j] - load[j]

    # Paso 3: violacion de cardinalidad P_min <= sum(y) <= P_max.
    v_card = 0.0
    if n_open < P_min:
        v_card += P_min - n_open
    elif n_open > P_max:
        v_card += n_open - P_max

    # Paso 4: penalizacion aditiva (Ec. 8).
    f_raw = facility_cost + assignment_cost
    v_total = v_cap + v_util + v_dist + v_card
    return f_raw + pi * v_total


# ---------------------------------------------------------------------------
# Adaptador MLPAPProblem (contrato WOProblem, mismo patron que TMLAP).
# ---------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class MLPAPProblem:
    """Adaptador MLPAP. Cumple el contrato WOProblem, inmutable."""

    name: str
    n: int                       # n_clientes
    m: int                       # n_hubs
    f: np.ndarray                # (m,) costos fijos de apertura
    o: np.ndarray                # (m,) costos operativos por unidad de demanda
    L: np.ndarray                # (m,) capacidad maxima por hub
    mu: np.ndarray               # (m,) utilizacion minima por hub
    q: np.ndarray                # (n,) demanda del cliente
    w: np.ndarray                # (n,) peso de prioridad del cliente
    d: np.ndarray                # (n, m) distancia caminando cliente-hub
    D_max: float                 # distancia maxima
    P_min: int                   # minimo hubs abiertos
    P_max: int                   # maximo hubs abiertos (P en el paper)
    pi: float                    # coeficiente de penalizacion
    family: str = "mlpap"
    optimum: object = None       # desconocido; se reporta gap = NaN si aplica
    benchmark: str = "mlpap"

    # ------------------------------------------------------------------
    # Propiedades del contrato WOProblem.
    # ------------------------------------------------------------------
    @property
    def n_clients(self):
        return int(self.n)

    @property
    def n_hubs(self):
        return int(self.m)

    @property
    def dim(self):
        return int(self.n)

    @property
    def lb(self):
        return np.zeros(self.n, dtype=float)

    @property
    def ub(self):
        return np.full(self.n, self.m - 1, dtype=float)

    # ------------------------------------------------------------------
    # Decoder + evaluate.
    # ------------------------------------------------------------------
    def decode(self, position):
        """Continuo [0, m-1]^n -> entero int64[n] (redondeo + clip)."""
        values = np.rint(np.asarray(position, dtype=float)).astype(np.int64)
        return np.clip(values, 0, self.m - 1)

    def evaluate(self, x):
        """Devuelve f_tilde(z) = f(z) + pi*v(z). Delega en el kernel Numba."""
        x_idx = self.decode(x)
        return float(_mlpap_kernel(
            x_idx, self.f, self.o, self.L, self.mu,
            self.q, self.w, self.d,
            float(self.D_max), int(self.P_min), int(self.P_max), float(self.pi),
        ))

    def initial_population(self, n_agents, rng=None, **_kwargs):
        """Muestreo uniforme en [0, m-1]^n. No gasta FES (sin on_eval)."""
        if rng is None:
            rng = np.random.default_rng()
        return rng.uniform(0.0, float(self.m - 1), size=(int(n_agents), self.n))


# ---------------------------------------------------------------------------
# Loader de instancias JSON.
# ---------------------------------------------------------------------------
def load_problem(path, **_kwargs):
    """Carga una instancia MLPAP desde un archivo JSON.

    Busqueda de ruta con fallback (mismo patron que TMLAP):
    1. Ruta tal cual (absoluta o relativa al cwd).
    2. ``<repo>/data/mlpap/<basename>``.
    3. ``<repo>/data/<basename>``.
    """
    path = Path(path)
    if not path.exists():
        repo_root = Path(__file__).resolve().parents[1]
        for fallback in (repo_root / "data" / "mlpap" / path.name,
                         repo_root / "data" / path.name):
            if fallback.exists():
                path = fallback
                break

    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)

    required = ("n", "m", "f", "o", "L", "mu", "q", "w", "d",
                "D_max", "P_min", "P_max", "pi")
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(
            f"Instancia MLPAP {path.name}: faltan campos {missing}."
        )

    n = int(data["n"])
    m = int(data["m"])
    f_arr = np.asarray(data["f"], dtype=np.float64)
    o_arr = np.asarray(data["o"], dtype=np.float64)
    L_arr = np.asarray(data["L"], dtype=np.float64)
    mu_arr = np.asarray(data["mu"], dtype=np.float64)
    q_arr = np.asarray(data["q"], dtype=np.float64)
    w_arr = np.asarray(data["w"], dtype=np.float64)
    d_arr = np.asarray(data["d"], dtype=np.float64)

    # Validaciones de forma.
    if f_arr.shape != (m,) or o_arr.shape != (m,) or L_arr.shape != (m,) or mu_arr.shape != (m,):
        raise ValueError(
            f"Instancia {path.name}: vectores por hub (f/o/L/mu) deben tener forma ({m},)."
        )
    if q_arr.shape != (n,) or w_arr.shape != (n,):
        raise ValueError(
            f"Instancia {path.name}: vectores por cliente (q/w) deben tener forma ({n},)."
        )
    if d_arr.shape != (n, m):
        raise ValueError(
            f"Instancia {path.name}: matriz d debe tener forma ({n}, {m}), tiene {d_arr.shape}."
        )

    return MLPAPProblem(
        name=str(data.get("instance_id", path.stem)),
        n=n, m=m,
        f=f_arr, o=o_arr, L=L_arr, mu=mu_arr,
        q=q_arr, w=w_arr, d=d_arr,
        D_max=float(data["D_max"]),
        P_min=int(data["P_min"]),
        P_max=int(data["P_max"]),
        pi=float(data["pi"]),
    )
