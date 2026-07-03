"""Dinamica del Walrus Optimizer (Han et al. 2024).

Implementacion unica, independiente del problema. Soporta:

- 4 regimenes de movimiento (eq. 9, 12-14, 17, 18-21 del paper).
- Senales temporales alpha, beta, danger_signal, safety_signal (eq. 4-8, 11).
- Roles macho/hembra/cria 45 / 45 / 10 (paper §3.2.1).
- ``gbest_x`` actualizado con X_best vigente cada iteracion (paper fiel).
- ``rng`` opcional (``np.random.Generator``). Si None, usa ``np.random`` global.

DESVIACIONES CONSCIENTES RESPECTO AL MATLAB OFICIAL
---------------------------------------------------
El codigo MATLAB de Han et al. (``WO.m``) tiene 3 discrepancias con el texto
del paper. Esta implementacion sigue el paper en los 3 casos:

1. **Proporcion de roles**. MATLAB usa ``P=0.4`` (40/40/20). El paper §3.2.1
   dice 90% adultos + 10% juveniles con M:F=1:1, i.e. 45/45/10. Aqui
   ``walrus_role_counts`` usa P=0.45.

2. **Indice de macho en update de hembra (eq. 12)**. MATLAB:
   ``X(j,:) = X(j,:)+Alpha*(X(i,:)-X(j,:))+...`` reutiliza el ``i`` filtrado
   del loop anterior de machos, por lo que TODAS las hembras se actualizan
   contra el ULTIMO macho. Aqui cada hembra ``j`` usa su macho correspondiente
   ``male_index = min(j - male_count, male_count - 1)`` (ver ``apply_wo_movement``).

3. **Refresco de ``GBestX`` (eq. 12, 13-14, 17)**. MATLAB inicializa
   ``GBestX=repmat(Best_Pos,SearchAgents_no,1)`` con ``Best_Pos=zeros`` y
   NUNCA reasigna ``GBestX`` dentro del ``while``, asi que ``GBestX`` queda
   permanentemente en ceros. Aqui los runners reconstruyen
   ``gbest_x = np.tile(best_pos, (n_agents, 1))`` cada iteracion tras
   ``evaluate_and_update_leaders``.

Para reproducir EXACTAMENTE el MATLAB oficial (no el paper), cambiar P=0.45
a 0.4 en ``walrus_role_counts`` y usar ``best_pos`` inicial en ``gbest_x``
sin refrescarlo. No se recomienda; los 3 fixes son paper-faithful.
"""

import numpy as np

from .halton import halton
from .initialization import uniform_population
from .levy_flight import levy_flight


def _rand(rng, size=None):
    if rng is None:
        return np.random.rand() if size is None else np.random.rand(size)
    return rng.random() if size is None else rng.random(size=size)


def _permutation(n, rng):
    if rng is None:
        return np.random.permutation(n)
    return rng.permutation(n)


def walrus_role_counts(n_agents):
    """Devuelve (machos, hembras, crias) siguiendo el paper Han 2024 (45/45/10).

    Paper §3.2.1: "Walrus populations are divided into adults and juveniles,
    which account for 90% and 10% of the population, respectively. Among adult
    walruses, the male to female ratio is 1:1." Es decir 45% machos, 45%
    hembras, 10% crias. Aqui se implementa con P=0.45 (proporcion de hembras),
    F = round(N*0.45), M = F, C = N - 2F. Para N=100 da (45, 45, 10) exacto.

    DESVIACION CONSCIENTE DEL MATLAB OFICIAL: el codigo MATLAB de Han et al.
    usa P=0.4 (proporcion 40/40/20), inconsistente con el texto del paper. Si
    quieres reproducir EXACTAMENTE las corridas del MATLAB oficial, cambia P
    a 0.4. Si quieres reproducir el paper, deja P=0.45 (este default).
    """
    if n_agents < 3:
        females = n_agents // 2
        males = n_agents - females
        return males, females, 0

    females = int(round(n_agents * 0.45))
    males = females
    children = n_agents - males - females
    if children < 0:
        # Salvaguarda numerica (no deberia disparar con round()).
        children = 0
        females = n_agents // 2
        males = n_agents - females
    return males, females, children


def enforce_bounds(positions, lb, ub):
    """Proyecta cada agente al rectangulo [lb, ub]."""
    return np.clip(positions, lb, ub)


def evaluate_and_update_leaders(
    positions, lb, ub, objective, best_score, best_pos, second_score, second_pos,
    budget=None,
):
    """Evalua la poblacion proyectada y actualiza X_best/X_second (elitista).

    ``objective`` es una funcion ``np.ndarray -> float`` (la del problema, o un
    wrapper que cuenta FES via ``wo_core.fes.counting_objective``).

    Si se pasa ``budget`` (``wo_core.fes.FESBudget``), la evaluacion es
    **consciente del presupuesto**: evalua agente por agente y corta en cuanto
    ``budget.exhausted()`` es verdadero, garantizando la parada **exacta** en
    MaxFES. Los agentes no evaluados quedan con fitness ``+inf`` (no compiten por
    liderazgo). Con ``budget=None`` el comportamiento es el de siempre (evalua
    toda la poblacion).

    Retorna ``(positions_proj, fitness_values, best_score, best_pos,
    second_score, second_pos)``.
    """
    positions = np.clip(positions, lb, ub)
    fitness_values = np.full(positions.shape[0], np.inf, dtype=float)
    for i in range(positions.shape[0]):
        if budget is not None and budget.exhausted():
            break
        fitness = float(objective(positions[i, :]))
        fitness_values[i] = fitness
        if fitness < best_score:
            best_score = fitness
            best_pos = positions[i, :].copy()
        if fitness > best_score and fitness < second_score:
            second_score = fitness
            second_pos = positions[i, :].copy()
    return positions, fitness_values, best_score, best_pos, second_score, second_pos


def iteration_signals(iteration, max_iter, rng=None):
    """Calcula las senales del WO en la iteracion ``iteration`` de ``max_iter``.

    Implementacion literal de las Eq. 4-8 y 11 del paper Han 2024. Cada variable
    aparece en el cuerpo con el mismo nombre y mismo orden que en el paper y
    en el ``WO.m`` oficial, sin inlineados ni renombramientos.

    Retorna ``(alpha, beta, A, R, danger_signal, safety_signal)``.
    Consume 2 numeros aleatorios del rng (r1 y r2).
    """
    alpha = 1 - iteration / max(max_iter, 1)                                        # Eq. 5
    beta = 1 - 1 / (1 + np.exp((0.5 * max_iter - iteration) / max(max_iter, 1) * 10))  # Eq. 11
    A = 2 * alpha                                                                   # Eq. 6
    r1 = _rand(rng)                                                                 # r1 ~ U(0, 1)
    R = 2 * r1 - 1                                                                  # Eq. 7 (signado en [-1, 1])
    danger_signal = A * R                                                           # Eq. 4
    r2 = _rand(rng)                                                                 # r2 ~ U(0, 1)
    safety_signal = r2                                                              # Eq. 8
    return float(alpha), float(beta), float(A), float(R), float(danger_signal), float(safety_signal)


def apply_wo_movement(
    positions,
    lb,
    ub,
    dim,
    n_agents,
    male_count,
    female_count,
    child_count,
    best_pos,
    second_pos,
    gbest_x,
    alpha,
    beta,
    R,
    danger_signal,
    safety_signal,
    rng=None,
):
    """Aplica el regimen del WO segun (danger_signal, safety_signal).

    Modifica ``positions`` in place y devuelve la version proyectada al dominio.
    Sigue exactamente las 4 ramas del paper:

    1. ``|danger| >= 1``               -> exploracion por diferencias (eq. 9).
    2. ``|danger| < 1 y safety >= 0.5``-> reproduccion: machos Halton, hembras
       hacia macho+gbest, crias con vuelo de Levy (eq. 12-14).
    3. ``safety < 0.5 y |danger| >= 0.5`` -> contraccion guiada (eq. 17).
    4. ``safety < 0.5 y |danger| < 0.5`` -> explotacion por dos lideres (eq. 18-21).
    """
    lb_arr = np.asarray(lb, dtype=float)
    ub_arr = np.asarray(ub, dtype=float)

    if abs(danger_signal) >= 1:
        r3 = _rand(rng)
        p1 = _permutation(n_agents, rng)
        p2 = _permutation(n_agents, rng)
        positions = positions + (beta * r3**2) * (positions[p1, :] - positions[p2, :])
        return enforce_bounds(positions, lb_arr, ub_arr)

    if safety_signal >= 0.5:
        # Machos: secuencia de Halton.
        for i in range(male_count):
            positions[i, :] = lb_arr + halton(i + 1, 7) * (ub_arr - lb_arr)

        # Hembras: combinacion macho + gbest.
        for j in range(male_count, male_count + female_count):
            male_index = min(j - male_count, male_count - 1)
            positions[j, :] = (
                positions[j, :]
                + alpha * (positions[male_index, :] - positions[j, :])
                + (1 - alpha) * (gbest_x[j, :] - positions[j, :])
            )

        # Crias: vuelo de Levy alrededor de gbest.
        for i in range(n_agents - child_count, n_agents):
            p = _rand(rng)
            o = gbest_x[i, :] + positions[i, :] * levy_flight(dim, rng)
            positions[i, :] = p * (o - positions[i, :])
        positions = enforce_bounds(positions, lb_arr, ub_arr)
        return positions

    if abs(danger_signal) >= 0.5:
        # Eq. 17: X^(t+1)_i,j = X^t_i,j * R - |X^t_best - X^t_i,j| * r4^2
        for i in range(n_agents):
            r4 = _rand(rng)
            positions[i, :] = (
                positions[i, :] * R
                - np.abs(gbest_x[i, :] - positions[i, :]) * r4**2
            )
        return enforce_bounds(positions, lb_arr, ub_arr)

    # safety < 0.5 y |danger| < 0.5: explotacion por dos lideres (eq. 18-21).
    for i in range(n_agents):
        for j_dim in range(dim):
            theta1 = _rand(rng)
            a1 = beta * _rand(rng) - beta
            b1 = np.tan(theta1 * np.pi)
            x1 = best_pos[j_dim] - a1 * b1 * abs(best_pos[j_dim] - positions[i, j_dim])

            theta2 = _rand(rng)
            a2 = beta * _rand(rng) - beta
            b2 = np.tan(theta2 * np.pi)
            x2 = second_pos[j_dim] - a2 * b2 * abs(second_pos[j_dim] - positions[i, j_dim])
            positions[i, j_dim] = (x1 + x2) / 2
    return enforce_bounds(positions, lb_arr, ub_arr)


def agent_role(index, male_count, female_count, child_count):
    """Rol del agente ``index`` segun el particionado de roles del WO.

    Machos: ``[0, male_count)``; hembras: ``[male_count, male_count+female_count)``;
    crias: ``[n - child_count, n)``. Devuelve ``'male'`` / ``'female'`` / ``'child'``.
    """
    if index < male_count:
        return "male"
    if index < male_count + female_count:
        return "female"
    return "child"


def apply_wo_movement_single(
    positions,
    index,
    lb,
    ub,
    dim,
    n_agents,
    male_count,
    female_count,
    child_count,
    best_pos,
    second_pos,
    gbest_row,
    alpha,
    beta,
    R,
    danger_signal,
    safety_signal,
    rng=None,
):
    """Aplica el regimen del WO a UN SOLO agente (la fila ``index``).

    Reproduce exactamente las 4 ramas de ``apply_wo_movement`` restringidas al
    agente ``index``, leyendo el resto de ``positions`` como **poblacion
    congelada** (para los regimenes acoplados: migracion usa pares aleatorios;
    reproduccion usa el macho correspondiente). ``gbest_row`` es la fila de
    ``GBestX`` para este agente (tipicamente ``best_pos``).

    No muta ``positions``; devuelve el nuevo vector de posicion del agente
    ``index`` ya proyectado a ``[lb, ub]``. Es la primitiva que usa el simulador
    SHAP por agente (``wo_core.agent_sim``).
    """
    lb_arr = np.asarray(lb, dtype=float)
    ub_arr = np.asarray(ub, dtype=float)
    x_i = positions[index, :].astype(float)

    # Regimen 1: migracion / exploracion por diferencias (Eq. 9-10).
    if abs(danger_signal) >= 1:
        r3 = _rand(rng)
        a = _randint_below(n_agents, rng)
        b = _randint_below(n_agents, rng)
        new_i = x_i + (beta * r3 ** 2) * (positions[a, :] - positions[b, :])
        return np.clip(new_i, lb_arr, ub_arr)

    # Regimen 2: reproduccion (Eq. 12-14), depende del rol del agente.
    if safety_signal >= 0.5:
        role = agent_role(index, male_count, female_count, child_count)
        if role == "male":
            new_i = lb_arr + halton(index + 1, 7) * (ub_arr - lb_arr)
        elif role == "female":
            i_male = min(index - male_count, max(male_count - 1, 0))
            new_i = (
                x_i
                + alpha * (positions[i_male, :] - x_i)
                + (1 - alpha) * (gbest_row - x_i)
            )
        else:  # child
            p = _rand(rng)
            o = gbest_row + x_i * levy_flight(dim, rng)
            new_i = p * (o - x_i)
        return np.clip(new_i, lb_arr, ub_arr)

    # Regimen 3: huida guiada por un solo lider (Eq. 17).
    if abs(danger_signal) >= 0.5:
        r4 = _rand(rng)
        new_i = x_i * R - np.abs(gbest_row - x_i) * r4 ** 2
        return np.clip(new_i, lb_arr, ub_arr)

    # Regimen 4: explotacion por dos lideres (Eq. 18-21).
    new_i = x_i.copy()
    best_arr = np.asarray(best_pos, dtype=float)
    second_arr = np.asarray(second_pos, dtype=float)
    for j_dim in range(dim):
        theta1 = _rand(rng)
        a1 = beta * _rand(rng) - beta
        b1 = np.tan(theta1 * np.pi)
        x1 = best_arr[j_dim] - a1 * b1 * abs(best_arr[j_dim] - x_i[j_dim])

        theta2 = _rand(rng)
        a2 = beta * _rand(rng) - beta
        b2 = np.tan(theta2 * np.pi)
        x2 = second_arr[j_dim] - a2 * b2 * abs(second_arr[j_dim] - x_i[j_dim])
        new_i[j_dim] = (x1 + x2) / 2
    return np.clip(new_i, lb_arr, ub_arr)


def _randint_below(high, rng):
    if rng is None:
        return int(np.random.randint(0, high))
    return int(rng.integers(0, high))


def r_signal_from_alpha_and_danger(alpha, danger_signal):
    """Despeja ``r_signal`` de ``danger_signal = 2 * alpha * r`` (convencion signada).

    Usado por la value function de Shapley para reconstruir r_signal a partir
    de la coalicion (danger_signal puede venir de la coalicion o del baseline).
    """
    alpha = float(alpha)
    danger_signal = float(danger_signal)
    if not np.isfinite(alpha) or abs(alpha) <= 1e-12:
        return 0.0
    return float(np.clip(danger_signal / (2.0 * alpha), -1.0, 1.0))


def wo(SearchAgents_no, Max_iter, lb, ub, dim, fobj, rng=None):
    """Walrus Optimizer - espejo linea-por-linea del MATLAB ``WO.m`` con 3 fixes.

    Firma identica al ``WO.m`` original. Retorna ``(Best_Score, Best_Pos,
    Convergence_curve)`` igual que MATLAB. El cuerpo reproduce la estructura,
    los nombres de variables (con casing MATLAB: ``Best_Pos``, ``GBestX``,
    ``Alpha``, ``Beta``, ``Danger_signal``, ``Safety_signal``, ``F_number``,
    ``M_number``, ``C_number``, ``Migration_step``...) y el flujo de control
    (``while`` + 4 ramas anidadas) tal cual el archivo MATLAB.

    DIVERGENCIAS RESPECTO A ``WO.m``: las 3 lineas que fixean los bugs del
    MATLAB llevan el marcador ``# FIX vs WO.m:`` con explicacion en sitio. Son
    los 3 puntos donde el MATLAB se aparta del texto del paper Han 2024:

    1. Proporcion de roles 45/45/10 (paper §3.2.1) en vez de 40/40/20.
    2. Hembra ``j`` usa SU macho ``j - M_number`` (no el ultimo, bug del loop).
    3. ``GBestX`` refrescado a ``Best_Pos`` cada iteracion (no queda en ceros).

    Cambios estilisticos no-fix (no afectan resultados):

    - ``Safety_signal`` en vez del typo ``Satey_signal``.
    - ``P_juv`` para el factor de distress de las crias, en vez de reusar la
      variable ``P`` (que en MATLAB se sombrea desde la proporcion 0.4 a un
      ``rand``, lo que es valido pero confuso).
    - ``np.clip`` en vez del clip flag-based del MATLAB (matematicamente
      identico, mas legible).

    Esta funcion es la implementacion de referencia "MATLAB-fiel" para uso
    vainilla (sin intervencion SHAP). Las primitivas
    ``iteration_signals`` / ``apply_wo_movement`` se mantienen separadas para
    el controlador SHAP, que necesita acceso por iteracion al estado interno.
    """
    lb_arr = np.asarray(lb, dtype=float)
    ub_arr = np.asarray(ub, dtype=float)

    # Initialize Best_pos and Second_pos
    Best_Pos = np.zeros(dim)
    Second_Pos = np.zeros(dim)
    Best_Score = np.inf
    Second_Score = np.inf
    GBestX = np.tile(Best_Pos, (SearchAgents_no, 1))

    # Initialize the positions of search agents
    X = uniform_population(SearchAgents_no, dim, lb, ub, rng)

    Convergence_curve = np.zeros(Max_iter)

    # FIX vs WO.m (1/3): P=0.45 (paper §3.2.1: 90% adultos + 10% juveniles,
    # ratio M:F=1:1 -> 45/45/10) en vez de P=0.4 (40/40/20) del MATLAB oficial.
    P = 0.45                                          # Proportion of females
    F_number = int(round(SearchAgents_no * P))        # Number of females
    M_number = F_number                               # Males equal to females
    C_number = SearchAgents_no - F_number - M_number  # Number of children

    for t in range(Max_iter):
        for i in range(SearchAgents_no):
            # Check boundaries (clip a [lb, ub])
            X[i, :] = np.clip(X[i, :], lb_arr, ub_arr)

            fitness = float(fobj(X[i, :]))  # Calculate objective function

            if fitness < Best_Score:
                Best_Score = fitness
                Best_Pos = X[i, :].copy()  # Update Best_pos

            if fitness > Best_Score and fitness < Second_Score:
                Second_Score = fitness
                Second_Pos = X[i, :].copy()  # Update Second_pos

        # FIX vs WO.m (3/3): refrescar GBestX cada iteracion con Best_Pos vigente.
        # MATLAB hace ``GBestX = repmat(Best_Pos, N, 1)`` UNA vez antes del while,
        # cuando ``Best_Pos`` aun es ``zeros``, y NUNCA lo reasigna dentro del
        # while -> en MATLAB GBestX queda ≡ 0 toda la corrida, asi que las Eq. 12,
        # 13-14 y 17 operan contra el origen en vez de contra el mejor real.
        GBestX = np.tile(Best_Pos, (SearchAgents_no, 1))

        Alpha = 1 - t / Max_iter
        Beta = 1 - 1 / (1 + np.exp((0.5 * Max_iter - t) / Max_iter * 10))
        A = 2 * Alpha  # A decreases linearly from 2 to 0
        r1 = _rand(rng)
        R = 2 * r1 - 1
        Danger_signal = A * R
        r2 = _rand(rng)
        Safety_signal = r2  # (typo ``Satey_signal`` del MATLAB corregido)

        if abs(Danger_signal) >= 1:
            # Migracion (Eq. 9-10)
            r3 = _rand(rng)
            Rs = SearchAgents_no
            Migration_step = (Beta * r3 ** 2) * (
                X[_permutation(Rs, rng), :] - X[_permutation(Rs, rng), :]
            )
            X = X + Migration_step

        elif abs(Danger_signal) < 1:
            if Safety_signal >= 0.5:
                # Reproduccion - machos: redistribucion por Halton
                base = 7
                for i in range(M_number):
                    X[i, :] = lb_arr + halton(i + 1, base) * (ub_arr - lb_arr)

                # Reproduccion - hembras (Eq. 12)
                for j in range(M_number, M_number + F_number):
                    # FIX vs WO.m (2/3): macho de referencia = ``j - M_number``
                    # (la hembra j usa SU macho correspondiente). En MATLAB la
                    # linea es ``X(j,:) = X(j,:)+Alpha*(X(i,:)-X(j,:))+...`` y
                    # reutiliza el ``i`` filtrado del loop anterior de machos,
                    # que al salir vale ``M_number`` -> TODAS las hembras se
                    # actualizan contra el ULTIMO macho.
                    i_male = min(j - M_number, M_number - 1)
                    X[j, :] = (
                        X[j, :]
                        + Alpha * (X[i_male, :] - X[j, :])
                        + (1 - Alpha) * (GBestX[j, :] - X[j, :])
                    )

                # Reproduccion - crias (Eq. 13-14)
                for i in range(SearchAgents_no - C_number, SearchAgents_no):
                    P_juv = _rand(rng)  # MATLAB lo llama ``P`` (sombrea la P=0.4)
                    o = GBestX[i, :] + X[i, :] * levy_flight(dim, rng)
                    X[i, :] = P_juv * (o - X[i, :])

            if Safety_signal < 0.5 and abs(Danger_signal) >= 0.5:
                # Huida con un solo lider (Eq. 17)
                for i in range(SearchAgents_no):
                    r4 = _rand(rng)
                    X[i, :] = X[i, :] * R - np.abs(GBestX[i, :] - X[i, :]) * r4 ** 2

            if Safety_signal < 0.5 and abs(Danger_signal) < 0.5:
                # Agrupacion con dos lideres (Eq. 18-21)
                for i in range(SearchAgents_no):
                    for j in range(dim):
                        theta1 = _rand(rng)
                        a1 = Beta * _rand(rng) - Beta
                        b1 = np.tan(theta1 * np.pi)
                        X1 = Best_Pos[j] - a1 * b1 * abs(Best_Pos[j] - X[i, j])

                        theta2 = _rand(rng)
                        a2 = Beta * _rand(rng) - Beta
                        b2 = np.tan(theta2 * np.pi)
                        X2 = Second_Pos[j] - a2 * b2 * abs(Second_Pos[j] - X[i, j])

                        X[i, j] = (X1 + X2) / 2

        Convergence_curve[t] = Best_Score

    return Best_Score, Best_Pos, Convergence_curve

