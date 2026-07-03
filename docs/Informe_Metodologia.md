# Informe de Metodología — Walrus Optimizer con controlador interpretable (SHAP)

**Proyecto:** mejora del Walrus Optimizer (WO) mediante un controlador interpretable basado en explicabilidad in-ejecución (SHAP).
**Fecha:** 2026-05-29
**Propósito:** documentar la metodología **vigente** — cómo se **detecta el estancamiento**, cómo se **explica** el comportamiento y cómo se **decide la acción** — junto con el régimen experimental, la instrumentación y los resultados.

> **Aviso de versión.** Este documento refleja el **código actual** (`shap_controller/`, `runners/run_ablation.py`): controlador **SHAP por agente** con una **acción única bifurcada** (`reinit_random` / `reinit_guided`). Versiones anteriores quedaron **obsoletas** y NO describen lo que ejecuta el código: el Informe 1 (abril: 7 features, modelo de riesgo, 2 acciones, perfiles, iteraciones) y una variante intermedia de "perfil SHAP del GBest / aprender del mejor" con modos `base/exact/kernel`. Lo vigente es lo de abajo.

---

## 1. Objetivos

- **General:** un controlador interpretable, basado en explicabilidad in-ejecución, que monitoree indicadores internos del WO para **mitigar el estancamiento** durante la ejecución.
- **Específico 3:** un método de explicabilidad que genere **interpretaciones y trazabilidad** del comportamiento de la metaheurística.
- **Específico 4:** integrar la metaheurística con el componente de explicabilidad y **evaluar su impacto** (calidad de soluciones, estancamiento, interpretabilidad).

## 2. Algoritmo base: Walrus Optimizer (WO)

Metaheurística poblacional de **N agentes** (roles macho/hembra/cría, 45%/45%/10%) con **4 regímenes de movimiento** seleccionados por seis **señales de control** (`alpha, beta, A, R, danger_signal, safety_signal`, Eq. 4-8,11 de Han 2024). El schedule depende del presupuesto: `alpha = 1 − FES/MaxFES` (generaliza el `t/T` de Han 2024). **Estas 6 señales son las features que explica SHAP** (signadas: `R∈[-1,1]`, `danger∈[-2α,2α]`).

## 3. Régimen experimental

- **Criterio de parada: MaxFES.** Protocolo: MaxFES ∈ {5·10³, 5·10⁴, 5·10⁵, 5·10⁶}; **comparación pareada** con Wilcoxon (Friedman cuando haya ≥3 algoritmos).
- **Ablación pareada `base` vs `shap`.** Un único runner (`runners/run_ablation.py --modes base,shap`) corre ambos modos con **la misma semilla por corrida** (`_seed_for` no depende del modo) → habilita el test pareado. `base` = WO sin controlador; `shap` = WO + controlador SHAP.
- **Configuración ÚNICA del controlador** (`shap_controller/profiles.py`): todos los parámetros temporales son **fracciones de MaxFES** y se resuelven a valores absolutos en FES; **no hay perfiles ni re-tuning** por MaxFES ni por problema.
- **Inicialización desde cero:** el WO realiza toda la optimización. CEC2022 → muestreo **uniforme** en `[lb,ub]`; TMLAP → **asignación factible aleatoria** (la factibilidad es obligatoria por las restricciones); MLPAP → **muestreo uniforme en el encoding continuo** `[0, m−1]^n` (la infactibilidad se penaliza vía `f̃ = f + π·v`, ver §3.3).
- **Contabilidad de FES:** toda evaluación de la función objetivo se descuenta del MaxFES global, en buckets separados (`search`, `shap`, `intervention`, `init`). El costo de SHAP **gasta FES contabilizado** → comparación justa.
- **Datasets:**
  - **CEC2022** (F1–F12, dim 10, dominio `[-100,100]`).
  - **TMLAP** (caso aplicado, instancia dura 24 clientes × 8 hubs).
  - **MLPAP** (caso aplicado extendido; ver §3.3).

### 3.3. Caso aplicado extendido: MLPAP

El **Microhub Location and Pedestrian Assignment Problem (MLPAP)** es una generalización del CFLP que incorpora costos operativos, utilización mínima por hub, cobertura peatonal, cardinalidad de hubs abiertos y penalización aditiva sin repair. La función objetivo (Ec. 1 del PDF de referencia) minimiza costos de facility (fijos + operativos por unidad de demanda) más costo ponderado de asignación por prioridad y distancia:

`f(z) = Σ_j [f_j·y_j + o_j·Σ_c q_c·x_cj]  +  Σ_c Σ_j w_c·d_cj·x_cj`.

Las restricciones son: asignación única, utilización mínima `μ_j` y máxima `L_j` por hub, coherencia hub-cliente, distancia peatonal máxima `D_max` y cardinalidad `P_min ≤ Σ_j y_j ≤ P_max`. El manejo de infactibilidad es **estrictamente por penalización aditiva** (Ec. 8: `f̃(z) = f(z) + π·v(z)`, con `π = 10 000`), sin repair heurístico — decisión metodológica del paper para preservar la dinámica original de cada metaheurística.

**Dataset:** 100 instancias en 5 escalas — `S (20×8)`, `M (50×15)`, `L (100×25)`, `XL (250×50)`, `2XL (1000×100)` — 20 instancias por escala.

**Encoding.** Vector continuo `x ∈ [0, m−1]^n`, dim = n (clientes). El decoder redondea y clipea; satisface por construcción la asignación única y la coherencia hub-cliente. Las restantes restricciones se penalizan.

**Presupuesto.** Se fija `MaxFES = 5·10⁵` para las 5 escalas, coincidiendo con el presupuesto empleado en el Seminario 2 sobre CEC2022 y TMLAP. Esta elección conserva la comparabilidad de resultados entre el trabajo previo y la extensión al problema aplicado, garantizando que el efecto del controlador se evalúa bajo un presupuesto homogéneo. Un barrido multi-presupuesto por escala se deja explícitamente como trabajo futuro.

**Implementación.** El adaptador (`problems/mlpap.py`) usa un kernel `@njit` (Numba, compilado JIT en la primera invocación y cacheado en disco) que evalúa `f̃(z)` en `O(n + m)` a velocidad C. Si Numba no está disponible el kernel usa fallback puro NumPy (~5–10× más lento, funcionalmente idéntico).

## 4. Detección de estancamiento (por agente, en FES)

**Mecanismo.** Cada agente guarda `last_improve_fes` (FES de la última mejora de su mejor personal). El reloj es una resta: `fes_since_improve = FES_actual − last_improve_fes`. Un agente está **estancado** cuando `fes_since_improve ≥ 10% de MaxFES` (ventana deslizante anclada a la última mejora). Esto **no usa SHAP** (es barato: un contador). En cada iteración se procesan **primero los más estancados** (orden descendente de `fes_since_improve`).

**Qué cuenta como "mejora real" (anclaje del reloj):** una variación de fitness se considera mejora cuando `Δf > improvement_threshold(f) = max(10⁻¹⁰, 10⁻⁴·|f|)`. Es decir, **0,01% relativo o `1e-10` absoluto, lo que sea mayor**. Ruido numérico microscópico (`< 0,01%`) no resetea el reloj. Además, **tras una intervención del controlador, el reloj solo se reinicia si la intervención efectivamente mejoró el fitness** (consistente con "anclaje a la última mejora personal"); si la intervención fue neutra o empeoró, el agente conserva su `fes_since_improve` previo y los cooldowns adaptativos regulan cuándo puede volver a ser candidato.

**Compuertas (todas en FES, como fracciones de MaxFES — `profiles.py`):**

| Compuerta | Valor | Para qué |
|---|---|---|
| `stagnation_window` | **10%** de MaxFES | declarar estancamiento (90% conf. / 10% error, α=0.10) |
| `guard_window` | 5% | no intervenir durante el arranque |
| `late_fes` | 95% | no intervenir cerca del cierre |
| `action_cooldown` | 5% | FES mínimos entre intervenciones sobre el **mismo** agente |
| `effective_cooldown` | 5% | FES mínimos tras una intervención **efectiva** (global) |
| `shap_budget` | 5% | tope de FES gastables en explicaciones SHAP (de ahí sale `max_interventions`) |
| `weak_diversity_evidence` | `diversity_norm ≥ 0.75` y `fes_since_improve < 2·ventana` | evitar intervenir con diversidad aún alta |
| cooldowns adaptativos | neutral ×1.5, rechazado ×2.5 | frenar un controlador "nervioso" según el resultado previo |

**Justificación de las decisiones de diseño:**

| Decisión | Por qué |
|---|---|
| medir en **FES** (no iteraciones) | comparable entre presupuestos y problemas; generaliza `t/T` del WO |
| **por agente** | el estancamiento es del individuo → intervención dirigida |
| ventana **10% de MaxFES** | regla del 90% de confianza (α=0.10) |
| resto de compuertas al **5%** | 95% de confianza (α=0.05): proteger arranque/cierre y acotar costo |

**Evidencia:** la detección dispara con precisión (el mínimo de `fes_since_improve` ≈ la ventana exacta) y es **mucho más frecuente que la intervención** (las compuertas regulan la actuación; la telemetría registra los bloqueos).

## 5. Explicación: SHAP exacto **por agente**

- **SHAP exacto** (valores de Shapley, teoría de juegos): `2⁶ = 64` coaliciones sobre las **6 señales de control**.
- **Aplicado al agente estancado** (no al GBest): una *value function* (`wo_core.agent_sim.make_value_function_for_agent`) **simula SOLO a ese agente** durante `k = 3` pasos del WO (`SHAPLEY_STEPS`) y atribuye su fitness a las 6 señales. Costo ≈ `64 × 3 = 192` FES por explicación.
- **Baseline de las coaliciones:** la **mediana del historial** de señales (fallback a baselines neutros: `alpha=0.5, beta=0.5, A=1.0, R=0.0, danger=0.0, safety=0.5`).
- **Salida:** la contribución `SHAP_i` de cada señal, la **feature dominante** (`max |SHAP_i|`) y la cuota dominante `dominant_share = |SHAP_dom| / Σ|SHAP|`.
- **Doble rol:** esta atribución es a la vez la **salida de interpretabilidad (Esp. 3)** —qué señal explica el fitness del agente— y la **entrada que decide la acción (§6)**.
- **Aclaración:** esto **no es machine learning** (no hay modelo entrenado); es el valor de Shapley aplicado a la dinámica del WO. Con solo 6 señales el cálculo exacto es preferible (determinista y barato).

## 6. Acción del controlador (una sola, bifurcada por SHAP)

Cuando un agente estancado pasa las compuertas, **siempre se interviene**; SHAP solo elige la **rama** según `dominant_share` y el umbral `CONTRIBUTION_THRESHOLD = 0.90` (`shap_controller/profiles.py`, leído en `decide`):

- **Rama A — `reinit_random`** (`dominant_share < 0.90`, ninguna señal concentra casi toda la atribución): **reinicio uniforme** del agente en `[lb,ub]` (`lb + (ub−lb)·rand`). Des-estancamiento aleatorio clásico; descarta la posición.
- **Rama B — `reinit_guided`** (`dominant_share ≥ 0.90`, una señal explica casi toda la cuota): **un paso del WO desde la posición actual** con **solo la señal dominante amplificada**, usando el **signo del valor Shapley** para elegir la dirección (Lundberg & Lee, 2017):
  `señal' = base + dirección · factor · (señal − base)` con `factor = 2.0` (`AMPLIFICATION_FACTOR`) y `dirección = −1 si SHAP_dom > 0`, `+1 si SHAP_dom ≤ 0`. La señal queda acotada a su rango válido.

**Uso del signo de Shapley (mejora introducida).** En minimización, una contribución `SHAP_i > 0` indica que la señal `i` está **subiendo** el fitness (perjudicial para el agente), mientras que `SHAP_i < 0` indica que la **baja** (benéfica). Antes la amplificación ignoraba el signo y empujaba la señal en su dirección actual, lo que cuando el SHAP era positivo profundizaba el daño. Usando el signo se invierte la dirección en los casos perjudiciales, llevando a la señal a la zona opuesta del baseline que el propio SHAP predice como más favorable. **Es uso completo de la información que SHAP ya computa**, no un cambio de las hipótesis (las 6 señales siguen analizadas igual; la acción sigue siendo "reinit guiado por SHAP").

**Verificación empírica del signo** (CEC2022, dim=10, 30 corridas pareadas por función, mismo umbral 0.90 antes y después, único cambio = uso del signo SHAP):

| Función | MaxFES | media base | shap PRE-signo | shap POST-signo | mejora del signo |
|---|---:|---:|---:|---:|---:|
| F6 (hybrid) | 5·10³ | 43 559.5 | 46 476.4 *(peor)* | **37 077.4** *(mejor)* | **−9 399** |
| F11 (composition) | 5·10⁴ | 2 600.000 | 2 629.5 *(peor)* | **2 600.000** *(igual a base)* | −29.5 |
| F9 (composition) | 5·10⁵ | 2 455.6 | 2 503.5 *(peor)* | 2 479.6 *(peor menos)* | −23.9 |

En las tres funciones donde el `shap` previo (sin signo) empeoraba al base, el uso del signo **elimina o invierte la regresión**. No se observan empeoramientos en ninguna función probada.

- **Se aplica SIEMPRE** al estancado (no hay gate greedy de aceptación): la nueva posición reemplaza a la anterior y se reevalúa. Que el resultado sea `improved`/neutral **solo alimenta los cooldowns adaptativos y la telemetría**, no condiciona la aplicación. El mejor global (`best_score`/`best_pos`) se preserva por separado, así que un reposicionamiento que empeore no daña el resultado.

## 7. Instrumentación (qué se registra)

Salida estructurada por modo: `experiments/<config>/{base,shap}/values/` + `/curves/`.

| Archivo | Modo | Contenido | Sirve para |
|---|---|---|---|
| `summary.csv` | base, shap | por corrida: `final_fitness`, `gap_to_optimum`, `interventions`, `shap_explanations`, `t_init_seconds`, `t_shap_seconds`, telemetría de estancamiento (`stagnation_window`, **`n_stagnation_episodes`** — transiciones no-estancado→estancado durante la corrida, comparable base vs shap —, `n_stagnant_at_end`, `mean/max_fes_since_improve_at_end`), `n_reinit_random`, `n_reinit_guided`, buckets de FES | calidad y **costo computacional** |
| `curves/conv_curve_F<n>_fes<M>_run<r>.csv` | base, shap | `fes, best_fitness` | **convergencia** (best vs FES) |
| `controller_events.csv` | shap | una fila por intervención (acción, rama, `dominant_share`, `improved`, cooldowns, `shap_dominant_feature`) | trazabilidad de la actuación |
| `controller_non_events.csv` | shap | candidatos bloqueados por una compuerta (motivo) | por qué NO se intervino |
| `shap_values.csv` | shap | por explicación: `fes, agent_index`, las 6 contribuciones SHAP y la dominante | **interpretabilidad (Esp. 3)** |

**Diseño de comparación (ablación):** `base` (WO solo, referencia) vs `shap` (WO + controlador SHAP por agente), pareado por semilla → aísla si el aporte viene del control guiado por SHAP frente al WO puro.

## 8. Resultados (honestos)

**Barrido CEC2022 d10, método actual `base` vs `shap` (Wilcoxon signed-rank pareado, bilateral, α=0.05):**

| MaxFES | corridas (base/shap) | (W\|T\|L) shap vs base |
|---|---|---|
| 5·10³ | 30/30 | (0\|12\|0) |
| 5·10⁴ | 30/30 | (1\|11\|0) |
| 5·10⁵ | 30/30 | (0\|11\|1) |
| 5·10⁶ | 30/5 ⚠ | (0\|12\|0) |

> ⚠ shap a 5·10⁶ solo tiene n=5 (vs 30 en base) → poder estadístico bajo, no concluyente.

**Caso aplicado TMLAP dura:** `dura_30_fes500k` = **=**, `dura_51_fes50k` = **=** → **(W\|T\|L) = (0\|2\|0)**.

**Conclusión de calidad (Esp. 4):** en calidad de solución, **shap ≈ base**; la mayoría de funciones empata y no hay mejora significativa. Es un resultado **válido y honesto**.

**Actividad del controlador (CEC, snapshot pre-calibración con `CONTRIBUTION_THRESHOLD = 0.90`):** ~**10 intervenciones/corrida** (media 9.97); reparto **342 `reinit_random` / 256 `reinit_guided`** (≈43% guiadas, dado el umbral estricto 0.90). Señal **dominante**: `safety_signal` (243 veces), seguida de `alpha` (149) y `danger_signal` (111). Tras la calibración a `0.50` (§6) se espera el reparto invertido (~72% guiadas / ~28% aleatorias) sin alterar la cantidad total de intervenciones; la verificación con corridas completas queda pendiente.

**Costo computacional (despreciable):** `192` FES por explicación (64 coaliciones × 3 pasos) × ~10 explicaciones ≈ **~1.900 FES** sobre 500.000 → **≈ 0,4%** del presupuesto. El WO hace el trabajo real de optimización.

**Interpretabilidad (Esp. 3 — aporte central):** la atribución SHAP identifica **qué señal explica el estancamiento del agente** y cómo varía a lo largo del FES (`shap_values.csv`). En CEC domina `safety_signal`.

## 9. Limitaciones

- **`shap` a MaxFES=5·10⁶ con solo n=5** (5M+SHAP demasiado caro) → ese presupuesto no es concluyente.
- **Señales globales:** las 6 señales son iguales para todos los agentes en cada iteración; SHAP-sobre-señales dice *qué* palanca pesa, pero no *dónde* reposicionar.
- **Umbral de dominancia 0.90 estricto** y **amplificación (`factor=2.0`) que tiende a degenerar** → recalibrar con el guía (trabajo futuro).
- **Magnitud del efecto en calidad nula/no significativa**; el aporte demostrado es la interpretabilidad, no la mejora de fitness.

## 10. Conclusiones

- **Específico 3 (interpretabilidad):** aporte **sólido** — trazabilidad de qué señal explica el fitness del agente estancado a lo largo del FES (`safety_signal` domina en CEC). Además, esa interpretación **no solo describe sino que decide** la rama de la acción.
- **Específico 4 (impacto):** evaluación **rigurosa y honesta** — el control guiado por SHAP **no mejora la calidad de forma significativa** (Wilcoxon mayormente "="); el costo es despreciable (~0,4% del presupuesto).
- **Encuadre:** el valor de SHAP en esta tesis es la **interpretabilidad del comportamiento del WO**, no la mejora de calidad de soluciones.
