# Informe de Metodología — Walrus Optimizer con controlador interpretable (SHAP)

**Proyecto:** mejora del Walrus Optimizer (WO) mediante un controlador interpretable basado en explicabilidad in-ejecución (SHAP).
**Fecha:** 2026-05-25
**Propósito:** documentar la metodología — cómo se **detecta el estancamiento**, cómo se **explica** el comportamiento y cómo se **determina la acción** — junto con el régimen experimental, la instrumentación y los resultados.

---

## 1. Objetivos

- **General:** un controlador interpretable, basado en explicabilidad in-ejecución, que monitoree indicadores de desempeño del WO para **mitigar el estancamiento**.
- **Específico 3:** un método de explicabilidad que genere **interpretaciones y trazabilidad** del comportamiento de la metaheurística.
- **Específico 4:** integrar la metaheurística con el componente de explicabilidad y **evaluar su impacto** (calidad de soluciones, anticipación, interpretabilidad).

## 2. Algoritmo base: Walrus Optimizer (WO)

Metaheurística poblacional de **N agentes** (roles macho/hembra/cría) con **4 regímenes de movimiento** seleccionados por seis **señales de control**: `alpha, beta, A, R, danger_signal, safety_signal`. El schedule depende del presupuesto: `alpha = 1 − FES/MaxFES` (generaliza el `t/T` de Han 2024). Estas 6 señales son las **features** que explica SHAP.

## 3. Régimen experimental

- **Criterio de parada: MaxFES** (protocolo: 51 corridas; MaxFES ∈ {5·10³, 5·10⁴, 5·10⁵, 5·10⁶}; comparación con Friedman/Wilcoxon).
- **Inicialización:** la población se inicializa **desde cero** y es el WO quien realiza toda la optimización. En CEC2022, muestreo **uniforme** aleatorio en [lb,ub]; en TMLAP, una **asignación factible aleatoria** (la factibilidad es obligatoria por las restricciones del problema).
- **Contabilidad de FES:** toda evaluación de la función objetivo se descuenta del MaxFES global, separada en buckets: `search`, `shap`, `intervention`, `init`.

## 4. Detección de estancamiento

**Mecanismo (por agente, en FES):** cada agente guarda `last_improve_fes` (FES de su última mejora de su mejor personal). El reloj es una resta:
`fes_since_improve = FES_actual − last_improve_fes`. Un agente está **estancado** si `fes_since_improve ≥ 10% de MaxFES`. Es una **ventana deslizante anclada a la última mejora**.

**Justificación de cada decisión:**
| Decisión | Por qué |
|---|---|
| medir en **FES** (no iteraciones) | comparable entre presupuestos y problemas; generaliza `t/T` del WO |
| **por agente** | el estancamiento es del individuo → intervención dirigida |
| ventana **10% de MaxFES** | regla del 90% de confianza (α=0.10) |
| **compuertas** (guard 5%, late 95%, cooldown 5% por agente, presupuesto SHAP 5%) | proteger arranque/cierre, evitar un controlador "nervioso" y acotar el costo |

**Evidencia:** dispara con precisión (el mínimo de `fes_since_improve` ≈ la ventana exacta) y la **detección es mucho más frecuente que la intervención** (las compuertas regulan la actuación).

## 5. Explicación: perfil SHAP del mejor agente (GBest)

- **SHAP exacto** (valores de Shapley, teoría de juegos): `2⁶ = 64` coaliciones sobre las **6 señales de control**.
- **Aplicado al GBest** (giro metodológico de la guía — *"aprender del mejor"*): una *value function* simula al **mejor agente** durante `k` pasos del WO y atribuye su fitness a las 6 señales. De ahí se obtiene un **perfil de contribuciones** `w_i = |SHAP_i| / Σ|SHAP|` (las 6 cuotas suman 1). El perfil se calcula **una sola vez por iteración** y se comparte para todos los estancados de esa iteración (1 explicación SHAP, costo acotado).
- **Trazabilidad:** se registra `shap_trace.csv` con `fes, agent_index, dominant_feature` y las 6 contribuciones. El perfil del GBest se marca con `agent_index = −1` → permite ver **qué señal explica el desempeño del mejor agente, y cómo varía a lo largo del FES**.
- **Aclaración:** esto **no es machine learning** (no hay modelo entrenado); es el valor de Shapley aplicado a la dinámica del WO.
- **Doble rol del perfil:** este perfil de contribuciones es a la vez **la salida de interpretabilidad (Esp. 3)** —explica qué señal pesa en el mejor agente— y **la entrada que guía la acción (§6)**. Es decir, la interpretación no es solo un artefacto de reporte: es lo que decide *cómo* se reposiciona al agente estancado.
- **Variante de contraste (local):** el mismo perfil puede estimarse con **KernelSHAP** (librería `shap`, modo `kernel`) en vez del cálculo exacto; con solo 6 señales el exacto es preferible (determinista y barato), por lo que `kernel` se usa únicamente para contrastar exacto-vs-aproximación.

## 6. Acción del controlador

- **Una sola acción** (decisión de la guía) — *"aprender del mejor, reposicionar al peor"*: se toman los **agentes peor evaluados / estancados** (los detectados por FES en §4) y se los **reposiciona** con un paso del WO, **replicándoles el perfil de contribuciones SHAP del mejor agente (GBest)**. La prioridad la tienen los **más estancados** (mayor `fes_since_improve`).
- **La interpretación SHAP guía la acción** (cierra el lazo Esp. 3 → control): el *cómo* reposicionar no es arbitrario, lo dicta el perfil de contribuciones del GBest calculado en §5.
- **Modulada por SHAP** — el paso del WO se ejecuta con las 6 señales **amplificadas según el perfil del GBest**:
  `señal' = base + (1 + amp · w_señal) · (señal − base)`, donde `w_señal` es la cuota SHAP de esa señal en el GBest y `base` su valor neutro.
  - señal con alta contribución en el GBest → mayor amplificación (esa palanca se acentúa al reposicionar);
  - señal irrelevante (`w ≈ 0`) → el paso WO queda casi sin cambios en esa dirección.
- El reposicionamiento se aplica **siempre** a los estancados; el GBest se preserva (un reposicionamiento que empeore no daña el resultado).

## 7. Instrumentación (qué se registra, para cualquier instancia)

| Archivo | Contenido | Sirve para |
|---|---|---|
| `ablation_summary.csv` | por corrida: `final_fitness`, `gap`, `n_interventions`, **`t_init/t_shap` + `ram_peak_mb`**, buckets de FES | calidad y **costo computacional** |
| `curves.csv` | `run_id, mode, fes, best_fitness` | **convergencia** (cómo se actualiza el best vs FES) |
| `shap_trace.csv` | `fes, agent_index` (`−1` = perfil GBest) + 6 contribuciones | **interpretabilidad** (Esp. 3) |
| `ablation_by_function.csv` | medias por modo + p-valores | comparación estadística |

**Diseño de comparación (ablación):** `base` (WO solo, referencia) · `exact` (reposicionamiento por perfil SHAP exacto del GBest, el método de la tesis) · `kernel` (mismo perfil por KernelSHAP, contraste local) → aísla si el aporte viene del control guiado por SHAP frente al WO puro.

## 8. Resultados (honestos)

Experimento de referencia: **TMLAP instancia dura (8h × 24c), 51 corridas, MaxFES = 50.000**, comparación pareada `base` vs `exact` (misma semilla por corrida). Salida: `experiments/ablation_b4_dura_51/`.

**Calidad de soluciones (Esp. 4):**

| modo | media | desv. est. | mediana | min | max |
|---|---|---|---|---|---|
| `base` (WO solo) | 277.76 | 3.06 | 278 | 271 | 283 |
| `exact` (control SHAP) | **277.53** | **2.43** | 278 | 272 | 284 |

- Pareado: `exact` gana 27, empata 6, pierde 18 (de 51); diferencia media −0.24. **Wilcoxon p = 0.61 → NO significativo.**
- *El resultado "significativo" obtenido antes con 30 corridas era un **falso positivo**; esto valida por qué el protocolo exige 51 corridas.*
- Único matiz a favor: `exact` es **más robusto** (desv. est. 2.43 < 3.06) — converge a resultados más estables, aunque la media es estadísticamente equivalente.

**Costo computacional (despreciable):** `exact` realiza ~**13 reposicionamientos/corrida** y gasta solo **~627 FES** en SHAP + intervención de 50.000 (**≈ 1,3 %**); tiempo 7.55 s vs 7.52 s del `base`; RAM idéntica (~116 MB). El WO hace el trabajo real: mejora **≈ 30,5** (308,3 → 277,8) desde el init aleatorio.

**Interpretabilidad (Esp. 3 — aporte sólido):** el perfil del GBest (163 explicaciones) muestra qué señal explica el desempeño del mejor agente en TMLAP:

| señal | cuota media | veces dominante |
|---|---|---|
| `danger_signal` | **0.51** | 54 |
| `safety_signal` | 0.25 | 22 |
| `beta` | 0.13 | 10 |
| `alpha` | 0.08 | 73 |
| `R` | 0.03 | 4 |
| `A` | **0.00** | 0 |

- `danger_signal` **concentra la mitad de la atribución** (cuota 0.51); la señal **`A` no contribuye nada** (cuota 0.000) en todo el experimento.
- Matiz de trazabilidad: `alpha` es la *más frecuente* como señal dominante (73 veces) pero con cuota media baja (0.08) — gana por poco y de forma dispersa, mientras `danger_signal` cuando pesa, pesa fuerte.

## 9. Limitaciones

- **Benchmark TMLAP:** las instancias pequeñas son triviales (su óptimo se alcanza sin esfuerzo de búsqueda); la grande (1000×500) es **computacionalmente inviable** (~25 días para 51 corridas con MaxFES=50.000); solo la **dura** ejercita realmente al WO.
- **Señales globales:** las 6 señales son iguales para todos los agentes en cada iteración; SHAP-sobre-señales puede decir *qué* palanca pesa, pero no *dónde* reposicionar. El perfil del GBest acota esto a *"aprender del mejor"*, lo que introduce un **sesgo hacia la mejor solución** (limita la diversificación).
- **Magnitud del efecto pequeña** (~0,1 % en la media) y **no significativa**; el beneficio observable se limita a una **menor varianza**.
- Resultados con MaxFES=50.000 (1 de los 4 del protocolo) y una sola instancia (dura).

## 10. Conclusiones

- **Específico 3 (interpretabilidad):** aporte **sólido** — trazabilidad del perfil de palancas del mejor agente (`danger_signal` domina con cuota 0.51; `A` es irrelevante con cuota 0.00) a lo largo del FES. Además, esa interpretación **no solo describe sino que guía la acción**: el perfil del GBest es lo que decide cómo se reposiciona al agente estancado.
- **Específico 4 (impacto):** evaluación **rigurosa** — el control guiado por SHAP (reposicionamiento por perfil del GBest) no mejora la calidad de forma significativa (Wilcoxon p=0.61), aunque sí reduce la varianza; es un resultado **válido y honesto**.
- **Encuadre:** el valor de SHAP en esta tesis es la **interpretabilidad del comportamiento del WO**, no la mejora de calidad de soluciones.
