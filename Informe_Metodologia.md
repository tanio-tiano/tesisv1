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

## 5. Explicación: SHAP por agente

- **SHAP exacto** (valores de Shapley, teoría de juegos): `2⁶ = 64` coaliciones sobre las **6 señales de control**.
- **Por agente:** una *value function* simula al agente estancado durante `k` pasos del WO y atribuye su fitness a las 6 señales. Cada explicación corresponde a **un agente** concreto.
- **Trazabilidad:** se registra `shap_trace.csv` con `fes, agent_index, dominant_feature` y las 6 contribuciones → permite ver **qué señal explica el estancamiento, por agente y a lo largo del FES**.
- **Aclaración:** esto **no es machine learning** (no hay modelo entrenado); es el valor de Shapley aplicado a la dinámica del WO.

## 6. Acción del controlador

- **Una sola acción** (decisión de la guía): **reinicializar** la posición del agente estancado, sin anclar al GBest.
- **Modulada por SHAP** — mutación:
  `x_nuevo = (1 − w) · x_actual + w · [ lb + (ub − lb) · rand ]`, con `w = cuota de la señal dominante` (`|SHAP_dom| / Σ|SHAP|`).
  - señal dominante clara → `w → 1` (reinicio fuerte);
  - contribución repartida → `w < 1` (mutación parcial que conserva parte de la solución).
- El reinicio se aplica **siempre**; el GBest se preserva (un reinicio que empeore no daña el resultado).

## 7. Instrumentación (qué se registra, para cualquier instancia)

| Archivo | Contenido | Sirve para |
|---|---|---|
| `ablation_summary.csv` | por corrida: `final_fitness`, `gap`, `n_interventions`, **`t_init/t_shap` + `ram_peak_mb`**, buckets de FES | calidad y **costo computacional** |
| `curves.csv` | `run_id, mode, fes, best_fitness` | **convergencia** (cómo se actualiza el best vs FES) |
| `shap_trace.csv` | `fes, agent_index` + 6 contribuciones | **interpretabilidad por agente** (Esp. 3) |
| `ablation_by_function.csv` | medias por modo + p-valores | comparación estadística |

**Diseño de comparación (ablación de 4 brazos):** `base` (WO solo) · `blind` (reinit ciego) · `wfix` (reinit parcial fijo) · `shap` (reinit modulado) → aísla si el aporte es de SHAP o de reiniciar.

## 8. Resultados (honestos)

- **Calidad:** con la muestra correcta (**51 corridas**), SHAP **no mejora la calidad de forma significativa** (TMLAP dura: Friedman p≈0.49; CEC2022: p≈0.075). *El resultado "significativo" con 30 corridas era un **falso positivo** — esto valida por qué el protocolo exige 51 corridas.*
- **Interpretabilidad:** la traza muestra **qué señal explica el estancamiento y que difiere por problema**: `danger_signal`/`alpha` dominan en TMLAP, `safety_signal` en CEC; la señal `A` no contribuye.
- **WO desde 0 (en consolidación):** el WO realiza toda la optimización (mejora ≈30 en la instancia dura); la corrida de confirmación de 51 corridas está finalizando.

## 9. Limitaciones

- **Benchmark TMLAP:** las instancias pequeñas son triviales (su óptimo se alcanza sin esfuerzo de búsqueda); la grande (1000×500) es **computacionalmente inviable** (~25 días para 51 corridas con MaxFES=50.000); solo la **dura** ejercita realmente al WO.
- **`w` degenera a ~1** la mayor parte del tiempo → el reinicio modulado ≈ reinicio ciego (razón mecánica de la ausencia de efecto).
- Resultados con MaxFES=50.000 (1 de los 4 del protocolo) y dim 10 en CEC.

## 10. Conclusiones

- **Específico 3 (interpretabilidad):** aporte **sólido** — trazabilidad por agente de las palancas que explican el estancamiento, distintas según el problema.
- **Específico 4 (impacto):** evaluación **rigurosa** — el control guiado por SHAP no mejora la calidad de forma significativa; es un resultado **válido y honesto**.
- **Encuadre:** el valor de SHAP en esta tesis es la **interpretabilidad del comportamiento del WO**, no la mejora de calidad de soluciones.
