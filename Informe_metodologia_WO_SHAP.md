# Informe de metodología — Detección de estancamiento y acción del controlador

**Proyecto:** Walrus Optimizer (WO) con controlador interpretable basado en SHAP
**Fecha:** 2026-05-25
**Propósito:** Documento de trabajo para discutir con el profesor guía las **decisiones metodológicas** (cómo se detecta el estancamiento y cómo se determina la acción), con los **resultados experimentales** que las respaldan o las cuestionan. **No es el informe final de tesis.**

---

## 1. Marco

**Objetivo general:** desarrollar un controlador interpretable, basado en explicabilidad in-ejecución, que monitoree indicadores de desempeño en un algoritmo bio-inspirado (WO) para mitigar el estancamiento.

**Decisiones ya fijadas** (con la guía):
- Las *features* de SHAP son las **6 señales de control del WO** (`alpha, beta, A, R, danger_signal, safety_signal`).
- Enfoque **reactivo**: se actúa una vez detectado el estancamiento (no anticipación).
- Criterio de parada y comparabilidad: **MaxFES** (protocolo del curso: 51 corridas, MaxFES ∈ {5·10³, 5·10⁴, 5·10⁵, 5·10⁶}).

---

## 2. Cómo se DETECTA el estancamiento

### 2.1 Mecanismo
La detección es **por agente** y se mide en **FES** (no en iteraciones). Cada agente lleva una marca `last_improve_fes[i]` con el FES de su última mejora de su mejor personal. El "reloj" de estancamiento es una resta:

```
fes_since_improve[i] = FES_global_actual − last_improve_fes[i]
```

Un agente está **estancado** si `fes_since_improve[i] ≥ 10% de MaxFES`. Es, en esencia, una **ventana deslizante anclada a la última mejora**: se pregunta "¿este agente mejoró en el último 10% del presupuesto?".

### 2.2 Argumentos de las decisiones

| Decisión | Justificación |
|---|---|
| **Medir en FES, no en iteraciones** | El protocolo compara bajo MaxFES; medir en FES hace la detección **comparable** entre presupuestos y problemas. Generaliza el `t/T` del WO original (Han 2024) a `FES/MaxFES`. |
| **Por agente, no global** | El estancamiento es una propiedad del **individuo**; detectarlo por agente permite una intervención **dirigida** al que dejó de progresar, sin penalizar a los que avanzan. |
| **Ventana = 10% de MaxFES** | Regla estadística del **90% de confianza** (α = 0.10): se considera estancado tras no mejorar en una fracción significativa del presupuesto. |
| **Compuertas** (guard 5% inicial, late 95% final, cooldown 5% por agente, presupuesto SHAP 5%) | Proteger el arranque y el cierre, **evitar un controlador "nervioso"** (que intervenga sin parar) y **acotar el costo** de la explicación. |

### 2.3 Evidencia de que funciona como se diseñó
- **Dispara con precisión:** el mínimo de `fes_since_improve` observado ≈ la ventana exacta (p. ej. 5.010 con MaxFES = 50.000), es decir, en cuanto un agente cruza el 10%.
- **La detección es mucho más frecuente que la intervención:** son las **compuertas** las que regulan la frecuencia real. Ejemplo (CEC2022, 5 corridas): **52.646** detecciones quedaron bloqueadas por las compuertas frente a **538** intervenciones efectivas. → El detector "ve" el estancamiento constantemente en la fase de explotación; las compuertas evitan sobre-actuar.

**Conclusión sobre la detección:** el mecanismo es simple, comparable (FES), por agente, y está validado empíricamente. **No depende de SHAP** — el estancamiento se detecta con el fitness y el reloj de FES.

---

## 3. Cómo se DETERMINA la acción

### 3.1 Lo acordado con la guía (reunión 2026-05-19)
- **Una sola acción** ("si tienes más de una, no sabes cuál afectó").
- La acción es **reinicializar la posición del agente estancado**, sin anclar al GBest ("no tiene que explorar sesgado").
- Esa acción **se bifurca según la explicabilidad** (SHAP): con/sin contribución dominante.

### 3.2 Cómo entra SHAP (y su límite conceptual)
SHAP atribuye el fitness del agente estancado a las **6 señales de control**. Pero esas señales son **globales** (iguales para todos los agentes en una iteración): lo que difiere entre agentes es su **posición**, no las señales. Por eso **SHAP no puede indicar *dónde* reiniciar** (en qué variable del espacio), pero sí **cuánto/qué tan fuerte** reiniciar, modulando un parámetro por agente.

La acción implementada (mutación modulada):
```
x_nuevo = (1 − w) · x_actual  +  w · [ lb + (ub − lb) · rand ]
```
donde `w = cuota de contribución de la señal dominante` (`|SHAP_dominante| / Σ|SHAP|`). Si una señal domina con claridad, `w → 1` (reinicio casi total); si la contribución está repartida, `w < 1` (mutación parcial que conserva parte de la solución).

### 3.3 Evaluación del impacto (estudio de ablación)
Para saber si SHAP **realmente aporta** a la acción, se compararon 4 variantes con las mismas semillas (comparación pareada):

| Brazo | Acción | ¿Usa SHAP? |
|---|---|---|
| **base** | ninguna (WO solo) | — |
| **blind** | reinit total (`w = 1`) | no |
| **wfix** | reinit parcial fijo (`w = 0.5`) | no |
| **shap** | reinit modulado (`w = cuota dominante`) | sí |

**Resultados (Wilcoxon/Friedman pareados):**

| Conjunto | Resultado |
|---|---|
| **CEC2022 (F1–F12, 30 runs)** | SHAP no supera a blind en **ninguna** función; Friedman global **p = 0.075** (no significativo). |
| **TMLAP dura, 30 runs** | shap parecía mejor (shap vs wfix **p = 0.034**; Friedman p = 0.005). |
| **TMLAP dura, 51 runs** *(muestra del protocolo)* | shap vs wfix **p = 0.778**; shap vs base **p = 0.228**; Friedman **p = 0.489** → **NO significativo**. |

**Lección metodológica:** el resultado "significativo" con 30 corridas era un **falso positivo**; al usar las **51 corridas del protocolo**, la diferencia se desvanece. Esto **valida por qué el protocolo exige 51 corridas** (evitar concluir de más con muestras chicas).

**Mecanismo que lo explica:** la cuota dominante `w` **degenera a ~1** la mayor parte del tiempo (mediana de `w` = 1.0; ~74% de los casos `w ≥ 0.95`), porque SHAP casi siempre encuentra una señal dominante. Con `w ≈ 1`, el reinicio modulado **equivale al reinicio ciego**, así que no hay diferencia que medir.

### 3.4 Conclusión honesta sobre la acción
Con la muestra correcta, **el control guiado por SHAP no mejora la calidad de las soluciones de forma estadísticamente significativa** (ni en CEC ni en TMLAP). El **valor demostrado de SHAP es la interpretabilidad/trazabilidad** del comportamiento del algoritmo (Objetivo específico 3), no la mejora de calidad. Reportar este resultado **es ciencia válida** y cumple el Objetivo específico 4 ("evaluar el impacto").

---

## 4. Hallazgo sobre el benchmark TMLAP (relevante para la discusión)
El inicializador de TMLAP usa una **búsqueda local greedy** que, por sí sola, **resuelve las instancias pequeñas**:

| Instancia | Clientes × Hubs | ¿El init la resuelve? (init == final) |
|---|---|---|
| simple | 6 × 3 | **sí, 5/5 corridas** (el algoritmo no aporta) |
| mediana | 12 × 5 | casi (3/5; mejora ≤ 2) |
| **dura** | 24 × 8 | **no (0/5)** → la única que ejercita el WO |
| grande | 1000 × 500 | computacionalmente **inviable** (~192 ms/evaluación → ~25 días para 51 corridas) |

→ El benchmark TMLAP, con este inicializador, **solo deja una instancia (dura) que realmente pone a prueba el algoritmo**. Es una limitación a discutir.

---

## 5. Puntos para discutir con el profesor

1. **Reposicionar SHAP como herramienta de interpretabilidad** (Objetivo 3) y reportar el control como **evaluación honesta de impacto** (Objetivo 4, con resultado negativo robusto). ¿Está de acuerdo con este encuadre?
2. **Benchmark TMLAP:** ¿conviene usar inicialización aleatoria (que el WO trabaje desde cero) y/o conseguir instancias de tamaño medio **viables** que ejerciten el algoritmo, ya que la chica la resuelve el init y la grande es inviable?
3. **La acción del controlador:** dado que la modulación por SHAP degenera a reinicio ciego, ¿simplificar la acción, o explorar una modulación donde `w` varíe más (mayor rango de respuesta)?
4. **El detector de estancamiento** parece sólido y defendible tal cual. ¿Confirma que la metodología de detección (FES, por agente, 10%) queda cerrada?

---

## 6. Resumen de decisiones metodológicas

| Componente | Decisión | Estado |
|---|---|---|
| Detección | por agente, en FES, ventana 10% de MaxFES, con compuertas | **validada** (dispara con precisión; compuertas regulan) |
| Explicación (SHAP) | exacta sobre las 6 señales de control, por agente | **funciona** como diagnóstico interpretable |
| Acción | reinicio modulado por la cuota dominante (`w`) | **sin efecto significativo** en calidad (51 runs) → valor en interpretabilidad |
| Benchmark | CEC2022 + TMLAP | TMLAP **limitado** (init resuelve las chicas; grande inviable) |
