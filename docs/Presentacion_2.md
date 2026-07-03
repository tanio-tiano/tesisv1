# Presentación 2 — Investigación Aplicada

> Outline slide-por-slide listo para armar las diapositivas. Cada slide indica el
> **contenido en pantalla**, la **figura** a insertar (rutas en `experiments/presentacion/`)
> y el **énfasis** al hablar. ~16 slides para 15 min (≈1 min/slide).
> Convertir a PDF: `python -m analysis.md_to_pdf Presentacion_2.md Presentacion_2.pdf`.

---

## Slide 1 — Portada

**En pantalla:**
- **EXPLICABILIDAD EN ALGORITMOS BIO-INSPIRADOS: DISEÑO DE UN CONTROLADOR PARA MITIGAR ESTANCAMIENTO Y CONVERGENCIA PREMATURA**
- Universidad de Valparaíso — Facultad de Ingeniería — Escuela de Ingeniería Informática
- Lucciano Fabián Cáceres Costa
- Prof. Guía: Rodrigo Olivares · Prof. Correferente: Pablo Olivares
- Presentación 2 — Investigación Aplicada — 2026

---

## Slide 2 — Contexto y motivación

**En pantalla:**
- Las metaheurísticas poblacionales **se estancan**: agentes dejan de mejorar y gastan evaluaciones sin avanzar.
- El Walrus Optimizer (WO) elige su movimiento por señales temporales fijas, **sin diagnosticar** el estado del agente.
- Además son **cajas negras**: se ve el resultado, no *por qué* la búsqueda se degrada.
- **Idea:** usar explicabilidad (SHAP) no como reporte posterior, sino como **controlador en línea** que decide cómo rescatar a un agente estancado.

**Énfasis:** el problema es doble — estancamiento + falta de interpretabilidad. Mi propuesta ataca ambos a la vez.

---

## Slide 3 — Marco conceptual: Walrus Optimizer

**En pantalla:**
- Metaheurística poblacional bio-inspirada (Han et al., 2024).
- 4 regímenes de movimiento según 2 señales (danger, safety): exploración por diferencias, reproducción (Halton + Lévy), huida por un líder, agrupación por dos líderes.
- 6 señales de control: `alpha, beta, A, R, danger_signal, safety_signal`.
- Población: 45% machos / 45% hembras / 10% crías.

**Figura:** `Diagramas/Diagrama_diseno_WO_SHAP.png`
**Énfasis:** esas 6 señales globales gobiernan todo el comportamiento — serán las *features* que SHAP explica.

---

## Slide 4 — Marco conceptual: SHAP / XAI

**En pantalla:**
- Valores de Shapley (teoría de juegos): reparten el "crédito" de un resultado entre las variables.
- Aquí: **features = las 6 señales del WO**; el "modelo" = una *value function* que simula al agente y devuelve su mejor fitness.
- **SHAP exacto** (2⁶ = 64 coaliciones) — factible porque solo hay 6 features.
- Diferencia clave: uso **online y prescriptivo**, no descriptivo post-hoc.

**Énfasis:** la novedad metodológica es *cuándo* y *para qué* se usa SHAP — durante la corrida, para decidir una acción.

---

## Slide 5 — Marco conceptual: FES (presupuesto de la búsqueda)

**En pantalla:**
- **FES = *Function Evaluations*:** número de veces que se evalúa la función objetivo (fitness). Es el recurso que **realmente** consume una metaheurística.
- **¿Por qué no contar iteraciones?** Cada iteración del WO hace un número variable de evaluaciones (las 4 ramas no cuestan lo mismo) → comparar por iteraciones sería injusto. **FES es la moneda común.**
- **MaxFES** = presupuesto máximo de evaluaciones; criterio de parada estándar en benchmarking (CEC). Aquí: 5·10³ a 5·10⁶.
- **Triple rol en este trabajo:**
  1. **Parada** — la corrida termina al agotar MaxFES.
  2. **Comparación justa** — el costo de SHAP también gasta FES y se contabiliza.
  3. **Estancamiento** — un agente está estancado si pasa ≥10% de MaxFES sin mejorar su mejor personal.

**Énfasis:** FES es la unidad que vuelve honesta la comparación *y* la base de la detección de estancamiento — por eso es un concepto central, no un detalle de implementación.

---

## Slide 6 — Estado del arte

**En pantalla:**
- Metaheurísticas bio-inspiradas: eficaces en problemas NP-hard y de alta dimensión [Talbi 2009; Engelbrecht 2007; Somvanshi 2025].
- **Estancamiento (definición):** un algoritmo *deja de mejorar la mejor solución durante un período prolongado*, típicamente por quedar atrapado en un óptimo local, desperdiciando evaluaciones sin progreso; ese período puede medirse en iteraciones, tiempo o **evaluaciones de la función (FES)** [Črepinšek et al. 2025].
- Limitación transversal: operan como **caja negra**, difícil detectar/ajustar el estancamiento [Das 2022; Kalasampath 2025].
- XAI / SHAP [Lundberg & Lee 2017]: explicabilidad usada casi siempre **post-hoc**, rara vez dentro del proceso de optimización.
- Hyper-heurísticas y enfoques adaptativos [Burke 2013; Cao & Tang 2018]: mejoran resultados **sin abordar la interpretabilidad**.
- **Hueco:** XAI como mecanismo de **control en línea** — lo que propone este trabajo.

**Énfasis:** nadie usa la explicación para *intervenir* durante la búsqueda; ahí está el aporte.

---

## Slide 7 — Problema y objetivos

**En pantalla:**
- **Problema:** las metaheurísticas estocásticas sufren convergencia prematura / estancamiento y, al ser cajas negras, no permiten intervenir de forma informada.
- **Objetivo general:** desarrollar un controlador interpretable que **monitoree, analice y mejore** el comportamiento de la metaheurística durante su ejecución, integrando SHAP con el WO.
- **Objetivos específicos** *(validar con anteproyecto):*
  1. Monitorear señales internas del WO (exploración/explotación, diversidad, estancamiento).
  2. Atribuir, vía SHAP, el deterioro de la búsqueda a las variables de control.
  3. Diseñar acciones de rescate **informadas por esa atribución**.
  4. Validar en CEC 2022 y un caso aplicado (TMLAP) con análisis estadístico.

**Énfasis:** leer el objetivo general textual; remarcar que la evaluación es honesta (Obj. 4).

---

## Slide 8 — Diseño de la solución: arquitectura del controlador

**En pantalla:**
- Pipeline por iteración:
  1. **Detección de estancamiento por agente** (≥10% de MaxFES sin mejorar su mejor personal) — barato, sin SHAP.
  2. **Explicación SHAP** solo del agente estancado.
  3. **Acción única bifurcada por la explicación:**
     - **Rama A — `reinit_random`** (sin feature dominante): reinicio uniforme.
     - **Rama B — `reinit_guided`** (feature dominante ≥ umbral): paso WO con la señal dominante amplificada.
- Configuración **única** auto-escalable: todos los parámetros son fracciones de MaxFES.

**Figura:** `Diagramas/Diagrama_diseno_WO_SHAP.png`
**Énfasis:** una sola acción, bifurcada por lo que dice SHAP — simple y trazable.

---

## Slide 9 — Diseño de la solución: metodología experimental

**En pantalla:**
- Criterio de parada = **MaxFES** (4 presupuestos: 5·10³, 5·10⁴, 5·10⁵, 5·10⁶).
- **30 corridas** independientes por configuración; ablación **pareada** base vs shap (misma semilla).
- SHAP cuenta en el presupuesto (bucket aparte) → comparación justa en FES.
- Tests: **Wilcoxon** signed-rank (mejor/igual/peor, notación +/=/−); Friedman cuando haya ≥3 competidores.

**Figura:** `experiments/presentacion/efecto_maxfes.png`
**Énfasis:** las semillas pareadas permiten un test pareado honesto; SHAP no "regala" FES.

---

## Slide 10 — Experimentos: variables, datasets e implementación

**En pantalla:**
- Variables: las 6 señales (features SHAP); parámetros del controlador (ventana 10% MaxFES, cooldowns 5%, umbral de dominancia, factor de amplificación).
- Datasets:
  - **CEC 2022 (F1–F12):** básicas (F1-5), híbridas (F6-8), composición (F9-12); dim 10; dominio [-100,100].
  - **TMLAP** (caso aplicado): asignación multinivel, instancia dura (24 clientes × 8 hubs).
- Implementación: Python modular (`wo_core/`, `shap_controller/`, `problems/`, `runners/`, `analysis/`); triple clipping garantiza soluciones factibles.

**Figura:** `Diagramas/Diagrama_implementacion_WO_SHAP.png`
**Énfasis:** dos mundos — benchmark continuo (CEC) y problema combinatorio real (TMLAP).

---

## Slide 11 — Resultados  *(hilo Calidad → Interpretabilidad; se puede dividir en 3 slides)*

> **Encuadre (R0 — decilo al cerrar Metodología):** se corrió el factorial CEC2022 (12 funciones × 4 MaxFES × 30 corridas) + TMLAP, **ablación pareada base vs shap**. Notación: `+` shap mejor · `=` igual · `−` peor; `(W|T|L)` = mejor | igual | peor por función.

### ▸ Calidad (hallazgo honesto)

**R1 — Veredicto global** · *tabla Wilcoxon (shap vs base):*

  | MaxFES | (W\|T\|L) |
  |---|---|
  | 5 000 | (0\|12\|0) |
  | 50 000 | (1\|11\|0) |
  | 500 000 | (0\|11\|1) |
  | 5 000 000 | (0\|12\|0)* |

  TMLAP dura: **(0\|2\|0)**.  *(\*shap a 5M con n=5 → no concluyente.)*
  *Mensaje:* shap ≈ base; la mayoría **empata** → hipótesis de mejora **no confirmada**.

**R2 — Distribución por función** · figura: `boxplots_por_funcion.png` (5·10⁵, 30/30).
  *Mensaje:* las cajas base/shap se **solapan** → confirma el empate; matiz: shap tiende a **menor varianza** (descriptivo, no test).

### ▸ El controlador en acción (puente)

**R3 — Actividad + costo** · figura: `acciones_rescate.png`.
  *Mensaje:* interviene de forma **acotada** (~10/corrida; reparto random/guiado) y gasta **~0,4%** del presupuesto → liviano y trazable.

### ▸ Interpretabilidad (aporte)

**R4 — Qué señal explica el estancamiento** · figura: `features_shap.png` (las 6, 5·10⁵).
  *Mensaje:* domina **`safety_signal`** (magnitud y frecuencia); **`A` no contribuye** (cuota 0). SHAP "abre la caja negra".

**R5 — La lectura evoluciona con el presupuesto** · figura: `features_shap_comparativa_cec.png`.
  *Mensaje:* **`danger_signal` → `safety_signal`** al crecer el FES; lectura **consistente** en CEC y TMLAP.

### ▸ Síntesis (R6 → enlaza con Conclusiones)
**Igual peso:** (1) evaluación **rigurosa** → calidad equivalente; (2) **interpretabilidad** → qué señal gobierna el estancamiento y cómo cambia; todo a **costo despreciable**.

**Cómo dividir:** Slide A = Calidad (R1+R2) · Slide B = Controlador en acción + R4 (R3+R4) · Slide C = R5 + Síntesis (R5+R6).
**Backup/anexo:** `tabla_resumen_fes500000.png`, `efecto_maxfes.png` (avisar 5M n=5), `convergencia/convergencia_cec_panel.png` (panel 4×3 de las 12 funciones, base+shap con banda ±std) + `convergencia/convergencia_tmlap_*.png` (2 instancias dura), `features_shap_cec_d10_fes{5000,50000,500000}.png`, `boxplots_tmlap.png`, `tabla_tmlap.png`.

---

## Slide 12 — Implantación: requerimientos y ambiente

**En pantalla:**
- **Software:** Python 3.11+ · numpy, pandas, scipy, opfunu (`requirements.txt` / `requirements-server.txt`).
- **Hardware mínimo:** cualquier CPU, ~4 GB RAM (1 corrida).
- **Hardware recomendado / usado:** servidor Mac Studio (Apple Silicon, 24 núcleos, 64 GB) para el barrido completo *(confirmar specs exactas)*.
- **Preparación:** `conda` + `pip install -r requirements.txt`.

**Énfasis:** el método es liviano; el costo está en el número de corridas × MaxFES, no en requisitos de máquina.

---

## Slide 13 — Implantación: ejecución, evidencia y manual

**En pantalla:**
- **Ejecución reproducible y resiliente:** `run_remote_cec.sh` / `run_remote.sh` con `nohup` (sobreviven a la desconexión SSH); flags `--runs/--max-fes/--budgets`.
- **Evidencia:** sesión SSH ejecutando el barrido + logs en vivo (`tail -f ablation_*.log`). *(insertar screenshot)*
- **Manual de usuario:** `COMO_EJECUTAR.txt` (uso, columnas de `summary.csv`, flujo, troubleshooting).
- **Salida estructurada:** `experiments/<config>/{base,shap}/values/summary.csv` + `curves/`.

**Figura:** *(screenshot de la sesión SSH / del log — pendiente de capturar)*
**Énfasis:** reproducibilidad total: seeds pareadas, presupuesto contabilizado, un comando lo corre.

---

## Slide 14 — Conclusiones

**En pantalla:**

**Éxitos conseguidos**
- Controlador interpretable **en línea** funcionando: lazo detectar → explicar → decidir → reposicionar sobre el WO (no post-hoc).
- **Interpretabilidad lograda (Esp. 3):** SHAP identifica la señal que explica el estancamiento — domina **`safety_signal`**, **`A` no contribuye** (cuota 0) — y cómo varía con el presupuesto.
- **Explicabilidad in-ejecución viable:** SHAP exacto (64 coaliciones) a **~0,4 %** del presupuesto (~10 intervenciones/corrida).
- **Evaluación rigurosa (Esp. 4):** ablación pareada, **Shapiro–Wilk + Wilcoxon signed-rank**, CEC2022 + caso aplicado TMLAP.

**Avances en el área**
- XAI usado como controlador **prescriptivo y en línea**, no descriptivo post-hoc → cubre el hueco del estado del arte.
- Evidencia empírica de la **dinámica interna del WO** (atribución del estancamiento a sus señales de control).
- Marco **general y reusable**, agnóstico del algoritmo.

**Hallazgo honesto:** en calidad de solución, **WO+SHAP ≈ WO base** (no significativo); el aporte demostrado es la **interpretabilidad**, no la mejora de fitness.

**Énfasis:** el resultado negativo en calidad es un hallazgo científico válido; el valor está en *abrir la caja negra*.

---

## Slide 15 — Trabajo futuro: problemas, riesgos y oportunidades

**En pantalla:**

**Problemas y riesgos identificados**
- No mejora la calidad de forma significativa (hipótesis de mejora **no confirmada**).
- **Señales globales:** SHAP indica *qué* palanca pesa, no *dónde* reposicionar.
- La **modulación guiada degenera** (umbral 0,90 / amplificación 2,0).
- Límites estadísticos: shap a 5·10⁶ con **n=5**; sin corrección por comparaciones múltiples; sin ranking de Friedman (solo 2 condiciones).
- Alcance acotado: dim 10, CEC2022; TMLAP sin óptimo exacto (solo la "dura" ejercita al WO).

**Oportunidades y propuestas**
- **Señales por-agente (locales)** → que SHAP indique *dónde* reposicionar (vía más prometedora para mejorar la calidad).
- Recalibrar **umbral de dominancia** y **factor de amplificación**; explorar umbrales adaptativos.
- Ampliar evidencia: **CEC 2014/2017** (72 problemas), dim 20; completar shap@5·10⁶ a 30 corridas.
- Comparar contra **competidores** (EA4eig, LSHADE, AGSK…) y el meta-nivel **MsMA (Črepinšek 2025)** → habilita el ranking de **Friedman**.
- Correcciones por comparaciones múltiples (Holm/Bonferroni); generalizar a otras metaheurísticas; medir **diversidad poblacional**; cerrar más el lazo (modular en continuo, no solo elegir rama).

**Énfasis:** ordenar éxito → riesgo → oportunidad; el negativo en calidad **abre** la siguiente línea (señales locales).

---

## Slide 16 — Cierre

**En pantalla:**
- **Problema** → metaheurísticas que se estancan y son cajas negras.
- **Solución** → controlador SHAP en línea sobre las señales del WO (una acción bifurcada).
- **Hallazgo** → no mejora el fitness de forma significativa, pero **explica** la dinámica interna; el aporte es la interpretabilidad.
- ¡Gracias! Preguntas.

**Énfasis:** cerrar en ≤15 min; dejar 2-3 min para preguntas.

---

## Bibliografía (slide opcional / backup)

Talbi 2009 · Engelbrecht 2007 · Somvanshi et al. 2025 · Garey & Johnson 1979 · Crawford 2023 ·
Sørensen & Glover 2013 · Das et al. 2022 · Kalasampath et al. 2025 · Burke et al. 2013 ·
Sampieri et al. 2014 · Londoño Palacio et al. 2014 · Cao & Tang 2018 · Črepinšek et al. 2025 (estancamiento / MsMA) ·
**Lundberg & Lee 2017 (SHAP)** · **Han et al. 2024 (Walrus Optimizer)**.

---

## Pendientes antes de presentar
1. **Screenshot** de la sesión SSH / log para slide 13 (evidencia de ambiente).
2. **Confirmar specs** exactas del servidor (slide 12).
3. **Validar** los 4 objetivos específicos contra el anteproyecto formal (slide 7).
4. (Opcional) Completar `shap` a 5M con 30 corridas para cerrar el veredicto a 5M.
