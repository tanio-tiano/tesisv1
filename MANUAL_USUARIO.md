# Manual de usuario — Walrus Optimizer con controlador interpretable (SHAP)

**Proyecto:** EXPLICABILIDAD EN ALGORITMOS BIO-INSPIRADOS: DISEÑO DE UN CONTROLADOR PARA MITIGAR
ESTANCAMIENTO Y CONVERGENCIA PREMATURA.
**Autor:** Lucciano Fabián Cáceres Costa · Universidad de Valparaíso, Escuela de Ingeniería Informática.
**Convertir a PDF:** `python -m analysis.md_to_pdf MANUAL_USUARIO.md MANUAL_USUARIO.pdf`

---

## 1. Propósito y objetivos

El sistema implementa el **Walrus Optimizer (WO)** —una metaheurística poblacional bio-inspirada—
acoplado a un **controlador interpretable en línea** basado en **SHAP** (valores de Shapley
exactos). El controlador observa las señales internas del WO durante la ejecución, detecta cuándo
un agente se **estanca** y decide una acción de rescate explicada por SHAP.

- **Objetivo general:** desarrollar un controlador interpretable, basado en explicabilidad
  in-ejecución, que monitoree indicadores del WO para **mitigar el estancamiento**.
- **Objetivo específico 3:** generar **interpretaciones y trazabilidad** del comportamiento de la
  metaheurística (qué señal explica el estancamiento).
- **Objetivo específico 4:** **integrar** la metaheurística con el componente de explicabilidad y
  **evaluar su impacto** (calidad de soluciones, estancamiento, interpretabilidad).

## 2. Qué hace el sistema

Se ejecuta en dos **modos**, que se comparan de forma pareada (misma semilla por corrida):

- **`base`** — el WO solo (referencia).
- **`shap`** — el WO + controlador SHAP. En cada iteración el controlador:
  1. **Detecta** estancamiento por agente (un agente sin mejorar su mejor personal por ≥10 % del
     presupuesto), usando un contador barato de evaluaciones (FES), sin SHAP.
  2. **Explica** con SHAP exacto (64 coaliciones sobre las 6 señales del WO) el fitness del agente
     estancado.
  3. **Decide** una única acción bifurcada según la explicación:
     - **Rama A (`reinit_random`):** reinicio uniforme del agente.
     - **Rama B (`reinit_guided`):** un paso del WO con la señal dominante amplificada.
  4. **Reposiciona** al agente (siempre), preservando el mejor global.

El criterio de parada es **MaxFES** (número máximo de evaluaciones de la función objetivo). El
costo de SHAP también se descuenta del presupuesto, de modo que la comparación `base` vs `shap`
es justa.

## 3. Requisitos

**Software**
- **Python 3.11 o superior.**
- Dependencias (`pip install -r requirements.txt`): `numpy`, `pandas`, `scipy`, `opfunu`
  (`opfunu` es imprescindible para el benchmark CEC 2022).

**Hardware**
- **Mínimo:** cualquier CPU x86/ARM, ~4 GB de RAM (suficiente para corridas individuales).
- **Recomendado (barrido completo):** CPU multinúcleo y ≥8–16 GB de RAM. El costo no está en la
  memoria sino en el **número de corridas × MaxFES** (el presupuesto de 5·10⁶ es el más caro).

## 4. Instalación

```bash
# desde la raíz del repositorio (carpeta que contiene runners/, wo_core/, experiments/)
python -m venv .venv && source .venv/bin/activate      # opcional
pip install -r requirements.txt
```

## 5. Estructura del proyecto

```
W.O Python/
├── wo_core/            Dinámica del WO (movimiento, señales, roles) + value function por agente
├── shap_controller/    Controlador SHAP: detección, explicación, política y acciones
│   ├── controller.py   SHAPFitnessController (gates, explain_fitness, decide)
│   ├── profiles.py     Configuración ÚNICA (parámetros como fracciones de MaxFES)
│   ├── features.py     Las 6 señales de control = features de SHAP
│   └── actions.py      reinit_random (Rama A) / reinit_guided (Rama B)
├── problems/           Adaptadores de problema (contrato WOProblem): cec2022.py, tmlap.py
├── runners/
│   └── run_ablation.py Runner ÚNICO: corre base + shap pareados en régimen MaxFES
├── analysis/           Post-proceso: tablas, figuras y reportes
└── experiments/        Salidas (una carpeta por configuración)
```

## 6. Uso — ejecutar experimentos

Todo se corre **desde la raíz del repositorio**. El runner único es `runners.run_ablation`.

```bash
python -m runners.run_ablation \
    --problem <familia>:<objetivo> \
    --modes base,shap \
    --max-fes <presupuesto(s)> \
    --runs <N> --agents 30 --dim 10 --seed 1234 \
    --output <carpeta_salida>
```

**Parámetros principales:**

| Flag | Default | Para qué |
|---|---|---|
| `--problem` | (requerido) | `cec2022:F6`, `cec2022:all`, o `tmlap:3.instancia_dura.txt` |
| `--modes` | `base,shap` | Modos a correr (pareados por semilla) |
| `--max-fes` | `50000` | Presupuesto(s) MaxFES; admite lista: `5000,50000,500000,5000000` |
| `--runs` | `51` | Corridas independientes por configuración |
| `--dim` | `10` | Dimensión (solo CEC) |
| `--agents` | `30` | Tamaño de población |
| `--init-mode` | `local_search` | TMLAP: `random` para instancias grandes |
| `--no-exact-optimum` | (off) | TMLAP sin óptimo exacto (gap = NaN) |
| `--seed` | `1234` | Semilla raíz (pareada base↔shap) |
| `--output` | (requerido) | Carpeta base de salida |

**Ejemplos:**

```bash
# CEC 2022: las 12 funciones, 4 presupuestos, 30 corridas
python -m runners.run_ablation --problem cec2022:all --modes base,shap \
    --max-fes 5000,50000,500000,5000000 --runs 30 --dim 10 \
    --output experiments/cec2022_d10

# TMLAP instancia dura, 30 corridas, MaxFES=500k (sin óptimo exacto)
python -m runners.run_ablation --problem tmlap:3.instancia_dura.txt --modes base,shap \
    --max-fes 500000 --runs 30 --no-exact-optimum --init-mode random \
    --output experiments/dura_30_fes500k
```

> Para correr en un servidor en segundo plano (sobreviviendo a la desconexión SSH) hay dos
> scripts envoltorio descritos en `COMO_EJECUTAR.txt`: `run_remote_cec.sh` (barrido CEC) y
> `run_remote.sh` (TMLAP dura), que llaman al mismo runner con `nohup`.

## 7. Configuración del controlador

El controlador usa **una sola configuración** (en `shap_controller/profiles.py`), válida en todos
los presupuestos sin re-tuning: todos los parámetros temporales son **fracciones de MaxFES**.

| Parámetro | Valor | Función |
|---|---|---|
| Ventana de estancamiento | 10 % de MaxFES | detectar agente estancado (90 % conf.) |
| Guard inicial | 5 % de MaxFES | no intervenir en el arranque |
| Límite tardío | 95 % de MaxFES | no intervenir cerca del cierre |
| Cooldown por agente / efectivo | 5 % de MaxFES | espaciar intervenciones |
| Presupuesto SHAP | 5 % de MaxFES | tope de FES gastables en explicaciones |
| Umbral de dominancia | 0.90 | elegir Rama B (guiada) vs Rama A (aleatoria) |
| Factor de amplificación | 2.0 | intensidad de la señal dominante en Rama B |
| Pasos Shapley | 3 | simulación por coalición (costo SHAP = 64×3 = 192 FES) |

## 8. Salidas que produce

Por modo, dentro de `<salida>/<modo>/`:

- **`values/summary.csv`** — 1 fila por corrida. Columnas clave: `final_fitness`, `gap_to_optimum`,
  `interventions`, `shap_explanations`, `t_init_seconds`, `t_shap_seconds`, telemetría de
  estancamiento (`stagnation_window`, `n_stagnant_at_end`, `mean/max_fes_since_improve_at_end`),
  `n_reinit_random`, `n_reinit_guided`, y los *buckets* de FES (`fes_search/shap/intervention/init`).
- **`curves/conv_curve_F<n>_fes<M>_run<r>.csv`** — curva de convergencia (`fes`, `best_fitness`).
- (Solo `shap`) **`values/controller_events.csv`** — una fila por intervención.
- (Solo `shap`) **`values/controller_non_events.csv`** — candidatos bloqueados por una compuerta.
- (Solo `shap`) **`values/shap_values.csv`** — las 6 atribuciones SHAP + señal dominante por explicación.

## 9. Análisis y figuras

```bash
# Resumen condensado para presentación (tablas + figuras + veredicto)
python -m analysis.presentation_summary --input experiments \
    --out-dir experiments/presentacion --highlight-fes 500000

# Reporte HTML detallado de una configuración
python -m analysis.report_html --input experiments/cec2022_d10_fes500000 \
    --output reporte.html

# Diagramas de diseño/implementación
python -m analysis.make_diagrams --output Diagramas
```

`presentation_summary` genera, en `experiments/presentacion/`: el veredicto Wilcoxon
(`veredicto.md`), la tabla por función (`tabla_resumen_fes500000.png`), boxplots, efecto de MaxFES,
las curvas de convergencia con banda ±std (`convergencia/`), la actividad del controlador
(`acciones_rescate.png`, `features_shap.png`) y las tablas de Shapiro–Wilk + Wilcoxon
(`shapiro_wilcoxon_*.png`).

## 10. Interpretación de resultados

- **`gap_to_optimum`** (CEC): distancia al óptimo conocido; menor es mejor. En TMLAP no hay óptimo
  exacto, se usa `final_fitness` (costo a minimizar).
- **Notación `+ / = / −`** (columna `H`): `+` = shap mejor, `=` sin diferencia significativa,
  `−` = shap peor, según **Wilcoxon signed-rank pareado** (α = 0.05).
- **`(W|T|L)`** = nº de funciones donde shap gana | empata | pierde.
- **Shapiro–Wilk**: si los p-valores de ambos grupos son < 0.05 las distribuciones **no son
  normales**, lo que justifica usar el test no paramétrico (Wilcoxon).
- **Actividad del controlador**: qué señal explica el estancamiento (señal dominante) y el reparto
  de acciones de rescate (random vs guiada).

## 11. Solución de problemas

- *Falta `opfunu`:* `pip install opfunu`.
- *Error "ejecuta desde la raíz del repo":* posicionarse en la carpeta que contiene `runners/`,
  `wo_core/`, `experiments/` y las instancias TMLAP.
- *Wall-clock muy largo:* el cuello de botella es `--max-fes 5000000`. Para iterar rápido, usar
  presupuestos bajos (`--max-fes 5000,50000`) y validar antes de lanzar el más grande.
- *`gap` NaN en TMLAP:* es esperado cuando se usa `--no-exact-optimum` (la instancia dura no tiene
  óptimo exacto); el análisis usa `final_fitness` en ese caso.

> Para el detalle operativo completo (scripts de servidor, columnas de `summary.csv`, flujo de una
> sesión, logs) ver `COMO_EJECUTAR.txt`.
