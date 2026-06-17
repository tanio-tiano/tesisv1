# Walrus Optimizer + Controlador SHAP

Implementacion unica del Walrus Optimizer (Han et al. 2024) con un controlador
**SHAP por agente** on-line, aplicable a cualquier problema que cumpla el contrato
`WOProblem`. Hoy hay dos adaptadores: CEC 2022 (12 funciones continuas) y TMLAP
(asignacion discreta de clientes a hubs).

El criterio de parada es **MaxFES** y la comparacion es una **ablacion pareada
`base` vs `shap`** (misma semilla por corrida) corrida por un **unico runner**.

## Estructura

```
W.O Python/
├── wo_core/                ← dinamica WO unica (paper Han et al. 2024)
│   ├── walrus.py           apply_wo_movement, apply_wo_movement_single, signals, role_counts
│   ├── agent_sim.py        value function que simula UN agente (para SHAP por agente)
│   ├── halton.py, levy_flight.py, initialization.py, diversity.py
│
├── shap_controller/        ← controlador SHAP exacto por agente + politica + cooldowns
│   ├── controller.py       SHAPFitnessController (deteccion, explain_fitness, decide)
│   ├── profiles.py         configuracion UNICA (fracciones de MaxFES; sin perfiles)
│   ├── features.py         FEATURE_COLUMNS (6 senales) + baselines neutros
│   ├── actions.py          reinit_random (Rama A) / reinit_guided (Rama B)
│
├── problems/               ← adaptadores; cumplen WOProblem
│   ├── base.py             contrato Protocol
│   ├── cec2022.py          envuelve opfunu
│   ├── tmlap.py            TMLAPProblem + load_problem + backtracking
│   ├── factory.py          parse_problem_spec("cec2022:F6") / ("tmlap:path")
│
├── runners/
│   └── run_ablation.py     runner UNICO: WO base + WO+SHAP en regimen MaxFES, pareado
│
├── analysis/               ← post-corrida
│   ├── presentation_summary.py  resumen condensado para slides (piramide de agregacion)
│   ├── report_html.py      reporte HTML estandarizado
│   ├── normality.py        Shapiro-Wilk + Anderson-Darling (+ D'Agostino)
│   ├── make_diagrams.py    diagramas de arquitectura/implementacion
│   ├── md_to_pdf.py        conversor Markdown -> PDF
│
├── data/                   ← instancias TMLAP
│   ├── 1.instancia_simple.txt   (6c x 3h)
│   ├── 2.instancia_mediana.txt  (12c x 5h)
│   ├── 3.instancia_dura.txt     (24c x 8h)
│   └── 4.instancia_grande.txt   (1000c x 500h)
│
├── scripts/                ← lanzadores remotos (nohup + barrido de MaxFES)
│   ├── run_remote.sh       TMLAP dura
│   └── run_remote_cec.sh   CEC2022 con sweep de presupuestos
│
├── docs/                   ← documentacion viva
│   ├── Informe_Metodologia.md    metodologia tecnica
│   ├── INFORME_2.md
│   ├── MANUAL_USUARIO.md
│   └── Presentacion_2.md
│
├── _archive/               ← codigo huerfano preservado por trazabilidad
│
└── experiments/            ← salidas: <config>/{base,shap}/values + /curves
```

## Dependencias

```bash
pip install -r requirements.txt
```

`numpy`, `pandas`, `scipy`, `opfunu`.

## Como correr (runner unico)

```bash
python -m runners.run_ablation \
    --problem <familia>:<target> \
    --modes base,shap \
    --max-fes <presupuesto(s)> \
    --runs <N> --agents 30 --dim 10 --seed 1234 \
    --output <carpeta_salida>
```

Donde `<familia>:<target>` puede ser:

- `cec2022:F6` para correr solo F6 (igual F1..F12).
- `cec2022:all` para las 12 funciones en una sola invocacion.
- `tmlap:3.instancia_dura.txt` (busqueda con fallback automatico a `data/`),
  o explicito `tmlap:data/3.instancia_dura.txt`.

`--max-fes` acepta varios presupuestos separados por coma (se corren todos):
`--max-fes 5000,50000,500000,5000000`.

## Ejemplos concretos

### CEC 2022 — ablacion base vs shap sobre F1..F12 (30 corridas, 4 MaxFES)

```bash
python -m runners.run_ablation \
    --problem cec2022:all \
    --modes base,shap \
    --max-fes 5000,50000,500000,5000000 \
    --runs 30 --agents 30 --dim 10 --seed 1234 \
    --output experiments/cec2022_d10
```

### TMLAP — instancia dura (sin optimo exacto)

```bash
python -m runners.run_ablation \
    --problem tmlap:3.instancia_dura.txt \
    --modes base,shap \
    --max-fes 50000 \
    --runs 51 --agents 30 --seed 1234 \
    --no-exact-optimum --init-mode random \
    --output experiments/dura_51_fes50k
```

## Flags utiles

| Flag | Default | Aplica a | Para que |
|------|---------|----------|----------|
| `--modes` | `base,shap` | todo | Modos a correr (pareados por semilla) |
| `--max-fes` | `50000` | todo | Presupuesto(s) MaxFES, separados por coma |
| `--runs` | `51` | todo | Corridas independientes por configuracion |
| `--dim` | `10` | CEC | Dimensionalidad del benchmark |
| `--agents` | `30` | todo | Tamano de poblacion |
| `--init-mode` | `local_search` | TMLAP | `random` salta local_search (instancias grandes) |
| `--no-exact-optimum` | (off) | TMLAP | No calcular optimo exacto (instancias intratables): gap=NaN |
| `--clients` / `--hubs` | — | TMLAP | Override de tamano de instancia |

> El controlador NO se parametriza por flags: su configuracion es **unica** y vive
> en `shap_controller/profiles.py` como fracciones de MaxFES (ventana de
> estancamiento 10%; guard/cooldowns/shap_budget 5%; late 95%; **umbral de
> dominancia 0.50** — mayoria simple; amplificacion 2.0; 3 pasos Shapley).

## Salidas estandar

El runner produce, por modo, dentro de `<output>/<modo>/`:

- `values/summary.csv`: 1 fila por corrida (`final_fitness`, `gap_to_optimum`,
  `interventions`, `shap_explanations`, `t_init_seconds`, `t_shap_seconds`,
  telemetria de estancamiento, `n_reinit_random`, `n_reinit_guided`, buckets de FES).
- `curves/conv_curve_F<n>_fes<M>_run<r>.csv`: curva de convergencia (`fes, best`).
- (solo `shap`) `values/controller_events.csv`: una fila por intervencion.
- (solo `shap`) `values/controller_non_events.csv`: candidatos bloqueados (motivo).
- (solo `shap`) `values/shap_values.csv`: las 6 atribuciones SHAP + dominante por explicacion.

Resumen para presentacion / analisis:

```bash
python -m analysis.presentation_summary --input experiments --out-dir experiments/presentacion
```

## Reproducibilidad

Las semillas se computan como
`seed + max_fes_idx*100000 + problem_idx*1000 + run_idx`
(**independiente del modo**), lo que garantiza:

1. `base` y `shap` usan **la misma semilla** por cada `run_id` → Wilcoxon pareado valido.
2. Corridas distintas de la misma funcion usan semillas distintas.
3. Misma `--seed` reproduce exactamente los resultados (misma version de Python/numpy).
