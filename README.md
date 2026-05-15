# Walrus Optimizer + Controlador SHAP

Implementacion unica del Walrus Optimizer (Han et al. 2024) con un controlador
SHAP on-line aplicable a cualquier problema que cumpla el contrato ``WOProblem``.
Hoy hay dos adaptadores: CEC 2022 (12 funciones continuas) y TMLAP (asignacion
discreta de clientes a hubs).

## Estructura

```
W.O Python/
├── wo_core/                ← dinamica WO unica (paper Han et al. 2024)
│   ├── walrus.py           apply_wo_movement, walrus_role_counts, signals, ...
│   ├── halton.py, levy_flight.py, initialization.py, diversity.py
│
├── shap_controller/        ← controlador SHAP exacto + politica + cooldowns
│   ├── controller.py       SHAPFitnessController
│   ├── profiles.py         ControllerProfile / PROFILE_DEFAULTS (soft/medium/hard)
│   ├── features.py         FEATURE_COLUMNS, baselines
│   ├── actions.py          partial_restart, random_reinjection (+ modos)
│
├── problems/               ← adaptadores; cumplen WOProblem
│   ├── base.py             contrato Protocol
│   ├── cec2022.py          envuelve opfunu
│   ├── tmlap.py            TMLAPProblem + load_problem + backtracking
│   ├── factory.py          parse_problem_spec("cec2022:F6") / ("tmlap:path")
│
├── runners/                ← 3 runners parametrizados por problema
│   ├── run_wo_base.py      WO sin controlador
│   ├── run_wo_shap.py      WO + controlador SHAP
│   └── compare_base_vs_shap.py   pareado, mismas semillas, Wilcoxon
│
├── analysis/               ← post-corrida
│   ├── normality.py        Shapiro-Wilk + Anderson-Darling
│   ├── effectiveness.py    tasa improved/neutral/worsened
│   ├── intervention_effectiveness.py
│   ├── reporting.py        reportes HTML estandarizados
│   ├── dashboard.py        dashboard visual
│
├── 1.instancia_simple.txt  (6c x 3h)
├── 2.instancia_mediana.txt (12c x 5h)
├── 3.instancia_dura.txt    (24c x 8h)
│
└── experiments/            ← salidas de experimentos + tmlap_grande/
    ├── README.md
    ├── cec_baseline_30runs/        (resultados WO base CEC2022, 30 corridas)
    ├── tmlap_smaller_instances/    (simple/mediana/dura, base vs shap, 30 corridas)
    └── tmlap_grande/
        ├── README.md                instrucciones para correr en servidor
        ├── generate_instance.py     1000c x 500h, parametros analogos a las chicas
        ├── 4.instancia_grande.txt
        ├── run.sh, run.bat          one-shot runner
        └── outputs/
```

## Dependencias

```bash
pip install -r requirements.txt
```

`numpy`, `pandas`, `scipy`, `opfunu==1.0.4`.

## Como correr cualquier experimento

**Estructura del CLI** (sirve para CEC y TMLAP indistintamente):

```bash
python -m runners.run_wo_base \
    --problem <familia>:<target> \
    --runs <N> --agents 30 --iterations <T> --seed 1234 \
    --output <carpeta_salida>

python -m runners.run_wo_shap \
    --problem <familia>:<target> \
    --runs <N> --agents 30 --iterations <T> --seed 1234 \
    --profile soft --shapley-steps 3 \
    --output <carpeta_salida>

python -m runners.compare_base_vs_shap \
    --problem <familia>:<target> \
    --runs <N> --agents 30 --iterations <T> --seed 1234 \
    --profile soft --shapley-steps 3 \
    --output <carpeta_salida>
```

Donde `<familia>:<target>` puede ser:

- `cec2022:F6` para correr solo F6 (igualmente F1..F12).
- `cec2022:all` para correr las 12 funciones en una sola invocacion.
- `tmlap:1.instancia_simple.txt` (o cualquier otra instancia).

## Ejemplos concretos

### CEC 2022 - WO base sobre F1..F12 (30 corridas)

```bash
python -m runners.run_wo_base \
    --problem cec2022:all \
    --runs 30 --agents 30 --iterations 500 --seed 1234 \
    --output experiments/cec_baseline_30runs
```

### CEC 2022 - WO + controlador SHAP sobre F6

```bash
python -m runners.run_wo_shap \
    --problem cec2022:F6 \
    --runs 30 --agents 30 --iterations 500 --seed 1234 \
    --profile soft --shapley-steps 3 \
    --output experiments/cec_F6_shap_soft
```

### TMLAP - comparacion pareada base vs shap sobre instancia mediana

```bash
python -m runners.compare_base_vs_shap \
    --problem tmlap:2.instancia_mediana.txt \
    --runs 30 --agents 30 --iterations 300 --seed 1234 \
    --profile soft --shapley-steps 3 \
    --output experiments/tmlap_mediana_compare_30runs
```

### TMLAP - instancia grande (server-ready)

```bash
cd experiments/tmlap_grande
RUNS=30 bash run.sh
```

Ver `experiments/tmlap_grande/README.md` para detalles.

## Flags utiles

| Flag | Default | Aplica a | Para que |
|------|---------|----------|----------|
| `--profile` | `soft` | run_wo_shap, compare | Perfil del controlador (soft/medium/hard) |
| `--init-mode` | `local_search` | TMLAP | Usar `random` para instancias grandes (>= 100 clientes) |
| `--shapley-steps` | 3 | run_wo_shap, compare | Pasos del WO simulados por coalicion Shapley |
| `--acceptance-mode` | `diversity` | run_wo_shap, compare | Compuerta de aceptacion del rescate |
| `--dim` | 10 | CEC | Dimensionalidad del benchmark |

## Salidas estandar

Cualquier runner produce dentro de `<output>/values/`:

- `summary.csv` o `summary_{base,shap}.csv`: 1 fila por corrida.
- `statistics.csv`: best/worst/mean/median/std por problema.
- `paired.csv`: 1 fila por par base/shap con `delta_base_minus_shap` (solo compare).
- `tests.csv`: Wilcoxon signed-rank emparejado + sign test (solo compare).
- `controller_events.csv`: eventos del controlador SHAP (solo shap/compare).
- `shap_values.csv`: atribuciones SHAP por evento (solo shap/compare).
- `curves/conv_curve_*.csv`: curvas de convergencia por (problema, run, algoritmo).

## Reproducibilidad

Las semillas se computan como `seed_base + problem_idx*1000 + (run_id - 1)`,
lo que garantiza:

1. WO_base y WO_shap usan **la misma semilla** por cada `run_id` -> Wilcoxon
   emparejado valido.
2. Corridas distintas de la misma funcion usan semillas distintas.
3. Corridas con la misma `seed_base` reproducen exactamente los resultados
   (mientras la version de Python y numpy no cambien).
