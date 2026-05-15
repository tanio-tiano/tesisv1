# TMLAP - Instancia "grande" (1000 clientes x 500 hubs)

Stress test pareado de WO_base vs WO_shap sobre una instancia grande pero
estructuralmente analoga a las pequenas (simple, mediana, dura). Todo lo
necesario para correrla vive aqui dentro; se puede mover esta carpeta a otro
servidor y ejecutar.

## Estructura

```
experiments/tmlap_grande/
├── README.md                 (este archivo)
├── generate_instance.py      (genera 4.instancia_grande.txt; seed fijo)
├── 4.instancia_grande.txt    (la instancia ya generada)
├── run.sh                    (runner Linux/macOS)
├── run.bat                   (runner Windows)
└── outputs/                  (se crea al correr; contiene los CSV/HTML)
```

## Dependencias

Las del proyecto base, listadas en `../../requirements.txt`:

- `numpy`, `pandas`, `scipy`, `opfunu==1.0.4`

En un servidor limpio:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r ../../requirements.txt
```

## Parametros de la instancia

| Parametro             | Valor              | Logica                                          |
|-----------------------|--------------------|-------------------------------------------------|
| `n_clientes`          | 1000               | ratio 2:1 c:h, igual a `mediana`                |
| `n_hubs`              | 500                | conserva tu cantidad pedida                     |
| Distancias            | enteros [4, 12]    | mismo rango que las chicas                      |
| Costos fijos          | enteros [15, 25]   | mismo rango que las chicas                      |
| Capacidades por hub   | enteros [2, 3]     | suma ~ 1.25x clientes (slack como chicas)       |
| `D_max`               | 9                  | mismo ratio D_max / mediana_dist ~ 1.1x         |

## Ejecucion rapida

**Linux / macOS:**

```bash
cd experiments/tmlap_grande
bash run.sh                 # 1 corrida (default)
RUNS=30 bash run.sh         # 30 corridas pareadas
RUNS=10 PROFILE=hard bash run.sh   # cambiar perfil del controlador
```

**Windows:**

```cmd
cd experiments\tmlap_grande
run.bat
set RUNS=30 && run.bat
```

## Que hace internamente el runner

Llama al runner unificado del proyecto:

```
python -m runners.compare_base_vs_shap \
    --problem tmlap:<PATH>/4.instancia_grande.txt \
    --runs <RUNS> --agents 30 --iterations 300 --seed 1234 \
    --profile soft --shapley-steps 3 --init-mode random \
    --output outputs/run_<RUNS>r
```

Detalles del runner:

- **Misma arquitectura para CEC y TMLAP.** El runner llama a
  ``problems.parse_problem_spec`` para construir el adaptador adecuado.
- **Semillas pareadas.** WO_base y WO_shap usan exactamente la misma semilla
  por cada ``run_id`` -> Wilcoxon signed-rank emparejado valido.
- **`--init-mode random`** salta `local_search` en la inicializacion (necesario
  para que esta instancia grande termine en tiempo razonable).
- **`--profile soft`** por defecto (cambiar via env var `PROFILE`).
- **`--shapley-steps 3`** simulaciones de WO por cada coalicion Shapley.
- **`--agents 30 --iterations 300`** (cambiable por env vars).

## Salidas

```
outputs/run_<N>r/
├── values/
│   ├── summary_base.csv        (1 fila por corrida WO_base)
│   ├── summary_shap.csv        (1 fila por corrida WO_shap)
│   ├── statistics.csv          (best/worst/mean/std por algoritmo)
│   ├── paired.csv              (1 fila por par base/shap con delta)
│   ├── tests.csv               (Wilcoxon + sign test; valido con N >= 5)
│   ├── controller_events.csv   (eventos del controlador SHAP)
│   └── shap_values.csv         (atribuciones SHAP por evento)
└── curves/
    ├── conv_curve_<problema>_run1_base.csv
    └── conv_curve_<problema>_run1_shap.csv
```

## Estimaciones de tiempo

Sobre un CPU moderno (3-4 GHz, 1 nucleo activo):

| RUNS | Tiempo base | Tiempo SHAP | Total estimado |
|------|-------------|-------------|----------------|
| 1    | ~1 min      | ~5 min      | **~6 min**     |
| 10   | ~10 min     | ~50 min     | **~1 h**       |
| 30   | ~30 min     | ~2.5 h      | **~3 h**       |

El cuello de botella es WO_shap: cada explicacion enumera 64 coaliciones y
simula `--shapley-steps` pasos del WO por cada una.

Para acelerar en multinucleo: lanzar varios `RUNS=10 SEED=X bash run.sh &` en
carpetas distintas y luego concatenar los `summary_*.csv`.

## Modo de uso en otro servidor

1. Copiar el proyecto completo (incluye `wo_core/`, `shap_controller/`,
   `problems/`, `runners/`, `experiments/tmlap_grande/`). El runner vive en
   `runners/compare_base_vs_shap.py` y requiere los modulos `wo_core`,
   `shap_controller` y `problems`.
2. Crear venv e instalar dependencias (ver arriba).
3. Ejecutar `bash run.sh` (Linux) o `run.bat` (Windows).
4. Para procesar localmente, copiar solo `outputs/run_<N>r/`.

## Limpieza

```bash
rm -rf outputs/
rm 4.instancia_grande.txt   # opcional: el generador lo recrea de forma determinista
```
