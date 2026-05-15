# experiments/

Carpeta para todos los resultados experimentales, datos generados y scripts
de analisis. Separada del codigo base (`WO_base_cec/`, `WO_shap_cec/`,
`WO_tmlap_base/`, `WO_tmlap_shap/`, `PSO_tmlap_compare/`) para poder limpiar
sin afectar la implementacion.

## Estructura

```
experiments/
├── README.md                       (este archivo)
├── cec_baseline_30runs/            (CEC 2022, WO base, 30 corridas con guia "updated")
│   ├── values/                     (CSV agregados, statistics, paper_statistics, normality)
│   ├── plots/                      (PNG: boxplots, convergencia, tiempos por funcion)
│   ├── report.html                 (reporte estandar del experimento)
│   └── reporte_visual.html
├── tmlap_smaller_instances/        (comparaciones WO_base vs WO_shap, n=30)
│   ├── simple/                     (6 clientes x 3 hubs)
│   ├── mediana/                    (12 clientes x 5 hubs)
│   └── dura/                       (24 clientes x 8 hubs)
├── tmlap_grande/                   (stress test 1000 clientes x 500 hubs - separable)
│   ├── README.md                   (ver dentro para instrucciones de runner / servidor)
│   ├── generate_instance.py
│   ├── 4.instancia_grande.txt
│   ├── run.sh, run.bat
│   └── outputs/                    (se crean al correr)
├── tmlap_effectiveness.py          (script: tasa improved/neutral/worsened por evento)
└── tmlap_intervention_effectiveness.py
```

## Como reproducir los resultados

### CEC 2022 - WO base (carpeta `cec_baseline_30runs/`)

```bash
cd WO_base_cec
python run_wo_base_cec2022_multirun.py \
    --agents 30 --iterations 500 --runs 30 --seed 1234 \
    --output ../experiments/cec_baseline_30runs
```

### TMLAP - instancias chicas (carpeta `tmlap_smaller_instances/`)

```bash
# Desde la raiz del proyecto
python WO_tmlap_shap/compare_tmlap_base_vs_online_shap.py \
    --instance 1.instancia_simple.txt \
    --runs 30 --agents 30 --iterations 300 --seed 1234 --profile soft \
    --output experiments/tmlap_smaller_instances/simple
# Idem con 2.instancia_mediana.txt y 3.instancia_dura.txt
```

### TMLAP - instancia grande (carpeta `tmlap_grande/`)

```bash
cd experiments/tmlap_grande
bash run.sh                 # 1 corrida
RUNS=30 bash run.sh         # full
```

Ver `experiments/tmlap_grande/README.md` para detalles.

## Scripts de analisis

- `tmlap_effectiveness.py` - calcula tasa `improved`/`neutral`/`worsened` por
  evento del controlador SHAP a partir de los CSV en cada subcarpeta TMLAP.
- `tmlap_intervention_effectiveness.py` - desglosa la efectividad por accion
  y por feature SHAP dominante.

Uso:

```bash
python experiments/tmlap_effectiveness.py
python experiments/tmlap_intervention_effectiveness.py
```

## Limpieza

```bash
# Para limpiar todo y dejar solo lo necesario para regenerar:
rm -rf experiments/cec_baseline_30runs experiments/tmlap_smaller_instances
rm -rf experiments/tmlap_grande/outputs
```
