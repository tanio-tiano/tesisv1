# Walrus Optimizer Base - CEC 2022

Este repositorio queda enfocado en una sola linea experimental:

- `WO_base_cec`: Walrus Optimizer base.
- `opfunu_cec_adapter.py`: adaptador para instanciar funciones CEC 2022 desde `opfunu`.

## Requisitos

Dependencias principales:

- `numpy`
- `pandas`
- `scipy`
- `opfunu`

## Ejecucion

```powershell
cd "WO_base_cec"
python run_wo_base_cec2022_multirun.py --agents 30 --iterations 500 --runs 30 --gbest-mode legacy --output all_functions_outputs_30runs_legacy_20260429
```

## Salidas

- `values/summary_wo_base_cec2022_30x500_30runs_legacy.csv`
- `values/statistics_wo_base_cec2022_30x500_30runs_legacy.csv`
- `values/conv_curve_Fx_runY.csv`
- `values/result_wo_base_Fx_runY.csv`
