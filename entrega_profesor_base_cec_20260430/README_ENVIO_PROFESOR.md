# Entrega WO base CEC 2022

Esta carpeta contiene el algoritmo base y los resultados necesarios para revisar la distribucion de las ejecuciones.

## Configuracion experimental

- Algoritmo: Walrus Optimizer base.
- Benchmark: CEC 2022 mediante `opfunu`.
- Funciones: F1-F12.
- Dimension: 10.
- Corridas independientes: 30 por funcion.
- Agentes: 30.
- Iteraciones: 500.
- Modo: `gbest-mode legacy`, alineado con el comportamiento literal del codigo MATLAB publico revisado.
- Semilla base: 1234.

## Codigo

La carpeta `codigo` incluye:

- `opfunu_cec_adapter.py`: adaptador para CEC 2022 con `opfunu`.
- `WO_base_cec/run_wo_base_cec2022_multirun.py`: runner principal del WO base.
- `WO_base_cec/analyze_normality_wo_base_cec2022.py`: script del analisis de normalidad.
- `WO_base_cec/halton.py`, `initialization.py`, `levy_flight.py`: funciones auxiliares del algoritmo.
- `requirements.txt`: dependencias minimas.

## Resultados

La carpeta `resultados` incluye:

- `summary_wo_base_cec2022_30x500_30runs_legacy.csv`: resultados crudos por corrida y funcion. Es el archivo principal para revisar las ejecuciones.
- `statistics_wo_base_cec2022_30x500_30runs_legacy.csv`: estadistica descriptiva agregada por funcion.
- `shapiro_simple_wo_base_cec2022_30runs_legacy.csv`: analisis simplificado de normalidad por Shapiro-Wilk.
- `shapiro_simple_wo_base_cec2022_30runs_legacy.html`: reporte HTML simplificado del analisis Shapiro-Wilk.

## Normalidad

Se aplico Shapiro-Wilk con alpha = 0.05 sobre `final_fitness`.

Regla de decision:

- Si `shapiro_p >= 0.05`, no se rechaza normalidad.
- Si `shapiro_p < 0.05`, se rechaza normalidad.

Las funciones compatibles con normalidad fueron:

- F4: Shifted and Rotated Non-Continuous Rastrigin's Function.
- F6: Hybrid Function 1.

Las otras funciones rechazaron normalidad en el analisis por funcion, por lo que no conviene asumir normalidad global para todo el experimento.
