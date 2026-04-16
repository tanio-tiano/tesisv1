Ejecucion base WO sobre CEC 2022:

`python run_wo_cec.py --function F8 --agents 50 --iterations 4000 --output all_functions_outputs_final`

Benchmark:
- Usa el codigo Python oficial de CEC 2022 desde `../cec2022_official`.
- No usa `opfunu` para evaluar las funciones.

Ejecucion base WO sobre CEC 2021:

`python run_wo_cec.py --benchmark cec2021 --function F8 --agents 30 --iterations 500 --output all_functions_outputs_final`

CEC 2021:
- No existe un Python oficial equivalente al de CEC 2022 dentro del paquete descargado.
- La integracion usa una traduccion local directa de las formulas oficiales de CEC 2021.
- Los datos provienen de `../cec2021_official/input_data`, copiados desde el paquete oficial descargado.

Benchmark del paper WO:

`python run_all_wo_cec.py --benchmark wo_paper_23 --agents 30 --iterations 500 --output all_functions_outputs_final`

- Este benchmark replica las 23 funciones definidas en `Get_Functions_details.m` del paquete MATLAB original de Walrus Optimizer.
- No se mezcla con `cec2021_official`, porque la suite oficial que integramos aparte tiene 10 funciones, no 23.
- Cada funcion usa su dimension y sus limites originales del paquete MATLAB.

Corrida de todas las funciones:

`python run_all_wo_cec.py --benchmark cec2021 --agents 30 --iterations 500 --output all_functions_outputs_final`

Salidas:
- `all_functions_outputs_final/values/conv_curve_Fx.csv`
- `all_functions_outputs_final/values/summary_wo_base_50x4000_all_functions.csv`
- `all_functions_outputs_final/graphs/interactive_monitor_base_all_functions.html`
- `all_functions_outputs_final/values/conv_curve_cec2021_Fx.csv`
- `all_functions_outputs_final/values/summary_wo_base_cec2021_30x500_all_functions.csv`
- `all_functions_outputs_final/graphs/interactive_monitor_base_cec2021_all_functions.html`
- `all_functions_outputs_final/values/conv_curve_wo_paper_23_Fx.csv`
- `all_functions_outputs_final/values/summary_wo_base_wo_paper_23_30x500_all_functions.csv`
- `all_functions_outputs_final/graphs/interactive_monitor_base_wo_paper_23_all_functions.html`
- `all_functions_outputs_final/values/comparison_current_wo_paper_23_cec2022_and_wo_shap.csv`

Comparacion oficial vigente:
- Usar `comparison_current_wo_paper_23_cec2022_and_wo_shap.csv`.
- Esa tabla reemplaza la comparacion intermedia que mezclaba el bloque `cec2021` de 10 funciones.

El panel HTML permite seleccionar las funciones disponibles del benchmark ejecutado y revisar la curva de convergencia del WO base.
