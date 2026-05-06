Ejecucion base WO sobre CEC 2022 con opfunu:

`python run_wo_base_cec2022_multirun.py --agents 30 --iterations 500 --runs 30 --output all_functions_outputs_30runs`

Benchmark:
- Usa `opfunu` mediante `../opfunu_cec_adapter.py`.
- Evalua CEC 2022 F1-F12 por defecto.
- No depende de carpetas oficiales locales de CEC.

Salidas principales:
- `values/summary_wo_base_cec2022_30x500_30runs_legacy.csv`
- `values/statistics_wo_base_cec2022_30x500_30runs_legacy.csv`
- `report_wo_base_cec2022_opfunu_30runs.html`

El modo `--gbest-mode legacy` reproduce el comportamiento literal del codigo MATLAB publico de WO; `--gbest-mode updated` usa la mejor posicion global actualizada como guia.
