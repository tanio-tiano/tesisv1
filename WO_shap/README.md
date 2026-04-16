WO + controlador interpretable basado en valores de Shapley exacto.

Benchmark:
- Usa el codigo Python oficial de CEC 2022 desde `../cec2022_official`.
- No usa `opfunu` para evaluar las funciones.

Ejecucion:

`python run_wo_shap_cec.py --function F8 --iterations 500 --agents 30 --output all_functions_outputs_final`

Configuracion final usada en la corrida actual:
- 50 agentes.
- 500 iteraciones.
- Dimension 10.
- Ventana de estancamiento `delta_window = 50`.
- SHAP exacto solo en una seleccion representativa de episodios severos de estancamiento, con tope por ejecucion.

Salidas:
- `conv_curve_shap_Fx.csv`
- `controller_state_Fx.csv`
- `controller_events_Fx.csv`
- `shap_values_Fx.csv`
- `summary_wo_shap_all_functions.csv`
- `comparison_wo_base_vs_wo_shap_50x500.csv`
- `comparison_current_wo_paper_23_cec2022_and_wo_shap.csv`
- `interactive_monitor_all_functions.html`

Comparacion oficial vigente:
- Usar `comparison_current_wo_paper_23_cec2022_and_wo_shap.csv` cuando necesites contrastar el benchmark de 23 funciones del paper WO con CEC 2022 y WO_SHAP.

Metodologia implementada:
- Nucleo WO alineado con la version base estilo MATLAB-literal.
- Deteccion de estancamiento por no mejora consecutiva.
- Diagnostico por diversidad poblacional normalizada.
- Acciones reactivas: `adjust_alpha_beta`, `partial_restart`, `random_reinjection`.
- SHAP/Shapley exacto explica el fitness usando variables internas escalares del WO, incluyendo `diversity` y `diversity_norm`, sin usar `pop_size` como feature.
