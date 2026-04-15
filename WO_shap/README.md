WO + controlador interpretable basado en valores de Shapley exacto.

Ejecucion:

`python run_wo_shap_cec.py --function F8 --iterations 500 --agents 50 --output all_functions_outputs_final`

Configuracion final usada en la corrida actual:
- 50 agentes.
- 500 iteraciones.
- Dimension 10.
- Ventana de estancamiento `delta_window = 50`.
- SHAP exacto solo en intervenciones espaciadas por `shap_interval`.

Salidas:
- `conv_curve_shap_Fx.csv`
- `controller_state_Fx.csv`
- `controller_events_Fx.csv`
- `shap_values_Fx.csv`
- `summary_wo_shap_all_functions.csv`
- `comparison_wo_base_vs_wo_shap_50x500.csv`
- `interactive_monitor_all_functions.html`

Metodologia implementada:
- Nucleo WO alineado con la version base estilo MATLAB-literal.
- Deteccion de estancamiento por no mejora consecutiva.
- Diagnostico por diversidad poblacional normalizada.
- Acciones reactivas: `adjust_alpha_beta`, `partial_restart`, `random_reinjection`.
- SHAP/Shapley exacto explica el fitness usando variables internas escalares del WO.
