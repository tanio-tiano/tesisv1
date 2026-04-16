# Tabla Interpretativa por Gap al Optimo

Referencia: CEC 2022, 30 agentes, 500 iteraciones, dim=10.

| Funcion | Optimo | Gap WO_BASE | Gap WO_SHAP | Reduccion de gap | Lectura corta |
| --- | ---: | ---: | ---: | ---: | --- |
| F1 | 300.000000 | 757.845723 | 757.845723 | 0.000000 | Sin intervencion; el controlador no altero el comportamiento base. |
| F2 | 400.000000 | 81.033424 | 81.033424 | 0.000000 | Empate exacto; no hubo necesidad de control. |
| F3 | 600.000000 | 2.757365 | 2.002103 | 0.755262 | Mejora pequena pero limpia; SHAP ayudo a ajustar sin desestabilizar. |
| F4 | 800.000000 | 32.833743 | 32.833580 | 0.000163 | Mejora marginal; la ganancia vino de volver el control mas conservador. |
| F5 | 900.000000 | 11.825033 | 10.779065 | 1.045968 | Mejora clara y consistente; las intervenciones fueron efectivas. |
| F6 | 1800.000000 | 221.737666 | 147.274435 | 74.463231 | Mejor caso del controlador; filtrar senales de calendario evito rescates equivocados. |
| F7 | 2000.000000 | 24.513051 | 57.258002 | -32.744951 | El control sigue perjudicando; la politica actual no se ajusta bien a esta dinamica. |
| F8 | 2200.000000 | 23.529027 | 25.055638 | -1.526611 | Leve deterioro; hubo intervencion, pero no genero ventaja real. |
| F9 | 2300.000000 | 240.099450 | 240.099450 | 0.000000 | Empate; el controlador no aporto ni dano. |
| F10 | 2400.000000 | 100.478085 | 100.478085 | 0.000000 | Empate; sin activacion relevante del mecanismo. |
| F11 | 2600.000000 | 7.866769 | 7.866769 | 0.000000 | Empate total; la corrida ya era estable sin control. |
| F12 | 2700.000000 | 164.663989 | 162.296817 | 2.367172 | Mejora defendible; SHAP guio intervenciones utiles sin sobreintervenir. |

## Resumen

- WO_SHAP reduce el gap en 5 funciones: F3, F4, F5, F6 y F12.
- WO_BASE queda mejor en 2 funciones: F7 y F8.
- Hay 5 empates: F1, F2, F9, F10 y F11.
- La mayor reduccion de gap ocurre en F6.
- La evidencia global favorece a WO_SHAP si el criterio principal es gap al optimo, aunque la mejora no es uniforme en todas las funciones.
