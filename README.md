# Walrus Optimizer y WO + SHAP

Este repositorio contiene dos implementaciones sobre benchmarks `CEC 2022`:

- `WO_base_cec`: Walrus Optimizer base
- `WO_shap`: Walrus Optimizer con controlador en linea, machine learning y SHAP

Tambien incluye un adaptador separado para tu problema aplicado:

- `custom_problem`: instancias `soft`, `medium` y `hard` como perfiles de sensibilidad

## Requisitos

Dependencias principales:

- `numpy`
- `pandas`
- `matplotlib`
- `scipy`
- `scikit-learn`
- `shap`
- `opfunu`

## Ejecucion

### WO base

Editar en `WO_base_cec/run_wo_cec.py`:

```python
SELECTED_FUNCTION = "F1"
```

Ejecutar:

```powershell
cd "WO_base_cec"
python run_wo_cec.py --output test_outputs
```

### WO + SHAP

Editar en `WO_shap/run_wo_shap_cec.py`:

```python
SELECTED_FUNCTION = "F1"
```

Ejecutar:

```powershell
cd "WO_shap"
python run_wo_shap_cec.py --output test_outputs
```

## Salidas

### WO base

- CSV de convergencia
- PNG de convergencia

### WO + SHAP

- CSV de convergencia
- PNG de convergencia
- CSV de estado del controlador
- CSV de valores SHAP

## Nota

Los outputs generados y cachés quedan excluidos por `.gitignore`.
