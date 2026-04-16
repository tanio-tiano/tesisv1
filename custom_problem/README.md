# Adaptador de Sensibilidad del Problema

Este modulo define tres instancias del problema:

- `soft`
- `medium`
- `hard`

La intencion no es mezclar este problema con `CEC 2021/2022`, sino dejar un
adaptador separado para tu caso aplicado.

## Criterio

- `soft` y `medium` usan directamente las instancias que ya habias definido.
- `hard` se construye como una escalada determinista de sensibilidad sobre el
  perfil `medium`, aumentando:
  - numero de clientes
  - numero de hubs
  - distancias efectivas
  - costos fijos
  - presion de capacidad
  - restriccion de `D_max`

## Uso

```python
from custom_problem import get_problem_instance

soft = get_problem_instance("soft")
medium = get_problem_instance("medium")
hard = get_problem_instance("hard")

print(soft.to_dict())
```
