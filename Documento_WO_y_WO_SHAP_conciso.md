# Documento tecnico breve: WO y WO + SHAP

## 1. Walrus Optimizer base

Archivo principal:

- `WO_base_cec/wo.py`

Funcion principal:

- `run_wo(search_agents_no, max_iter, lb, ub, dim, objective)`

Flujo del algoritmo:

1. inicializa la poblacion;
2. evalua el fitness de cada agente;
3. actualiza lideres;
4. calcula parametros adaptativos;
5. mueve a los agentes;
6. guarda la convergencia.

Variables centrales del WO:

- `positions`: poblacion
- `best_score`: mejor fitness global
- `best_pos`: mejor solucion
- `second_score`: segundo mejor fitness
- `second_pos`: segunda mejor solucion
- `convergence_curve`: curva de convergencia

Parametros internos del WO:

- `alpha`
- `beta`
- `danger_signal`
- `safety_signal`

Funciones auxiliares:

- `initialization(...)`
- `hal(...)`
- `levy_flight(dim)`

## 2. Parametros clave de tu ejecucion

En los runners:

- `SELECTED_FUNCTION`
- `args.dim`
- `args.agents`
- `args.iterations`
- `args.seed`
- `args.output`

Valores frecuentes:

- `dim = 10`
- `agents = 50`
- `iterations = 500`

## 3. Extension WO + SHAP

Archivos principales:

- `WO_shap/wo_controlled.py`
- `WO_shap/online_controller.py`

Elementos principales:

- `run_wo_controlled(...)`
- `OnlineXAIController`

Esta version agrega:

1. metricas de estado;
2. deteccion de eventos;
3. modelo de machine learning;
4. explicabilidad con SHAP;
5. acciones adaptativas sobre el WO.

## 4. Metricas del controlador

Bloque actual:

- `iteration`
- `best_score`
- `mean_fitness`
- `fitness_std`
- `delta_best_window`
- `stagnation_length`
- `population_diversity`
- `diversity_ratio`
- `mean_distance_to_best`
- `success_rate_recent`

### Que hace cada una

- `iteration`: iteracion actual
- `best_score`: mejor fitness encontrado
- `mean_fitness`: promedio del fitness poblacional
- `fitness_std`: dispersion del fitness de la poblacion
- `delta_best_window`: mejora del mejor fitness en una ventana reciente
- `stagnation_length`: iteraciones consecutivas sin mejora significativa
- `population_diversity`: dispersion espacial promedio de la poblacion
- `diversity_ratio`: diversidad actual respecto de la inicial
- `mean_distance_to_best`: distancia media de los agentes al mejor agente
- `success_rate_recent`: proporcion reciente de iteraciones con mejora efectiva

## 5. Cuales son propias del WO y cuales son agregadas

### Propias o directamente derivadas del WO

- `best_score`
- `mean_fitness`
- `fitness_std`
- `population_diversity`

Estas salen directamente del estado de la poblacion y del fitness.

### Agregadas para el controlador

- `iteration`
- `delta_best_window`
- `stagnation_length`
- `diversity_ratio`
- `mean_distance_to_best`
- `success_rate_recent`

Estas no pertenecen al WO original. Fueron creadas para monitoreo, deteccion de estancamiento y control.

## 6. Variables auxiliares usadas para construir metricas

- `best_history`
- `recent_successes`
- `stagnation_length`
- `initial_diversity`
- `fitness_values`
- `positions`
- `best_pos`

## 7. Integracion de machine learning

Modelo utilizado:

- `RandomForestClassifier`

Variable de codigo:

- `self.model = RandomForestClassifier(...)`

Features de entrada:

- `delta_best_window`
- `stagnation_length`
- `population_diversity`
- `diversity_ratio`
- `mean_distance_to_best`
- `success_rate_recent`

Variables de implementacion:

- `self.training_X`
- `self.training_y`
- `current_features`
- `risk_score`

## 8. Integracion de SHAP

SHAP se usa sobre el modelo de riesgo del controlador, no sobre la funcion objetivo.

Variables y objetos de codigo:

- `self.explainer = shap.TreeExplainer(self.model)`
- `shap_values = self.explainer.shap_values(...)`
- `dominant_feature`

Pregunta que responde SHAP:

"Que metricas del estado actual estan explicando que el controlador estime alto riesgo?"

## 9. Activacion por eventos

SHAP no se ejecuta en cada iteracion. Se activa por eventos.

Eventos actuales:

- `stagnation`
- `low_diversity`
- `low_progress`
- `high_risk`
- `periodic_check`

Variables:

- `event_active`
- `event_reason`

## 10. Acciones del controlador

El controlador construye `action` con:

- `mode`
- `risk_score`
- `dominant_feature`
- `alpha_scale`
- `beta_scale`
- `danger_scale`
- `safety_shift`
- `exploration_weight`
- `partial_reset_fraction`

Modos:

- `baseline`
- `preemptive`
- `rescue`

## 11. Como modifica el WO

- `alpha_scale` modifica `alpha`
- `beta_scale` modifica `beta`
- `danger_scale` modifica `danger_signal`
- `safety_shift` modifica `safety_signal`
- `exploration_weight` agrega perturbacion adicional
- `partial_reset_fraction` reinicializa parte de los peores agentes

## 12. Conclusión

Tu proyecto tiene dos niveles:

1. `WO_base_cec`: Walrus Optimizer base para benchmarking
2. `WO_shap`: Walrus Optimizer con controlador, machine learning y SHAP

El WO base optimiza.

La version `WO_shap` agrega:

- monitoreo del estado;
- deteccion de eventos;
- modelo de riesgo;
- explicabilidad con SHAP;
- control adaptativo sobre el optimizador.
