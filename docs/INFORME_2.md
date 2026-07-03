# INFORME 2 — Investigación Aplicada

**Título del proyecto:** Explicabilidad en algoritmos bio-inspirados: diseño de un controlador para mitigar estancamiento y convergencia prematura.

**Autor:** Lucciano Fabián Cáceres Costa.

**Universidad:** Universidad de Valparaíso, Escuela de Ingeniería Informática.

**Profesor guía:** Rodrigo Olivares.

**Profesor correferente:** Pablo Olivares.

**Asignatura:** Seminario de Título II.

**Fecha:** Junio 2026.

---

## Resumen ejecutivo

El presente trabajo aborda uno de los problemas transversales de las metaheurísticas estocásticas poblacionales —el estancamiento y la convergencia prematura— mediante un mecanismo poco explorado dentro del lazo de optimización: la inteligencia artificial explicable (XAI). En concreto, se diseña, implementa y evalúa un **controlador interpretable basado en valores de Shapley exactos** que actúa **en línea** sobre el Walrus Optimizer (WO) [Han 2024], detectando agentes estancados, atribuyendo su deterioro a las seis señales internas del operador (`alpha`, `beta`, `A`, `R`, `danger_signal`, `safety_signal`) y disparando una acción de rescate trazable. La detección es reactiva por agente, en evaluaciones de la función objetivo (FES): un agente se declara estancado cuando han transcurrido al menos un 10 % del MaxFES desde su última mejora personal. La explicación se realiza mediante SHAP exacto sobre las 2⁶ = 64 coaliciones posibles, con una *value function* que simula al agente durante k = 3 pasos del WO. La acción es única bifurcada según la cuota dominante (`dominant_share ≥ 0,90`): rama A (reinicio aleatorio uniforme) cuando la atribución está repartida, rama B (reposicionamiento guiado con amplificación 2.0 de la señal dominante seguida de un paso del operador WO) cuando concentra el peso.

La evaluación adopta una **ablación pareada por semilla** entre `base` (WO sin controlador) y `shap` (WO + controlador), sobre las 12 funciones de CEC2022 en dimensión 10 a cuatro presupuestos (5·10³, 5·10⁴, 5·10⁵, 5·10⁶ FES) con 30 corridas por configuración, más un caso aplicado combinatorio (TMLAP, instancia dura 24 × 8). El contraste estadístico utiliza Wilcoxon signed-rank pareado (α = 0,05), justificado por Shapiro–Wilk al rechazarse normalidad en la mayoría de las distribuciones.

**Hallazgo honesto en calidad:** `shap ≈ base`. Sobre 36 contrastes pareados con muestras completas, se observa un único `+`, treinta y cuatro `=` y un único `−`; el veredicto agregado por presupuesto es (0|12|0), (1|11|0), (0|11|1) y (0|12|0) para 5k, 50k, 500k y 5M FES respectivamente (este último no concluyente con n = 5 en shap). La hipótesis de mejora de calidad **no se confirma**. **Aporte confirmado en interpretabilidad:** sobre 3 563 explicaciones, `safety_signal` domina (cuota 0,39; 1 193 dominancias), seguida por `danger_signal` (0,27; 848) y `beta` (0,20; 547); `A` **no contribuye en ningún caso** (cuota 0,0000). Se observa un shift sistemático con el presupuesto: a 5k FES domina `danger_signal` (≈ 0,50), a partir de 50k FES domina `safety_signal`. **Aporte confirmado en viabilidad:** el costo de SHAP exacto es de ~1 900 FES por corrida (~0,4 % del presupuesto a 5·10⁵), neutralizando la crítica habitual al cálculo exacto.

**Problema central identificado:** la rama B amplifica el desvío de la señal dominante a ciegas (s' = b + 2·(s − b)), sin controlar la dirección; como SHAP marca la señal **más influyente**, no la **errónea**, el reposicionamiento a veces ayuda y a veces empeora, cancelándose en promedio.

**Próximo paso:** *refactor direccional* del controlador, usando el **signo** del SHAP para empujar la señal dominante **hacia** donde mejora el fitness, complementado con señales por-agente que aporten información posicional.

## Tabla de contenidos

- [Resumen ejecutivo](#resumen-ejecutivo)
- [1. Introducción](#1-introducción)
 - [1.1 Marco conceptual](#11-marco-conceptual)
 - [1.2 Estado del arte](#12-estado-del-arte)
 - [1.3 Definición del problema](#13-definición-del-problema)
 - [1.4 Objetivos](#14-objetivos)
- [2. Diseño de la solución](#2-diseño-de-la-solución)
 - [2.1 Modelamiento de la solución](#21-modelamiento-de-la-solución)
 - [2.2 Metodología de investigación](#22-metodología-de-investigación)
 - [2.3 Técnicas de análisis](#23-técnicas-de-análisis)
 - [2.4 Procesos: recolección, almacenamiento, análisis y visualización](#24-procesos-recolección-almacenamiento-análisis-y-visualización)
 - [2.5 Evolución respecto a la Entrega 1 (correcciones aplicadas)](#25-evolución-respecto-a-la-entrega-1-correcciones-aplicadas)
- [3. Experimentos](#3-experimentos)
 - [3.1 Variables](#31-variables)
 - [3.2 Diseño arquitectónico](#32-diseño-arquitectónico)
 - [3.3 Implementación realizada](#33-implementación-realizada)
 - [3.4 Casos de uso y datasets](#34-casos-de-uso-y-datasets)
- [4. Resultados](#4-resultados)
 - [4.1 Análisis cuantitativo](#41-análisis-cuantitativo)
 - [4.2 Visualización de resultados](#42-visualización-de-resultados)
 - [4.3 Interpretación de resultados](#43-interpretación-de-resultados)
- [5. Implantación](#5-implantación)
 - [5.1 Requerimientos mínimos y recomendados](#51-requerimientos-mínimos-y-recomendados)
 - [5.2 Preparación del ambiente](#52-preparación-del-ambiente)
 - [5.3 Evidencia de preparación del ambiente](#53-evidencia-de-preparación-del-ambiente)
 - [5.4 Documentación y manual de usuario](#54-documentación-y-manual-de-usuario)
- [6. Conclusiones](#6-conclusiones)
 - [6.1 Éxitos conseguidos](#61-éxitos-conseguidos)
 - [6.2 Avances en el área](#62-avances-en-el-área)
 - [6.3 Problemas y/o riesgos identificados](#63-problemas-yo-riesgos-identificados)
 - [6.4 Oportunidades y/o propuestas de trabajo futuro](#64-oportunidades-yo-propuestas-de-trabajo-futuro)
- [Bibliografía](#bibliografía)
- [Anexo A — Glosario de siglas](#anexo-a--glosario-de-siglas)

---

## 1. Introducción

La presente tesis aborda un fenómeno transversal a las metaheurísticas estocásticas poblacionales —el estancamiento y la convergencia prematura— desde una perspectiva que rara vez se ha aplicado dentro del lazo de optimización: la inteligencia artificial explicable (XAI). En concreto, se propone, implementa y evalúa un controlador en línea que, durante la ejecución del Walrus Optimizer (WO) [Han 2024], emplea valores de Shapley exactos como mecanismo de diagnóstico interno para detectar agentes estancados, atribuir su deterioro a las señales de control del algoritmo y disparar una acción de rescate trazable. Esta introducción ofrece el marco conceptual necesario para entender la propuesta (Sección 1.1), revisa el estado del arte y el vacío que se pretende cubrir (Sección 1.2), formula el problema de investigación (Sección 1.3) y presenta los objetivos general y específicos del trabajo (Sección 1.4).

### 1.1 Marco conceptual

Las metaheurísticas son procedimientos de optimización aproximada cuya finalidad es obtener soluciones de buena calidad para problemas computacionalmente intratables en tiempos razonables. Dentro de ese amplio paraguas, las metaheurísticas poblacionales bio-inspiradas comparten una idea común: mantener un conjunto de agentes que exploran el espacio de soluciones de manera cooperativa, balanceando exploración (cobertura amplia del espacio) y explotación (intensificación local), inspiradas en metáforas biológicas como el comportamiento colectivo de enjambres, manadas o colonias [Somvanshi 2025]. Su atractivo radica en su generalidad —pueden tratar problemas continuos, combinatorios y mixtos— y en su robustez frente a paisajes de búsqueda no convexos, ruidosos o multimodales típicos de aplicaciones reales [Crawford 2023; Kalasampath 2025].

**Walrus Optimizer (WO).** El Walrus Optimizer, propuesto por [Han 2024], es una metaheurística poblacional bio-inspirada relativamente reciente que se inspira en la dinámica social de las morsas. Su población de `N` agentes se reparte en tres roles: 45% machos, 45% hembras y 10% crías; este reparto, ajustado al paper original (§3.2.1), gobierna qué operadores se aplican a cada subgrupo. El comportamiento del enjambre se organiza en torno a cuatro regímenes de movimiento que se seleccionan en función de dos señales temporales agregadas, `danger_signal` y `safety_signal`. Los cuatro regímenes son: (i) exploración por diferencias de posiciones entre pares de individuos; (ii) reproducción mediante secuencias cuasi-aleatorias de Halton combinadas con vuelos de Lévy; (iii) huida coordinada conducida por un único líder ante peligro alto; y (iv) agrupación en torno a dos líderes en condiciones de seguridad. La activación de cada régimen y la magnitud de los desplazamientos quedan totalmente determinadas por seis señales de control: `alpha` (presupuesto restante, equivalente generalizado del `t/T` del paper original), `beta` (transición exploración-explotación), `A` (factor de escala), `R` (componente direccional acotado en `[-1, 1]`), `danger_signal` (`[-2α, 2α]`) y `safety_signal`. Estas seis variables son, en términos prácticos, las palancas internas que dictan el comportamiento del algoritmo en cada paso y serán las features sobre las que opera la explicación SHAP en este trabajo.

**Valores de Shapley y SHAP.** Los valores de Shapley provienen de la teoría de juegos cooperativos: dado un conjunto de `n` jugadores y una función de valor `v(S)` definida sobre subconjuntos `S` de jugadores, el valor de Shapley reparte de manera única, eficiente y justa el "pago" colectivo entre los jugadores, recorriendo todas las coaliciones posibles. El marco SHAP (SHapley Additive exPlanations) adapta este formalismo al ámbito del aprendizaje automático, interpretando cada feature como un jugador y la salida del modelo como el pago. En su versión exacta, SHAP requiere enumerar `2^n` coaliciones, lo cual es prohibitivo en escenarios con decenas o cientos de features pero perfectamente tratable cuando `n` es pequeño. En este trabajo, las features son las seis señales del WO descritas arriba; por tanto, el cálculo exacto requiere solo `2^6 = 64` coaliciones por explicación, lo que lo vuelve viable como componente en línea del optimizador. La función de valor `v(S)` no se define sobre un modelo entrenado, sino sobre el propio operador del WO: dada una coalición de señales fijada a un baseline neutro y el resto en sus valores actuales, se simula al agente durante `k = 3` pasos del WO y se devuelve su fitness resultante (`wo_core.agent_sim.make_value_function_for_agent`). De este modo, la explicación es a la vez una medida de interpretabilidad —qué señal "explica" el fitness del agente— y una entrada accionable para el controlador, que decide qué hacer en función de la atribución obtenida.

**Function Evaluations (FES) como moneda común.** En metaheurísticas, el recurso que realmente se gasta para resolver un problema es el número de evaluaciones de la función objetivo (Function Evaluations, FES). Cada llamada a la función de fitness puede ser computacionalmente costosa (simulaciones, evaluaciones de modelos físicos o decodificaciones de soluciones combinatorias factibles); por eso la comunidad de benchmarking, en particular las competencias CEC, ha estandarizado el criterio de parada en torno a un presupuesto máximo `MaxFES` y no en torno a iteraciones del algoritmo. Contar por iteraciones, en cambio, sería injusto: en el WO cada una de las cuatro ramas de movimiento implica un número variable de evaluaciones, de modo que dos corridas con el mismo número de iteraciones pueden haber consumido cantidades muy distintas de cómputo. FES, en cambio, es independiente del operador y comparable entre algoritmos.

En la presente tesis, FES adquiere un triple rol que conviene subrayar: (1) **criterio de parada** —la corrida termina al agotar `MaxFES`, fijado en cuatro niveles `5·10³, 5·10⁴, 5·10⁵, 5·10⁶`; (2) **moneda de comparación justa** —el costo de las explicaciones SHAP también se contabiliza dentro del presupuesto (cada explicación gasta `64 × 3 = 192` FES, y todo se imputa a buckets separados de `search`, `shap`, `intervention`, `init`), de modo que la comparación entre el WO base y el WO+SHAP es honesta a igualdad de presupuesto; y (3) **base de la detección de estancamiento** —un agente se considera estancado cuando han transcurrido al menos un 10% de `MaxFES` desde su última mejora personal. Este último uso es lo que permite que la detección sea barata (un contador) y, sobre todo, comparable entre presupuestos: la ventana de 10% se escala automáticamente con `MaxFES`, evitando ajustes manuales por configuración.

### 1.2 Estado del arte

La literatura sobre metaheurísticas poblacionales ha crecido sostenidamente en las últimas dos décadas, motivada por la dificultad intrínseca de los problemas combinatorios NP-difíciles y por la abundancia de problemas reales que carecen de estructura matemática explotable [Crawford 2023]. Algoritmos bio-inspirados como PSO, GA, DE, GWO, WOA y, más recientemente, WO [Han 2024] han demostrado eficacia para abordar funciones objetivo no diferenciables, alta dimensionalidad y restricciones complejas [Somvanshi 2025; Kalasampath 2025]. Las competencias CEC se han consolidado como banco de pruebas estandarizado y han permitido comparaciones sistemáticas entre algoritmos bajo presupuestos de FES uniformes.

**Estancamiento y convergencia prematura.** Una limitación reconocida transversalmente en esta familia de algoritmos es su tendencia a perder diversidad poblacional y quedar atrapada en óptimos locales, fenómeno generalmente denominado convergencia prematura o, de manera más precisa, estancamiento [Črepinšek 2025; Das 2022]. Siguiendo la formulación textual reciente de, el estancamiento se entiende como una situación en la que "el algoritmo no logra mejorar la mejor solución actual durante un período prolongado de cómputo o tiempo", y subraya que dicho período "puede medirse en iteraciones, tiempo o número de evaluaciones de la función". El mismo trabajo formaliza este intuición a través de la noción de `δ-estancamiento` —ausencia de mejora superior a un umbral `δ` durante una ventana dada— y define indicadores agregados como `MinSFEs` (mínimo número de FES sin mejora) o `MsMA` (multistart meta-aware), que permiten detectar y, eventualmente, reaccionar ante el fenómeno. La definición propuesta por resulta especialmente alineada con el enfoque adoptado aquí: el estancamiento se mide en evaluaciones, no en iteraciones, y se detecta por agente como ausencia de progreso durante una ventana proporcional al presupuesto disponible.

**Caja negra y explicabilidad.** Pese a su eficacia empírica, las metaheurísticas estocásticas tienen una limitación heredada de su naturaleza no determinista: el usuario observa el resultado final, pero rara vez dispone de información estructurada sobre por qué la búsqueda se degradó, qué operadores fueron determinantes o qué señales internas explican un mal desempeño en una región dada del espacio [Das 2022; Kalasampath 2025]. Esta opacidad ha sido señalada como un obstáculo para su adopción en dominios sensibles —decisión médica, planificación logística, ingeniería estructural— donde la trazabilidad es un requisito tan importante como la calidad numérica de la solución.

La inteligencia artificial explicable (XAI) ha emergido como respuesta sistemática a la opacidad de los modelos complejos. SHAP, en particular, se ha consolidado como técnica de referencia por sus propiedades teóricas (eficiencia, simetría, dummy, aditividad heredadas del valor de Shapley) y por la disponibilidad de aproximaciones eficientes (TreeSHAP, KernelSHAP, DeepSHAP). Sin embargo, su uso predominante en la literatura ha sido **post-hoc y descriptivo**: se entrena un modelo, se obtiene una predicción y, a posteriori, se explican las contribuciones de cada feature. Tanto en aprendizaje supervisado como en aplicaciones a optimización, los trabajos que utilizan SHAP suelen analizar relaciones entre parámetros de entrada y desempeño de un algoritmo después de las corridas, no durante ellas.

**Hyper-heurísticas y enfoques adaptativos.** Existe una línea de investigación —hyper-heurísticas y mecanismos auto-adaptativos— que sí busca alterar el comportamiento del algoritmo en tiempo de ejecución. Estos enfoques seleccionan o combinan operadores de bajo nivel mediante reglas, aprendizaje por refuerzo o políticas evolutivas, con el objetivo de mejorar la calidad o la robustez. No obstante, en su gran mayoría operan como sistemas opacos: ajustan parámetros sin generar una traza interpretable de **qué señal o variable interna del optimizador motivó el ajuste**. La interpretabilidad, cuando existe, suele aplicarse a posteriori al ranking de operadores, no al estado interno del algoritmo en el instante de la decisión. Esta misma laguna ha sido recientemente señalada por, que propone meta-mecanismos como MsMA para detectar estancamiento mediante FES y reiniciar la búsqueda, pero a nivel meta y sin abordar la atribución por señales internas.

**El hueco identificado.** Del cruce de las tres líneas —metaheurísticas bio-inspiradas, detección de estancamiento por FES y XAI— emerge un vacío claro y específico: nadie ha empleado explicabilidad como **mecanismo de control en línea, prescriptivo y por agente**, de una metaheurística poblacional. La explicabilidad se ha usado para analizar resultados, no para intervenir; las hyper-heurísticas adaptativas modulan el comportamiento, pero no explican; los esquemas de detección de estancamiento reinician sin diagnosticar. La presente tesis se sitúa precisamente en esa intersección: usar SHAP exacto, calculado durante la corrida y aplicado al agente estancado, para decidir una acción de rescate informada por la atribución obtenida, sin abandonar el régimen FES estándar ni los protocolos de benchmarking de la comunidad CEC.

### 1.3 Definición del problema

A partir del marco y del estado del arte revisados, el problema de investigación de esta tesis se enuncia en los siguientes términos:

> Las metaheurísticas estocásticas poblacionales sufren estancamiento y convergencia prematura, y al ser cajas negras no permiten intervenir sobre la búsqueda de forma informada por la dinámica interna del propio algoritmo.

Este problema se descompone en tres dimensiones que motivan el diseño del controlador propuesto:

1. **Estancamiento como fenómeno observable pero no diagnosticado.** El estancamiento es detectable —existen indicadores externos como la ausencia de mejora del mejor global o del mejor por agente durante una ventana de FES— pero los algoritmos clásicos no aprovechan esa señal para reconfigurar la búsqueda más allá de heurísticas globales (e.g. reinicio completo, perturbación uniforme). En particular, el WO [Han 2024] no incorpora ningún mecanismo de detección de estancamiento por agente: sus señales temporales (`alpha = 1 − FES/MaxFES`) avanzan monotónicamente, ignorando si los agentes progresan o no.
2. **Opacidad de la dinámica interna.** Aun cuando un usuario observe que un agente lleva muchas evaluaciones sin mejorar, no dispone de información estructurada sobre **qué señal o combinación de señales** del optimizador está gobernando ese comportamiento. Los seis controles del WO (`alpha`, `beta`, `A`, `R`, `danger_signal`, `safety_signal`) interactúan de manera no lineal en cada uno de los cuatro regímenes; sin una herramienta de atribución, la pregunta "¿por qué este agente está estancado?" no admite respuesta cuantitativa.
3. **Intervención no informada.** Las intervenciones clásicas para mitigar estancamiento —reinicios aleatorios, perturbaciones uniformes, restarts a partir del mejor— operan a ciegas, sin considerar el estado interno del optimizador en el momento de la intervención. Esto genera un acoplamiento débil entre diagnóstico y acción: o se interviene siempre (cara y disruptivo) o se interviene rara vez (pierde oportunidades de rescate), pero en ningún caso se ajusta la acción al "porqué" del estancamiento.

La conjunción de estas tres dimensiones —fenómeno detectable, dinámica opaca y acciones no informadas— justifica el diseño de un controlador que (i) detecte el estancamiento por agente en términos de FES, (ii) explique con valores de Shapley exactos qué señal lo gobierna y (iii) decida la acción de rescate en función de esa explicación, todo dentro del bucle del algoritmo y con un costo computacional contabilizado dentro del presupuesto.

### 1.4 Objetivos

#### Objetivo general

Desarrollar un controlador interpretable, basado en explicabilidad in-ejecución, que monitoree, analice y mejore el comportamiento del Walrus Optimizer durante su ejecución, integrando valores de Shapley exactos como mecanismo de diagnóstico y decisión sobre las señales internas de control del algoritmo.

#### Objetivos específicos

1. **Monitorear las señales internas del Walrus Optimizer y detectar estancamiento por agente.** Instrumentar las seis señales de control del WO (`alpha`, `beta`, `A`, `R`, `danger_signal`, `safety_signal`) y establecer una detección de estancamiento por agente operada en FES, sin participación de SHAP, con compuertas calibradas como fracciones de `MaxFES` (10% para la ventana de estancamiento, 5% para guard, cooldowns y presupuesto SHAP, 95% para late-stage). Esta capa proporciona la traza observacional sobre la que opera el resto del sistema.

2. **Atribuir, mediante SHAP exacto, el deterioro de la búsqueda a las variables de control del algoritmo.** Implementar un componente de explicabilidad que, sobre el agente detectado como estancado, calcule de forma exacta los valores de Shapley (`2^6 = 64` coaliciones) usando como función de valor la simulación del agente durante `k = 3` pasos del WO. La salida —contribución por señal, señal dominante y `dominant_share`— constituye la interpretabilidad in-ejecución del comportamiento del algoritmo (`shap_controller/controller.py`, `shap_controller/features.py`).

3. **Diseñar acciones de rescate informadas por la atribución obtenida.** Construir una política de control que, a partir de la explicación SHAP del agente estancado, decida una acción única bifurcada según `dominant_share`: una rama de reinicio uniforme (`reinit_random`) cuando ninguna señal concentra la atribución y una rama de reposicionamiento guiado (`reinit_guided`) cuando una señal domina (`dominant_share ≥ 0.90`), amplificando su desvío respecto del baseline mediante un factor `2.0` antes de aplicar un paso del operador del WO (`shap_controller/actions.py`). La acción se aplica siempre al agente estancado tras pasar las compuertas, y el sistema de cooldowns adaptativos retroalimenta el ritmo de intervención según el resultado previo.

4. **Integrar la metaheurística con el componente de explicabilidad y evaluar rigurosamente su impacto.** Implementar la integración WO + controlador SHAP como un único pipeline reproducible (`runners/run_ablation.py`) que permita ablación pareada `base` vs `shap` con la misma semilla por corrida. Evaluar el impacto en (i) calidad de solución sobre las 12 funciones del benchmark CEC 2022 (dim 10) en cuatro presupuestos `MaxFES ∈ {5·10³, 5·10⁴, 5·10⁵, 5·10⁶}` y (ii) un caso aplicado al Telecommunications Multilevel Assignment Problem (TMLAP) sobre la instancia dura `3.instancia_dura.txt` (24 clientes × 8 hubs). El análisis incorpora pruebas de normalidad de Shapiro–Wilk, pruebas pareadas de Wilcoxon signed-rank (`α = 0.05`, dos colas), contabilidad explícita del costo de SHAP en FES y reporte agregado por señal de las explicaciones generadas durante el barrido completo.

El cumplimiento conjunto de estos cuatro objetivos específicos materializa el aporte central de la tesis: cerrar el lazo **detectar → explicar → decidir → reposicionar** sobre una metaheurística poblacional usando explicabilidad como mecanismo prescriptivo en línea, en lugar de como herramienta descriptiva post-hoc, y evaluarlo bajo el régimen de FES y los estándares estadísticos vigentes en la comunidad de optimización [Črepinšek 2025; Han 2024; Lundberg & Lee 2017].


## 2. Diseño de la solución

Esta sección formaliza la propuesta metodológica del trabajo: un controlador interpretable, basado en explicabilidad in-ejecución (SHAP), acoplado al Walrus Optimizer (WO) con el objeto de mitigar el estancamiento durante la búsqueda. La presentación se organiza en cinco apartados. El apartado 2.1 describe el modelamiento del controlador (lazo de cuatro etapas, señales utilizadas, decisión bifurcada y configuración única en FES). El apartado 2.2 detalla la metodología de investigación (experimental, cuantitativa, comparativa por simulación con ablación pareada). El apartado 2.3 describe las técnicas de análisis estadístico (Shapiro-Wilk, Wilcoxon signed-rank pareado, descriptivos estilo Han 2024 y cuantificación SHAP). El apartado 2.4 documenta los procesos de recolección, almacenamiento, análisis y visualización. Finalmente, el apartado 2.5 enumera la evolución del diseño respecto a la Entrega 1, listando las seis correcciones aplicadas.

### 2.1 Modelamiento de la solución

#### 2.1.1 Arquitectura conceptual: lazo de cuatro etapas

El controlador propuesto se acopla al WO como un lazo cerrado de cuatro etapas que se ejecuta dentro de cada iteración del algoritmo poblacional. La arquitectura sigue el esquema mostrado en el diagrama de diseño (`Diagramas/Diagrama_diseno_WO_SHAP.png`) y el diagrama de implementación (`Diagramas/Diagrama_implementacion_WO_SHAP.png`):

1. **Detección reactiva por FES (sin SHAP).** Cada agente mantiene un contador `last_improve_fes` correspondiente al instante de su última mejora personal. El reloj de estancamiento es una resta sobre el presupuesto global: `fes_since_improve = FES_actual - last_improve_fes`. Un agente se declara estancado cuando `fes_since_improve >= 10% de MaxFES`. Esta etapa es deliberadamente barata: no invoca SHAP y se reduce a comparar enteros.
2. **Explicación SHAP por agente.** Si el agente estancado supera además las compuertas globales y por-agente (subapartado 2.1.3), se calculan los valores de Shapley exactos sobre las seis señales de control del WO. La explicación se construye sobre el agente individual (no sobre la población ni sobre el mejor global), de modo que la interpretación sea dirigida.
3. **Decisión bifurcada.** A partir de la cuota de la señal dominante (`dominant_share`) se elige entre dos ramas: si la atribución está concentrada (`dominant_share >= 0.90`), se aplica la rama B (reinit guiado); en otro caso se aplica la rama A (reinit aleatorio uniforme). Las dos ramas constituyen una **única acción bifurcada**: en ambas el agente se reposiciona; SHAP únicamente decide la rama.
4. **Reposicionamiento y registro.** El nuevo vector reemplaza la posición del agente y se reevalúa contra la función objetivo, descontándose la evaluación del MaxFES. El resultado alimenta los cooldowns adaptativos y la telemetría, pero **no** condiciona la aplicación (la intervención siempre se ejecuta, sin gate greedy de aceptación). El mejor global se preserva por separado, de modo que un reposicionamiento que empeore al agente no daña el resultado final.

Esta arquitectura encarna el aporte central de la tesis: la explicabilidad opera **en línea y de manera prescriptiva**, no como reporte post-hoc. La misma atribución de Shapley que sirve como evidencia interpretativa (objetivo específico 3) es la entrada que decide la acción del controlador (objetivo específico 4).

#### 2.1.2 Las seis señales del WO como features de SHAP

Las features que SHAP atribuye son exactamente las seis señales de control que gobiernan los regímenes de movimiento del WO original [Han 2024, §3.2.2, Eq. 4-8 y 11]. El módulo `shap_controller/features.py` fija la convención canónica:

```
alpha = 1 - t/T (Eq. 5)
beta = 1 - 1 / (1 + exp((T/2 - t)/T * 10)) (Eq. 11)
A = 2 * alpha (Eq. 6)
R = 2 * r1 - 1 con r1 ~ U(0, 1) (Eq. 7)
danger_signal = A * R (Eq. 4)
safety_signal = r2 con r2 ~ U(0, 1) (Eq. 8)
```

Conviene subrayar tres elecciones de diseño. Primero, las señales `R` y `danger_signal` se mantienen **signadas** (`R in [-1, 1]`, `danger in [-2*alpha, 2*alpha]`), coincidiendo con la implementación oficial `WO.m` y con el texto del paper; no se toman valores absolutos. Segundo, en régimen FES la coordenada temporal `t/T` se generaliza a `alpha = 1 - FES/MaxFES`, lo que vuelve comparable el comportamiento entre presupuestos heterogéneos. Tercero, los baselines neutros utilizados como punto de referencia de las coaliciones (`alpha=0.5, beta=0.5, A=1.0, R=0.0, danger=0.0, safety=0.5`) corresponden a las esperanzas teóricas de las variables aleatorias subyacentes; sólo se emplean como respaldo cuando el historial empírico está vacío, ya que durante la ejecución se utiliza la **mediana del historial** observado.

La elección de seis features —y no de un conjunto enriquecido con métricas poblacionales o de iteración— es deliberada: las seis señales son las palancas reales del WO. Atribuir el fitness del agente a estas señales responde directamente a la pregunta interpretativa de la tesis (¿qué hace el algoritmo cuando se estanca?) sin introducir variables artificiales ajenas a la dinámica original.

#### 2.1.3 SHAP exacto: 64 coaliciones y value function por agente

Con seis features, el espacio de coaliciones tiene `2^6 = 64` elementos; el cálculo exacto del valor de Shapley es tractable y se prefiere a las aproximaciones tipo KernelSHAP. El controlador (`shap_controller/controller.py`, método `explain_fitness`) enumera explícitamente las 64 máscaras de bits sobre las seis features, evalúa la **value function** sobre cada coalición y combina los términos con los pesos combinatorios del Shapley value.

La value function utilizada es `wo_core.agent_sim.make_value_function_for_agent`: dada una coalición de señales (las features presentes toman su valor actual y las ausentes toman el valor del baseline mediano), simula `k = 3` pasos del operador WO **únicamente para el agente estancado** y devuelve su mejor fitness alcanzado. El coste por explicación es por tanto `64 x 3 = 192` evaluaciones de la función objetivo, todas ellas descontadas del presupuesto global de FES en un bucket separado (`shap`). De este modo, la comparación con el WO base es justa: SHAP no "regala" FES.

Esta operacionalización tiene dos consecuencias importantes. (i) La atribución es **determinista** dada la trayectoria del agente (la única estocasticidad proviene del operador WO simulado, que recibe la misma semilla del controlador). (ii) Es **explícitamente no machine learning**: no hay un modelo entrenado que se explica; SHAP se aplica a la propia dinámica del WO tratando al algoritmo como el "modelo" cuya predicción —el fitness alcanzado— se descompone en aportes de sus seis señales.

La salida de `explain_fitness` incluye el vector `SHAP_i` por señal, la feature dominante (`max |SHAP_i|`) y la cuota dominante:

```
dominant_share = |SHAP_dom| / sum_i |SHAP_i|
```

Este último escalar es el que decide la rama de la acción.

#### 2.1.4 Decisión bifurcada por SHAP

Cuando un agente estancado pasa las compuertas, **siempre se interviene**. El controlador (`shap_controller/actions.py`, función `decide`) sólo elige la rama, comparando `dominant_share` contra el umbral `CONTRIBUTION_THRESHOLD = 0.90`:

- **Rama A — `reinit_random`** (`dominant_share < 0.90`, ninguna señal concentra la atribución). El agente se reinicializa uniformemente en el dominio de la búsqueda mediante la fórmula clásica del WO:

 ```
 X_i' = LB + r * (UB - LB), r ~ U(0, 1)^d
 ```

 La posición previa se descarta. Es el principio clásico de des-estancamiento puramente aleatorio; SHAP indica que ninguna palanca individual explica el estancamiento, por lo que no hay base para una intervención dirigida.

- **Rama B — `reinit_guided`** (`dominant_share >= 0.90`, una señal domina). Se construye un vector de señales modificado en el que únicamente la señal dominante se aleja del neutro mediante un factor de amplificación `factor = 2.0`:

 ```
 s' = b + factor * (s - b), con b = baseline neutro de la señal dominante
 ```

 El resto de las señales se conservan inalteradas. El valor `s'` se acota al rango válido de la señal (por ejemplo `[0, 1]` para `safety`, `[-2*alpha, 2*alpha]` para `danger`, ver `_SIGNAL_RANGES` en `actions.py`). A continuación se ejecuta **un único paso del operador WO desde la posición actual** (`apply_wo_movement_single`) con las señales modificadas. La nueva posición sustituye a la previa.

En ambas ramas, el vector resultante se proyecta al rectángulo `[lb, ub]` mediante `np.clip`. Este **triple clipping defensivo** (en la generación, tras el paso WO y tras cada movimiento poblacional posterior) garantiza que ninguna acción del controlador pueda violar las restricciones de caja del problema, incluso ante señales amplificadas que puedan empujar el operador WO fuera del dominio.

La aplicación es **incondicional**: el resultado (mejorado, neutral, peor) **no** condiciona si se acepta el nuevo vector, sino únicamente el cooldown adaptativo posterior. El mejor global se preserva por separado, así un reposicionamiento desafortunado no daña el resultado final ya alcanzado.

#### 2.1.5 Configuración única y compuertas en FES

La filosofía de diseño del controlador es **una sola configuración, válida para los cuatro presupuestos** (5e3, 5e4, 5e5, 5e6) y para todos los problemas (CEC2022 y TMLAP), sin re-tuning. Esto se logra expresando todos los parámetros temporales como **fracciones de MaxFES** y resolviéndolos a valores absolutos al inicio de cada corrida (`shap_controller/profiles.py`, método `ControllerProfile.resolve`). El diseño no admite perfiles intercambiables ni overrides en tiempo de ejecución; existe una única `DEFAULT_CONTROLLER`.

Las compuertas que regulan la actuación, todas en FES, son las siguientes:

| Compuerta | Valor (fracción de MaxFES) | Rol |
|---|---|---|
| `stagnation_window` | 10% | Declarar a un agente como estancado |
| `guard_window` | 5% | No intervenir durante el arranque |
| `late_fes` | 95% | No intervenir cerca del cierre |
| `action_cooldown` | 5% | FES mínimos entre intervenciones sobre el mismo agente |
| `effective_cooldown` | 5% | FES mínimos tras una intervención efectiva (global) |
| `shap_budget` | 5% | Tope de FES gastables en explicaciones SHAP |
| `weak_diversity_evidence` | `diversity_norm >= 0.75` y `fes_since_improve < 2*ventana` | Inhibir si la diversidad sigue alta |
| Cooldown adaptativo | x1.5 (neutral) / x2.5 (rechazado) | Frenar un controlador "nervioso" según el outcome previo |

Las elecciones de valores responden a una **regla estadística explícita**: la ventana de estancamiento corresponde a `alpha = 0.10` (90 % de confianza / 10 % de error), por considerarse un criterio menos sensible que el resto; el resto de compuertas corresponde a `alpha = 0.05` (95 % de confianza / 5 % de error), que protege arranque y cierre y acota el coste de SHAP a un máximo del 5 % del presupuesto. El límite tardío `late_fes = 95% de MaxFES` corresponde al nivel de confianza `1 - alpha`. El tope `max_interventions` se deriva del cociente `shap_budget // 192`, de modo que el coste total del controlador no pueda exceder por construcción el bucket asignado.

Esta configuración única auto-escalable es lo que vuelve operativa la promesa del régimen experimental: el mismo controlador, sin recalibración manual, opera coherentemente entre presupuestos heterogéneos.

### 2.2 Metodología de investigación

#### 2.2.1 Tipo y enfoque

La investigación es **cuantitativa, experimental, comparativa por simulación**, en el marco metodológico de Sampieri et al. (2014) y Londoño Palacio et al. (2014). No corresponde al marco PICO —específico de investigación clínica basada en evidencia— sino al diseño experimental clásico de la ingeniería computacional, articulado en torno a hipótesis falsables, manipulación de un factor experimental, control de variables y contraste estadístico.

La hipótesis general se desdobla en dos sub-hipótesis evaluadas honestamente:

- **H1 (calidad).** El acoplamiento del controlador SHAP al WO mejora la calidad de las soluciones finales respecto al WO base, evaluado por `final_fitness` y `gap_to_optimum`.
- **H2 (estancamiento).** El controlador SHAP reduce el estancamiento residual al cierre de la corrida (telemetría `n_stagnant_at_end`, `mean/max_fes_since_improve_at_end`).

La métrica principal es `final_fitness` (minimización en CEC2022; coste en TMLAP). La métrica secundaria es `gap_to_optimum` cuando el óptimo exacto está disponible.

#### 2.2.2 Diseño de ablación pareada

El factor experimental único es `mode in {base, shap}`, donde `base` ejecuta el WO sin controlador (referencia) y `shap` ejecuta el WO con el controlador SHAP descrito en 2.1. El diseño es **pareado por semilla**: para un mismo trío `(MaxFES, problema, run_id)`, base y shap comparten la semilla y, por tanto, la población inicial y la traza estocástica del WO hasta el primer momento en que el controlador interviene (`runners/run_ablation.py`, función `_seed_for`):

```
seed = base_seed + max_fes_idx * 100000 + problem_idx * 1000 + run_idx
```

La semilla es **independiente del modo**, lo que garantiza la condición necesaria para un test pareado: ambas configuraciones se aplican sobre la misma instancia estocástica del problema. Esto reduce drásticamente la varianza atribuible al ruido del muestreo y aísla el efecto causal del componente SHAP.

#### 2.2.3 Variables controladas

Para aislar el efecto del controlador, se mantienen constantes a lo largo de las comparaciones:

- **Tamaño de población** (`N`) y **dimensión** del problema (`d = 10` en CEC2022; instancia concreta en TMLAP).
- **Configuración del controlador** (la única `DEFAULT_CONTROLLER` documentada en 2.1.5).
- **Criterio de parada**: MaxFES global, con el coste de SHAP descontado del mismo presupuesto.
- **Hardware**: todas las corridas comparables se ejecutan sobre el mismo nodo (servidor Mac Studio, Apple Silicon, 24 núcleos, 64 GB), de modo que las medidas temporales auxiliares (`t_init_seconds`, `t_shap_seconds`) sean homogéneas. El barrido se lanza con `nohup` (`run_remote_cec.sh`, `run_remote.sh`) para sobrevivir a desconexiones SSH.

#### 2.2.4 Datasets y régimen de corridas

Se trabaja con dos familias complementarias:

- **CEC2022 d10** (`problems/cec2022.py`, vía `opfunu`). Doce funciones (F1-F12) que cubren los cuatro grupos del benchmark: básicas (F1-F5), híbridas (F6-F8) y de composición (F9-F12). Dominio `[-100, 100]^10`, óptimo conocido por función.
- **TMLAP** (Two-level Multi-facility Location-Allocation Problem) en su variante combinatoria aplicada, instancia `3.instancia_dura.txt` (24 clientes x 8 hubs). No tiene óptimo exacto y representa el caso aplicado real de la tesis.

El protocolo de barrido es:

- **MaxFES**: `{5e3, 5e4, 5e5, 5e6}`.
- **Corridas**: 30 por configuración (excepto `shap @ 5e6`, con 5 corridas, marcado explícitamente como **no concluyente**).
- **Modos**: `base` y `shap`, pareados por semilla.
- **Total**: `4 MaxFES x 12 funciones x 30 corridas x 2 modos = 2880 corridas CEC` (con la excepción documentada a 5e6), más las corridas de TMLAP-dura.

### 2.3 Técnicas de análisis

#### 2.3.1 Estadística descriptiva

Siguiendo la convención del paper original del WO [Han 2024, Tablas 4 y 5], los resultados se reportan por función como pares **Best** y **Avg ± Std**, calculados sobre las 30 corridas pareadas. La fila resumen por MaxFES adopta la notación de Han 2024 Tabla 6: `(W | T | L)` indica respectivamente cuántas funciones son **mejor** (`+`), **igual** (`=`) o **peor** (`-`) para `shap` respecto a `base`, según el criterio estadístico del subapartado 2.3.3.

Las tablas se renderizan en negrita la celda del mejor promedio por función y se acompañan de diagramas de caja para la inspección visual de la dispersión (`experiments/presentacion/boxplots_por_funcion.png`, panel 3x4 a 500k para CEC; `experiments/presentacion/boxplots_tmlap.png` para el caso aplicado).

#### 2.3.2 Verificación de normalidad: Shapiro-Wilk

Antes de elegir el test inferencial, se verifica la normalidad de las diferencias pareadas mediante el test de **Shapiro-Wilk** (`analysis/normality.py`). En la mayoría de funciones del CEC y en TMLAP las distribuciones rechazan la hipótesis nula de normalidad (`p < 0.05`), lo que justifica la elección de un test no paramétrico. F4 es la única función del CEC con ambos grupos normales; en TMLAP la rama `base` es normal (`p = 0.259`) pero la rama `shap` no (`p = 0.0037`). Los resultados completos se reportan en `experiments/presentacion/shapiro_wilcoxon_cec.csv` y `shapiro_wilcoxon_tmlap.csv` con sus visualizaciones correspondientes.

#### 2.3.3 Contraste inferencial: Wilcoxon signed-rank pareado

El test principal es el **Wilcoxon signed-rank pareado, bilateral, alpha = 0.05**. Es importante subrayar que **no se utiliza Mann-Whitney**: el diseño es pareado por semilla, lo que invalida el supuesto de independencia entre muestras requerido por Mann-Whitney. Cada función contribuye con un vector de diferencias `d_r = fitness_shap(run_r) - fitness_base(run_r)`, sobre el cual se aplica Wilcoxon. La etiqueta por función es:

- `+` (`W`) si la mediana de `d_r` es significativamente negativa (`shap` mejor).
- `-` (`L`) si la mediana es significativamente positiva (`shap` peor).
- `=` (`T`) si `p >= 0.05` (sin evidencia de diferencia).

El veredicto global por MaxFES es la tupla `(W | T | L)` sumando sobre las 12 funciones. Se reserva el test de **Friedman** para el escenario de comparación con `>= 3` algoritmos competidores (no aplica con sólo `base` vs `shap`). Por la misma razón no se aplica corrección por comparaciones múltiples en la presente entrega, aunque se documenta como límite estadístico en las conclusiones.

#### 2.3.4 Cuantificación de la interpretabilidad

La salida del controlador `shap_values.csv` se agrega por experimento para construir tres lecturas: (i) la **magnitud relativa** de cada señal, definida como `|SHAP_i| / sum_j |SHAP_j|` por explicación y promediada sobre la muestra; (ii) la **frecuencia de dominancia** de cada señal (cuántas veces aparece como `dominant_feature`); y (iii) el **shift por presupuesto**, comparando las tres lecturas a través de los cuatro MaxFES.

A nivel descriptivo se reporta además la **actividad del controlador**: número medio de intervenciones por corrida, reparto entre rama A (`reinit_random`) y rama B (`reinit_guided`), y costo en FES gastados en SHAP (`192 * n_explicaciones`, expresado como fracción del MaxFES).

### 2.4 Procesos: recolección, almacenamiento, análisis y visualización

#### 2.4.1 Recolección de datos

La instrumentación es **FES-native**: las trazas se indexan por evaluación de la función objetivo, no por iteración. Por cada corrida se registra una fila en `summary.csv` con los campos:

- Métricas de calidad: `final_fitness`, `gap_to_optimum`.
- Métricas de costo: `t_init_seconds`, `t_shap_seconds`, buckets de FES (`search`, `shap`, `intervention`, `init`).
- Telemetría de estancamiento: `stagnation_window`, `n_stagnant_at_end`, `mean_fes_since_improve_at_end`, `max_fes_since_improve_at_end`.
- Reparto de acciones (sólo modo shap): `n_reinit_random`, `n_reinit_guided`, `interventions`, `shap_explanations`.

Las curvas de convergencia se almacenan por `(función, corrida)` en `curves/conv_curve_F<n>_fes<M>_run<r>.csv` con columnas `fes, best_fitness`. En modo `shap` se añaden tres archivos de telemetría adicionales:

- `controller_events.csv`: una fila por intervención efectivamente ejecutada (acción, rama, `dominant_share`, `improved`, cooldowns activos, `shap_dominant_feature`).
- `controller_non_events.csv`: una fila por candidato bloqueado por una compuerta, con el motivo del bloqueo.
- `shap_values.csv`: una fila por explicación calculada, con `fes`, `agent_index`, las seis contribuciones SHAP y la feature dominante.

#### 2.4.2 Almacenamiento

La salida sigue una estructura plana y predecible que facilita el análisis posterior y el reportado HTML:

```
experiments/<config>/
├── base/
│ ├── values/{summary.csv, statistics.csv}
│ └── curves/conv_curve_*.csv
└── shap/
 ├── values/{summary.csv, statistics.csv,
 │ controller_events.csv, controller_non_events.csv,
 │ shap_values.csv}
 └── curves/conv_curve_*.csv
```

La organización separa modos en directorios hermanos para que el análisis pareado opere directamente con `pd.merge` sobre `(problem, run_id)`.

#### 2.4.3 Análisis (pirámide de agregación)

El módulo `analysis/presentation_summary.py` implementa una pirámide de seis niveles que va de lo más global a lo más fino:

1. **Veredicto Wilcoxon** por MaxFES: tupla `(W | T | L)` sobre las 12 funciones (`experiments/presentacion/veredicto.md`, `veredicto_tmlap.md`).
2. **Tablas por función**: `tabla_resumen_fes500000.csv`, `tabla_tmlap.csv` (Best y Avg ± Std, negrita del mejor).
3. **Boxplots**: distribución de `final_fitness` por función y modo, panel 3x4 para CEC a 500k, panel separado para TMLAP.
4. **Efecto MaxFES**: cómo cambia `(W | T | L)` a través de los cuatro presupuestos.
5. **Actividad del controlador**: intervenciones/corrida, reparto de ramas (`actividad_controlador.csv`, `acciones_rescate.png`).
6. **Features SHAP por experimento**: magnitud relativa y dominancia de las seis señales (`features_shap.csv`, `features_shap.png`, `features_shap_comparativa_cec.png`).

El módulo `analysis/normality.py` produce las tablas `shapiro_wilcoxon_cec.csv` y `shapiro_wilcoxon_tmlap.csv` (Shapiro + Anderson + D'Agostino opcional + Wilcoxon en un único archivo). El módulo `analysis/report_html.py` integra todo en un reporte HTML navegable.

#### 2.4.4 Visualización

El conjunto de figuras de la entrega está alojado en `experiments/presentacion/`:

- **Boxplots por función**: `boxplots_por_funcion.png` (CEC, panel 3x4 balanceado a 500k) y `boxplots_tmlap.png`.
- **Tablas resumen**: `tabla_resumen_fes500000.csv` y `tabla_tmlap.csv`, con render PNG correspondiente; el mejor promedio por función se destaca en negrita.
- **Actividad del controlador**: `acciones_rescate.png` (intervenciones y reparto rama A / rama B) y `features_shap.png` (cuota media de las seis señales).
- **Efecto del presupuesto**: `features_shap_comparativa_cec.png` (muestra el shift `danger_signal -> safety_signal` al crecer MaxFES) y `efecto_maxfes.png` (veredicto Wilcoxon por presupuesto).
- **Convergencia**: `convergencia/convergencia_cec_panel.png` (panel 4x3 con `best vs fes` para las 12 funciones, base + shap con banda `±std`) y `convergencia/convergencia_tmlap_dura_30_fes500k.png`.
- **Verificación de supuestos**: tablas y PNG de `shapiro_wilcoxon_cec.csv` y `shapiro_wilcoxon_tmlap.csv`.

Las figuras se referencian por su ruta relativa al proyecto y se integran en el informe sin embeberlas, de modo que cualquier actualización de los CSV regenera consistentemente las visualizaciones.

### 2.5 Evolución respecto a la Entrega 1 (correcciones aplicadas)

Entre abril y junio de 2026, el diseño documentado en la Entrega 1 fue revisado en conjunto con el profesor guía y reemplazado por la versión vigente descrita en los apartados anteriores. Por exigencia del rubric ("correcciones anteriores"), se enumeran a continuación los seis cambios sustantivos, indicando el estado original, el estado actual y la razón del cambio.

1. **Features de SHAP: de 7 a 6.** En la Entrega 1 se manejaba un set de 7 features (`alpha, beta, danger, safety, diversity, diversity_norm, iteration`). En el método vigente se conservan únicamente las **seis señales del operador WO** (`alpha, beta, A, R, danger_signal, safety_signal`); se descartaron `diversity`, `diversity_norm` e `iteration` por ser métricas derivadas o de tiempo, ajenas a las palancas reales del algoritmo. La decisión hace que SHAP atribuya el fitness a las variables que efectivamente lo gobiernan.

2. **Acciones: de 2 a 1 bifurcada.** La Entrega 1 proponía dos acciones distintas (partial restart elite + random reinjection). El método vigente las reemplaza por **una sola acción —reinicializar el agente estancado— bifurcada según SHAP** en rama A (`reinit_random`) y rama B (`reinit_guided`). Decisión tomada con el guía en la reunión del 19 de mayo de 2026, en favor de la trazabilidad y la simplicidad del lazo prescriptivo.

3. **Perfiles: de tres (soft / medium / hard) a configuración única.** La Entrega 1 contemplaba perfiles intercambiables con valores fijos. El método vigente adopta una **única configuración auto-escalable**, con todos los parámetros temporales expresados como **fracciones de MaxFES**. Esto elimina el re-tuning por presupuesto y por problema, y vuelve operativo el régimen experimental con cuatro MaxFES heterogéneos.

4. **Criterio de parada: de iteraciones a MaxFES.** La Entrega 1 usaba 500 iteraciones como criterio de parada. El método vigente adopta **MaxFES** como criterio de parada estándar (CEC), con un protocolo de cuatro presupuestos (5e3, 5e4, 5e5, 5e6). FES es la moneda común que permite comparar entre presupuestos y problemas, y descontar honestamente el coste de SHAP del mismo presupuesto.

5. **Roles del WO: de 40/40/20 a 45/45/10.** La Entrega 1 heredaba el `P = 0.40` del código MATLAB oficial (`WO.m`), inconsistente con el texto del paper. El método vigente sigue el **paper [Han 2024, §3.2.1]**: 90 % adultos con ratio macho:hembra 1:1 + 10 % crías, es decir 45/45/10. La desviación está documentada explícitamente en `wo_core/walrus.py`.

6. **Evaluación post-intervención: de ventana 10-iter improved/neutral/worsened a aplicación incondicional con cooldowns adaptativos.** La Entrega 1 contemplaba un gate greedy de aceptación basado en una ventana de 10 iteraciones que clasificaba el resultado como improved/neutral/worsened. El método vigente **aplica siempre la intervención** y deja que los **cooldowns adaptativos** (x1.5 si neutral, x2.5 si rechazada) regulen la frecuencia de actuación. El mejor global se preserva por separado, así que una intervención que no mejore no daña el resultado. La simplificación elimina una capa de heurística y vuelve la actuación trazable.

En conjunto, los seis cambios reflejan una madurez del diseño: simplificación del espacio de acciones, alineación estricta con el paper original del WO, generalización del criterio temporal a FES y centralización de toda la regulación en compuertas con justificación estadística explícita. El método vigente que aquí se documenta es el que efectivamente ejecuta el código actual (`shap_controller/`, `runners/run_ablation.py`) y al que se refieren los resultados reportados en la sección 3.


## 3. Experimentos

Este capítulo describe el diseño experimental que sustenta la evaluación empírica del Walrus Optimizer (WO) y su variante con controlador SHAP (WO+SHAP). El protocolo se construye sobre tres compromisos metodológicos: (i) el criterio de parada y la moneda de comparación son las evaluaciones de la función objetivo (FES), no las iteraciones, lo que homogeneiza el costo entre regímenes y entre problemas [Han 2024]; (ii) la comparación entre la variante de referencia (`base`) y la propuesta (`shap`) es una ablación pareada por semilla, de manera que ambas inician desde la misma población y consumen la misma secuencia aleatoria por cada (problema, presupuesto, corrida), habilitando un test de Wilcoxon signed-rank pareado válido sin re-mapeo posterior; y (iii) el costo del módulo de explicabilidad se contabiliza dentro del mismo presupuesto MaxFES (en un bucket separado), evitando que el modo `shap` "regale" evaluaciones respecto del modo `base`.

La sección se organiza en cuatro subsecciones: variables (3.1), diseño arquitectónico del experimento (3.2), implementación realizada (3.3) y descripción de casos de uso y datasets (3.4). El runner único `runners/run_ablation.py` despliega el factorial completo en una invocación; los runners legados `run_wo_base.py`, `run_wo_shap.py` y `compare_base_vs_shap.py` fueron eliminados al unificar la lógica para garantizar el pareado por semilla.

### 3.1 Variables

Siguiendo el estándar de diseño experimental para metaheurísticas, las variables se clasifican en independientes (factores manipulados), dependientes (respuestas medidas) y controladas (constantes para todas las celdas del experimento).

**Variables independientes (factores).** Definen el factorial completo:

| Factor | Niveles | Notas |
|---|---|---|
| Modo | {`base`, `shap`} | `base` = WO sin controlador (referencia); `shap` = WO + controlador SHAP por agente. |
| Problema | 12 funciones CEC 2022 (F1–F12) + instancia dura TMLAP | F1–F5 básicas (unimodal/multimodal), F6–F8 híbridas, F9–F12 composición; TMLAP dura = 24 clientes × 8 hubs. |
| MaxFES (presupuesto) | {5·10³, 5·10⁴, 5·10⁵, 5·10⁶} | Cuatro niveles cubren desde "vista temprana" hasta "régimen de cierre". |
| Corrida (run) | 1..30 | 30 corridas independientes por celda (5 en `shap`@5·10⁶ por costo computacional; documentado como limitación). |

**Variables dependientes (respuestas).** Capturan calidad de solución, comportamiento del controlador, costo computacional e interpretabilidad. Todas se registran a nivel de corrida (una fila en `summary.csv`) o de evento (filas en `controller_events.csv`, `controller_non_events.csv`, `shap_values.csv`):

- **Calidad de la solución:**
 - `final_fitness`: mejor fitness alcanzado al agotar MaxFES.
 - `initial_fitness`: mejor fitness tras la primera evaluación poblacional.
 - `optimum`: óptimo conocido (CEC: 300, 400, 600, 800, 900, 1800, 2000, 2200, 2300, 2400, 2600, 2700 para F1..F12; TMLAP dura: NaN, sin óptimo exacto).
 - `gap_to_optimum`: `final_fitness − optimum` (NaN cuando el óptimo es desconocido).
- **Actividad del controlador (modo `shap`):**
 - `interventions`: número total de intervenciones aceptadas en la corrida.
 - `shap_explanations`: número de explicaciones SHAP efectivamente computadas.
 - `n_reinit_random`, `n_reinit_guided`: desglose de la acción única bifurcada entre Rama A (reinicio uniforme) y Rama B (paso WO con señal dominante amplificada).
- **Telemetría de estancamiento (ambos modos, simétrica para comparabilidad):**
 - `stagnation_window`: tamaño absoluto de la ventana, calculado como `0.10 × MaxFES`.
 - `n_stagnant_at_end`: número de agentes con `fes_since_improve ≥ stagnation_window` al cierre.
 - `mean_fes_since_improve_at_end`, `max_fes_since_improve_at_end`: estadísticos de la distribución de FES desde la última mejora por agente.
- **Costo computacional:**
 - `elapsed_seconds`: wall-clock total de la corrida.
 - `t_init_seconds`: wall-clock de la inicialización (relevante en TMLAP con `local_search`).
 - `t_shap_seconds`: wall-clock acumulado de la maquinaria SHAP (value function + `explain_fitness`); no incluye `decide` ni la evaluación del candidato (esos viven en `elapsed_seconds`).
 - Buckets de FES: `fes_search`, `fes_shap`, `fes_intervention`, `fes_init`, `fes_total` ≤ MaxFES.
- **Interpretabilidad (modo `shap`, `shap_values.csv`):**
 - Atribuciones SHAP `phi_alpha`, `phi_beta`, `phi_A`, `phi_R`, `phi_danger_signal`, `phi_safety_signal` por explicación.
 - `dominant_feature`: señal con `max |SHAP_i|`.
 - `dominant_share`: cuota dominante `|SHAP_dom| / Σ|SHAP_i|`.
- **Trazabilidad (modo `shap`):**
 - `controller_events.csv`: por intervención (`fes`, `agent_index`, `action`, `dominant_feature`, `dominant_share`, `improved`, motivo de aceptación, cooldowns aplicados).
 - `controller_non_events.csv`: candidatos bloqueados con motivo (`max_interventions`, `shap_budget_exhausted`, `guard_window`, `late_fraction`, `adaptive_outcome_cooldown`, `effective_cooldown`, `weak_diversity_evidence`).

**Variables controladas (constantes).** Se mantienen fijas para aislar el efecto del factor `Modo`:

- Tamaño de población `n_agents = 30` agentes (roles 45% machos / 45% hembras / 10% crías, alineado con [Han 2024]).
- Dimensionalidad `dim = 10` en CEC; `dim = n_clients = 24` en TMLAP dura.
- Hardware: un único nodo (un servidor Mac Studio Apple Silicon para el barrido completo); ninguna corrida se mezcla entre máquinas.
- Configuración del controlador: **única**, en `shap_controller/profiles.py`, expresada como fracciones de MaxFES y resuelta a valores absolutos en FES por presupuesto. No hay perfiles ni re-tuning por problema o MaxFES (ventana de estancamiento 10%, `guard_window` / cooldowns / `shap_budget` 5%, `late_fes` 95%, umbral de dominancia 0.90, factor de amplificación 2.0, 3 pasos Shapley).
- Versiones de bibliotecas (Python 3.11+, `numpy`, `pandas`, `scipy`, `opfunu`) y semilla raíz `--seed 1234`.

### 3.2 Diseño arquitectónico

El experimento responde a un **diseño factorial completo Modo × Problema × MaxFES × Corrida**. Para los datasets CEC, las dimensiones son 2 × 12 × 4 × 30 = 2880 corridas por modo (1440 en `shap`@5·10⁶ por costo, ver §3.4); para TMLAP dura, 2 × 1 × {50k, 500k} × 30 = 120 corridas. Todo se despliega desde el runner único.

**Pareo por semilla.** El núcleo del diseño es que, para una celda fija (`max_fes_idx`, `problem_idx`, `run_idx`), las dos ramas (base y shap) compartan exactamente la misma fuente de aleatoriedad. La fórmula de semilla, definida en `runners/run_ablation.py`, es independiente del modo:

```python
def _seed_for(args, max_fes_idx, problem_idx, run_idx):
 """Semilla del run; INDEPENDIENTE del modo -> base y shap quedan pareados."""
 return int(args.seed + max_fes_idx * 100000 + problem_idx * 1000 + run_idx)
```

Esto garantiza tres propiedades:

1. Para un mismo `(MaxFES, problema, run_id)`, `base` y `shap` arrancan de la **misma población inicial** y la misma secuencia aleatoria del WO. Cualquier diferencia en `final_fitness` se atribuye al controlador SHAP, no a una semilla más afortunada.
2. Corridas distintas de la misma función usan semillas distintas, evitando colapso de varianza.
3. Reiniciar el experimento con la misma `--seed` reproduce exactamente los mismos resultados, sujeto a la versión de Python/numpy.

Este pareado habilita el test de **Wilcoxon signed-rank pareado** sobre los pares `(final_fitness_base, final_fitness_shap)` por problema, sin necesidad de re-mapeo, agregación previa o asunciones de normalidad —justificadas por Shapiro–Wilk, que en la mayoría de funciones rechaza normalidad (`p < 0.05`) y motiva el uso del test no paramétrico.

**Esquema de capas.** La arquitectura está dividida en capas con responsabilidades disjuntas (ver `Diagramas/Diagrama_diseno_WO_SHAP.png` y `Diagramas/Diagrama_implementacion_WO_SHAP.png`):

- **Orquestador (`runners/run_ablation.py`):** parsea CLI, construye la lista de problemas (vía `_make_problems`), itera sobre `max_fes_values × problems × runs` y delega cada corrida a `run_one_base` o `run_one_shap` según el modo. Mantiene la separación de salida por modo (`<output>/<mode>/values + curves`).
- **Dos ramas paralelas en una sola invocación:**
 - Rama `base` (`run_one_base`): bucle WO puro (`evaluate_and_update_leaders` + `iteration_signals` + `apply_wo_movement`). Mantiene en paralelo la telemetría de estancamiento por agente (`pbest`, `last_improve_fes`) pero **no** ejecuta acción alguna; solo registra para comparabilidad con `shap`.
 - Rama `shap` (`run_one_shap`): bucle WO con la lógica adicional de detección → explicación → decisión → intervención. La detección es O(1) por iteración (un `if fsi < window: break` sobre el orden descendente de `fes_since`); SHAP solo se invoca cuando hay agente estancado y las compuertas globales y por-agente lo permiten.
- **Capa de problema con contrato `WOProblem` (`problems/base.py`):** todo problema expone `lb`, `ub`, `dim`, `optimum`, `evaluate(x)`, `project(x)` e `initial_population(n_agents, rng)`. Hoy hay dos adaptadores: `problems/cec2022.py` (envuelve `opfunu.cec_based.cec2022`) y `problems/tmlap.py` (define la asignación multinivel, repair, decode y backtracking exacto). El factory `problems/factory.py` parsea cadenas tipo `cec2022:F6` o `tmlap:3.instancia_dura.txt`.
- **Recurso compartido `FESBudget` (`wo_core/fes.py`):** todas las evaluaciones pasan por `counting_objective(problem.evaluate, budget, bucket)`, que descuenta una unidad del bucket nombrado y verifica `budget.exhausted`. Los buckets relevantes son `search` (evaluación normal del WO), `shap` (coaliciones de la value function), `intervention` (evaluación del candidato `reinit_random`/`reinit_guided`) e `init` (local_search en TMLAP). El presupuesto total cumple `fes_search + fes_shap + fes_intervention + fes_init ≤ MaxFES`. Esto materializa el compromiso de comparación justa: el costo de explicar es FES "real" para `shap`, no un anexo gratuito.
- **Almacenamiento estructurado por modo:** cada corrida emite una fila en `values/summary.csv` y una curva en `curves/conv_curve_<label>_fes<MaxFES>_run<N>.csv` con pares `(fes, best_fitness)`. La rama `shap` añade tres CSV de trazabilidad: `controller_events.csv` (intervenciones), `controller_non_events.csv` (candidatos bloqueados con motivo) y `shap_values.csv` (atribuciones SHAP por explicación).
- **Análisis pareado posterior (`analysis/`):** `presentation_summary.py` ejecuta la pirámide de agregación (run → función → presupuesto), `normality.py` calcula Shapiro–Wilk + Anderson–Darling + D'Agostino por grupo, y `report_html.py` produce el reporte estandarizado. Wilcoxon y los gráficos derivados (boxplots por función, paneles de convergencia, distribución de SHAP por señal) se calculan sobre los CSV emitidos por el runner.

**Compuertas globales y por-agente.** Dentro del bucle iterativo del modo `shap`, las compuertas se evalúan en dos niveles. Las **globales** (`max_interventions`, `shap_budget_exhausted`, `insufficient_budget_for_shap`, `guard_window`, `late_fraction`, `adaptive_outcome_cooldown`, `effective_cooldown`) bloquean a *todos* los agentes y cortan el escaneo de la iteración con un `break`. Las **por-agente** (cooldown individual, `weak_diversity_evidence`) permiten avanzar al siguiente candidato con `continue`. Esta separación está codificada en el set `_GLOBAL_BLOCK_REASONS` del runner.

**Diagramas.** El diagrama de diseño (`Diagramas/Diagrama_diseno_WO_SHAP.png`) sintetiza el lazo "detectar → explicar → decidir → reposicionar" y la relación entre las 6 señales del WO y el módulo SHAP. El diagrama de implementación (`Diagramas/Diagrama_implementacion_WO_SHAP.png`) refleja el grafo de módulos Python (`wo_core/`, `shap_controller/`, `problems/`, `runners/`, `analysis/`) y los artefactos de salida.

### 3.3 Implementación realizada

La implementación es **Python modular**, sin dependencias propietarias, y vive en `Codigos/W.O Python/`. El stack de bibliotecas externas es minimalista: `numpy` (álgebra y RNG), `pandas` (CSV / agregación), `scipy.stats` (Wilcoxon, Shapiro–Wilk) y `opfunu` (implementación oficial del CEC 2022).

**Paquetes principales.**

- `wo_core/`: dinámica WO única según [Han 2024]. `walrus.py` expone `apply_wo_movement` (rotaciones macho/hembra/cría), `apply_wo_movement_single` (un agente, usado por Rama B), `iteration_signals` (las 6 señales con `alpha = 1 − FES/MaxFES`) y `walrus_role_counts` (45/45/10). `agent_sim.py` implementa `make_value_function_for_agent`, que envuelve un closure que simula al agente `i` durante `k = 3` pasos y devuelve su mejor fitness; esta función es la entrada a SHAP. `fes.py` define `FESBudget` y `counting_objective`. `diversity.py` calcula la diversidad poblacional (norma media de desviaciones), usada en la compuerta `weak_diversity_evidence`.
- `shap_controller/`: módulo del controlador. `controller.py` aloja `SHAPFitnessController` con métodos `record_state` (acumula historial para baselines de coaliciones), `should_consider_intervention` (compuertas), `explain_fitness` (SHAP exacto sobre 64 coaliciones), `decide` (bifurcación A/B según `dominant_share ≥ 0.90`) y `register_decision` (telemetría). `actions.py` implementa `dispatch_rescue_single` con las dos ramas: `reinit_random` (`X = LB + r·(UB−LB)`) y `reinit_guided` (amplificación `s' = b + 2·(s − b)` de la señal dominante + un paso del operador WO desde la posición actual). `features.py` define `FEATURE_COLUMNS = ("alpha", "beta", "A", "R", "danger_signal", "safety_signal")` y los baselines neutros. `profiles.py` aloja la configuración única (fracciones de MaxFES, umbral de dominancia, factor de amplificación, pasos Shapley).
- `problems/`: adaptadores que cumplen el contrato `WOProblem`. Detalle en §3.4.
- `runners/run_ablation.py`: **RUNNER ÚNICO**. Los tres runners legados (`run_wo_base.py`, `run_wo_shap.py`, `compare_base_vs_shap.py`) fueron eliminados al unificar la lógica de pareado por semilla. La unificación corrige un error histórico: las dos primeras versiones tenían semillas dependientes del modo, lo que invalidaba el test pareado.
- `analysis/`: post-procesamiento (resúmenes, normalidad, reporte HTML, generación de diagramas, conversor MD → PDF).

**Detalles clave de implementación.**

- **Triple clipping defensivo al dominio factible.** Las funciones de composición de CEC (F9–F12) son particularmente sensibles a posiciones fuera de `[lb, ub]` (devuelven `inf` o `NaN`); para garantizar evaluaciones válidas, el código clipea la posición en tres puntos independientes:
 1. **Antes de evaluar la población** (`evaluate_and_update_leaders` en `wo_core/walrus.py`): `positions = np.clip(positions, lb, ub)`.
 2. **Dentro de cada rama de movimiento** (`apply_wo_movement` y `apply_wo_movement_single`): cada vector candidato se clipea antes de ser devuelto.
 3. **En el adaptador del problema** (`CECProblem.evaluate` proyecta `x` vía `self.project(x)`, equivalente a `np.clip(x, lb, ub)`).

 Esta redundancia es deliberada: cualquier defecto en una capa queda cubierto por la siguiente, y los `inf` espurios provenientes de aritmética numérica en F9–F12 quedan eliminados sin afectar la dinámica del WO.

- **Contabilidad de FES en buckets.** Cada evaluación se descuenta del bucket correspondiente vía `counting_objective`. Esto permite verificar a posteriori que (a) `fes_total ≤ MaxFES`, (b) el costo de SHAP (`fes_shap`) ronda ~0.4% del presupuesto a 5·10⁵ (~1900 FES sobre 500 000) y (c) la inicialización TMLAP con `local_search` consume ~30–60 FES adicionales a `init` (despreciable). Para el modo `base`, todos los FES caen en `search` (excepto `init` en TMLAP).

- **Detección sin SHAP, O(1).** El bucle del runner mantiene dos vectores: `pbest[i]` (mejor personal) y `last_improve_fes[i]` (FES de la última mejora). La detección de estancamiento es una resta y un `if`: `fes_since_improve = budget.total - last_improve_fes`. No requiere modelo entrenado ni cálculo SHAP. El costo es despreciable comparado con la evaluación del objetivo.

- **SHAP exacto solo cuando hay agente estancado Y pasan las compuertas.** El orden descendente de `fes_since` (`order = np.argsort(-fes_since)`) recorre primero al más estancado; el `break` sobre `fsi < window` corta el bucle apenas se encuentra un agente que no está estancado, evitando trabajo inútil. Si una compuerta global se dispara, el escaneo se detiene; si es por-agente, se pasa al siguiente. Los candidatos bloqueados se registran en `controller_non_events.csv` con su motivo, lo que permite auditar a posteriori por qué el controlador "no actuó" en ciertas iteraciones (información que el reporte HTML usa para diagnosticar problemas de configuración).

- **Mejor global preservado.** Aunque la intervención puede empeorar la posición del agente reposicionado, el código mantiene `best_score`/`best_pos` por separado (líder global) y un `second_score`/`second_pos` (segundo líder, requerido por el WO). Una intervención que empeora *no daña* el resultado reportado: solo redirige al agente estancado a una nueva región, y los cooldowns adaptativos frenan al controlador si la intervención fue neutra (×1.5) o rechazada (×2.5).

- **Telemetría simétrica entre modos.** Tanto `run_one_base` como `run_one_shap` mantienen la ventana de estancamiento al 10% de MaxFES y reportan `n_stagnant_at_end`, `mean_fes_since_improve_at_end` y `max_fes_since_improve_at_end`. Esto permite comparar el estado de estancamiento al cierre entre ambas variantes sobre la misma base.

**Comandos de ejemplo.** El barrido completo CEC2022 se lanza con una sola invocación (que despliega Modo × Función × MaxFES × Corrida en bucle):

```bash
python -m runners.run_ablation \
 --problem cec2022:all \
 --modes base,shap \
 --max-fes 5000,50000,500000,5000000 \
 --runs 30 --agents 30 --dim 10 --seed 1234 \
 --output experiments/cec2022_d10
```

El caso aplicado TMLAP con la instancia dura, sin óptimo exacto y con inicialización aleatoria:

```bash
python -m runners.run_ablation \
 --problem tmlap:3.instancia_dura.txt \
 --modes base,shap \
 --max-fes 500000 \
 --runs 30 --agents 30 --seed 1234 \
 --no-exact-optimum --init-mode random \
 --output experiments/dura_30_fes500k
```

El controlador SHAP **no** se parametriza por flags del CLI; su configuración vive en `shap_controller/profiles.py` y es única para todos los presupuestos (compromiso explícito con la idea de "una sola receta, auto-escalable por MaxFES"). Cualquier cambio en umbrales o factores requiere editar el código fuente, lo que evita la deriva experimental.

### 3.4 Casos de uso y datasets

La evaluación cubre dos universos complementarios: un benchmark continuo estandarizado (CEC 2022) y un caso aplicado combinatorio (TMLAP). La intención es triangular el comportamiento del controlador entre un escenario sintético controlado y un problema real con restricciones, dimensión y geometría diferentes [Crawford 2023].

**CEC 2022 (benchmark continuo).** Conjunto de 12 funciones de prueba de la *IEEE CEC 2022 Competition on Single Objective Bound Constrained Optimization* (`problems/cec2022.py`, envuelve `opfunu.cec_based.cec2022`):

- **Características generales.** Dimensión `dim = 10`; dominio `[-100, 100]^10`; óptimos conocidos: F1=300, F2=400, F3=600, F4=800, F5=900, F6=1800, F7=2000, F8=2200, F9=2300, F10=2400, F11=2600, F12=2700. La función `gap_to_optimum` se computa directamente como `final_fitness − optimum`.
- **Categorización (Han 2024, sección de validación):**
 - Básicas (F1–F5): 2 unimodales (F1, F2) y 3 multimodales clásicas (F3, F4, F5). Sirven para verificar comportamiento básico (convergencia local, escape de mínimos locales).
 - Híbridas (F6–F8): combinaciones de funciones básicas con pesos por bloque. Tensión exploración-explotación.
 - Composición (F9–F12): múltiples óptimos locales con cuencas de atracción de distinta amplitud. Las más sensibles al clipping (justifican el triple clipping de §3.3) y donde típicamente se observan los efectos más fuertes del controlador.
- **Adaptador.** `CECProblem` expone `lb`, `ub`, `dim`, `optimum`, `evaluate(x)` (que internamente proyecta vía `project(x) = np.clip(x, lb, ub)` antes de delegar en `opfunu`), `initial_population(n_agents, rng)` (muestreo uniforme en `[lb, ub]` vía `uniform_population`), `name` y `family ∈ {"basic", "hybrid", "composition"}`.
- **Invocación.** `--problem cec2022:F6` para una sola función o `--problem cec2022:all` para el barrido completo F1–F12; el factory `parse_problem_spec` se encarga de construir los objetos.
- **Cobertura experimental.** 12 funciones × 4 MaxFES × 30 corridas × 2 modos = 2880 corridas por modo (con la salvedad de `shap`@5·10⁶ con 5 corridas; ver más abajo). El óptimo se completa automáticamente; `gap_to_optimum` es siempre finito.

**TMLAP — instancia dura (caso aplicado, combinatorio).** El *Two-stage Multi-Level Allocation Problem* (TMLAP, `problems/tmlap.py`) asigna cada cliente a exactamente un hub minimizando la suma de distancias cliente-hub más los costos fijos de los hubs abiertos, sujeto a restricciones de capacidad por hub y distancia máxima `D_max`. Es un problema combinatorio (decoder `np.rint`) con un repair determinista (`repair` reordena clientes por número de hubs factibles y los asigna minimizando un score que penaliza apertura de hubs, exceso de capacidad y violaciones de `D_max`).

- **Instancia.** `3.instancia_dura.txt` define 24 clientes × 8 hubs. Es la única instancia del set TMLAP que **ejercita realmente al WO**:
 - `1.instancia_simple.txt` (6 clientes × 3 hubs): trivial, el WO resuelve en pocas iteraciones; no discrimina entre `base` y `shap`.
 - `2.instancia_mediana.txt` (12 clientes × 5 hubs): aún trivial.
 - Instancia grande (1000 clientes × 500 hubs, no incluida en el experimento): **inviable** computacionalmente — una estimación conservadora arroja ~25 días para completar 51 corridas a 50 000 FES sobre un único nodo, con la mayor parte del tiempo dominado por `local_search` y `repair`.
- **Sin óptimo exacto.** El backtracking exacto (`solve_exact_by_backtracking`, con poda `min_future` + capacidad + `D_max`) está limitado a 24 clientes × 8 hubs por costo exponencial. En la instancia dura el backtracking funciona pero **no se invoca**: se ejecuta con `--no-exact-optimum`, lo que fuerza `gap_to_optimum = NaN`. La comparación se hace sobre `final_fitness` directo, no sobre el gap.
- **Inicialización.** `--init-mode random` salta la pasada de `local_search` (que costaría una cantidad significativa de FES en cada corrida); cada agente arranca de una asignación factible aleatoria reparada. Esto preserva el presupuesto MaxFES casi íntegro para el WO (`fes_init` ≈ 0).
- **Triple clipping no aplica directamente.** TMLAP es combinatorio: el operador continuo del WO produce vectores en `[0, n_hubs − 1]^{n_clients}`; el `decode` redondea y el `repair` corrige factibilidad. La equivalencia con el clipping es el repair: garantiza que toda evaluación sea válida.
- **Invocación.**

 ```bash
 python -m runners.run_ablation \
 --problem tmlap:3.instancia_dura.txt \
 --modes base,shap \
 --max-fes 500000 \
 --runs 30 --agents 30 --seed 1234 \
 --no-exact-optimum --init-mode random \
 --output experiments/dura_30_fes500k
 ```

- **Cobertura experimental.** Dos celdas: `dura_30_fes500k` (30 corridas, MaxFES = 5·10⁵) y `dura_30_fes50k` (30 corridas, MaxFES = 5·10⁴). Total: 120 corridas (60 base + 60 shap).

**Resumen del factorial efectivamente ejecutado.**

| Bloque | Modo | Funciones | MaxFES | Corridas/celda | Total corridas |
|---|---|---|---|---|---|
| CEC2022 d10 | base | F1–F12 | 5·10³, 5·10⁴, 5·10⁵, 5·10⁶ | 30 | 1440 |
| CEC2022 d10 | shap | F1–F12 | 5·10³, 5·10⁴, 5·10⁵ | 30 | 1080 |
| CEC2022 d10 | shap | F1–F12 | 5·10⁶ | 5 ⚠ | 60 |
| TMLAP dura | base + shap | dura (24×8) | 5·10⁴, 5·10⁵ | 30 | 120 |
| **Total** | | | | | **2700** corridas |

La celda `shap`@5·10⁶ se ejecuta con `n = 5` por costo computacional: un único nodo agota ~3–4 días por (problema × 30 corridas) a ese presupuesto, lo que para 12 funciones excede el horizonte de la entrega. Esta limitación está documentada explícitamente: el veredicto a 5·10⁶ se reporta como "(0|12|0) ⚠ no concluyente" en todas las tablas y figuras.

**Justificación de los presupuestos.** Los cuatro MaxFES cubren cuatro regímenes cualitativos del algoritmo:

- **5·10³**: vista temprana — el WO está aún explorando, hay alta proporción de agentes estancados al cierre y SHAP atribuye el estancamiento mayormente a `danger_signal` (cuota 0.50 según los datos observados).
- **5·10⁴**: régimen intermedio — `safety_signal` empieza a dominar.
- **5·10⁵**: régimen estándar de benchmarking CEC, comparable con la mayoría de la literatura [Han 2024]. Es el presupuesto principal de análisis (todas las tablas de resultados se centran aquí).
- **5·10⁶**: régimen de cierre — donde idealmente el algoritmo ha convergido; sirve para verificar si el controlador degrada o mantiene la calidad a largo plazo.

**Artefactos esperados por corrida.** Cada corrida produce, en `experiments/<config>/<modo>/`:

- Una fila en `values/summary.csv` con todas las variables dependientes de §3.1.
- Una curva de convergencia en `curves/conv_curve_<label>_fes<MaxFES>_run<N>.csv` con pares `(fes, best_fitness)` muestreados al final de cada iteración.
- (Solo `shap`) filas en `values/controller_events.csv` (una por intervención), `values/controller_non_events.csv` (una por candidato bloqueado) y `values/shap_values.csv` (una por explicación SHAP, con las 6 atribuciones y la dominante).

Sobre estos artefactos opera la cadena de análisis (Wilcoxon pareado, Shapiro–Wilk, boxplots, panel de convergencia, distribución SHAP por señal, tabla de actividad del controlador), cuyos resultados se reportan en el capítulo 4.


## 4. Resultados

La presente sección reporta la evidencia empírica recogida tras la ejecución del protocolo experimental descrito en la Sección 3. La organización sigue tres ejes consecutivos: en 4.1 se presenta el análisis cuantitativo (veredicto Wilcoxon, descriptivos por función, pruebas de normalidad, actividad del controlador, atribución SHAP por señal, caso aplicado TMLAP y costo computacional); en 4.2 se referencian y describen las figuras producidas; y en 4.3 se interpretan los hallazgos a la luz de la hipótesis de trabajo, distinguiendo los aportes confirmados (interpretabilidad, viabilidad operativa) del veredicto neutro en calidad, y derivando la lectura mecanística que conecta los resultados con la línea de trabajo futuro presentada en la Sección 6.

Todas las cifras citadas provienen directamente de los artefactos `experiments/presentacion/*.csv` y `experiments/presentacion/*.md` consolidados por el pipeline reproducible descrito en la metodología. Cuando una afirmación es descriptiva (sin contraste de hipótesis asociado), se señala explícitamente para evitar inferencias indebidas.

### 4.1 Análisis cuantitativo

#### 4.1.1 Veredicto de calidad por presupuesto computacional

El contraste central del estudio compara la variante con controlador SHAP (en adelante, `shap`) contra la línea base WO sin controlador (en adelante, `base`) bajo idéntica configuración, semillas pareadas y cuatro presupuestos crecientes de evaluaciones de la función objetivo (MaxFES). La métrica empleada en CEC2022 es el `final_fitness` por corrida; el contraste estadístico utiliza el test de Wilcoxon de rangos con signo apareado a dos colas con nivel de significancia α = 0.05, siguiendo la convención reportada por [Han 2024] y consistente con la guía estadística para metaheurísticas de [Crawford 2023]. La notación `(W|T|L)` indica, respectivamente, el número de funciones en las que `shap` resultó mejor, equivalente o peor que `base`.

Tabla 4.1. Veredicto Wilcoxon CEC2022 d=10 por MaxFES (12 funciones, 30 corridas pareadas, salvo nota).

| MaxFES | n funciones | corridas (base/shap) | (W \| T \| L) shap vs base |
|-----------|-------------|----------------------|----------------------------|
| 5 000 | 12 | 30 / 30 | (0 \| 12 \| 0) |
| 50 000 | 12 | 30 / 30 | (1 \| 11 \| 0) |
| 500 000 | 12 | 30 / 30 | (0 \| 11 \| 1) |
| 5 000 000 | 12 | 30 / 5 (advertencia) | (0 \| 12 \| 0) |

La lectura agregada es inequívoca: en los tres presupuestos con muestras pareadas completas (5 000, 50 000 y 500 000 FES) la hipótesis nula de igualdad entre `shap` y `base` no se rechaza en 34 de 36 contrastes individuales (94,4%). El único caso con `shap` mejor corresponde a una función a 50 000 FES, y el único caso con `shap` peor corresponde a F2 a 500 000 FES (p = 0,0154). El presupuesto extremo de 5 000 000 FES se reporta de forma descriptiva: por restricciones de tiempo de ejecución, la rama `shap` completó únicamente 5 de las 30 corridas planificadas, por lo que el pareo efectivo deja un n = 5 que hace el contraste no concluyente y empuja sistemáticamente a Wilcoxon hacia el veredicto `=`. Este punto debe leerse como una limitación operativa, no como evidencia de equivalencia.

En síntesis, el balance acumulado sobre los presupuestos con muestras completas es de **un (1) `+`, treinta y cuatro (34) `=` y un (1) `−` sobre treinta y seis (36) contrastes**, sin sesgo direccional discernible. La hipótesis de mejora de calidad del controlador SHAP frente a la línea base **no se confirma** bajo este protocolo.

#### 4.1.2 Descriptivos por función a 500 000 FES

A continuación se presenta la tabla descriptiva por función a 500 000 FES, presupuesto considerado representativo del régimen "intermedio-alto" en el que el controlador alcanza su madurez estadística (n=30 en ambas ramas). Los valores corresponden al mejor fitness por corrida (Best), el promedio ± desviación estándar sobre 30 corridas (Avg±Std) y el p-valor de Wilcoxon. El óptimo teórico de cada función F1–F12 sigue la especificación de la suite CEC2022. La columna `H` reporta el veredicto: `=` (sin diferencia significativa), `+` (`shap` mejor), `−` (`shap` peor).

Tabla 4.2. Descriptivos por función, CEC2022 d=10 a MaxFES = 500 000.

| Función | Óptimo | Best base | Best shap | Avg±Std base | Avg±Std shap | H | p (Wilcoxon) |
|---------|--------|-----------|-----------|--------------------|--------------------|---|--------------|
| F1 | 300 | 300 | 300 | 300 ± 2,6e-14 | 300 ± 4,1e-14 | = | 1,000 |
| F2 | 400 | 400,3 | 400,3 | 411,6 ± 17 | 411,6 ± 17 | − | 0,0154 |
| F3 | 600 | 600 | 600 | 600 ± 9,9e-14 | 600 ± 5,6e-14 | = | 1,000 |
| F4 | 800 | 814 | 820 | 831,8 ± 9,3 | 830,3 ± 6,9 | = | 0,626 |
| F5 | 900 | 900 | 900 | 900,1 ± 0,21 | 900,1 ± 0,20 | = | 0,657 |
| F6 | 1 800 | 2 247 | 3 011 | 1,67e+04 ± 1,4e+04 | 1,585e+04 ± 1,3e+04| = | 0,245 |
| F7 | 2 000 | 2 018 | 2 020 | 2 023 ± 4,8 | 2 022 ± 2,5 | = | 0,440 |
| F8 | 2 200 | 2 201 | 2 202 | 2 220 ± 6,8 | 2 222 ± 4,0 | = | 0,114 |
| F9 | 2 300 | 2 300 | 2 300 | 2 456 ± 1,8e+02 | 2 503 ± 1,8e+02 | = | 0,516 |
| F10 | 2 400 | 2 619 | 2 619 | 2 629 ± 26 | 2 622 ± 1,4 | = | 0,754 |
| F11 | 2 600 | 2 600 | 2 600 | 2 600 ± 3,8e-10 | 2 600 ± 2,6e-10 | = | 1,000 |
| F12 | 2 700 | 2 864 | 2 864 | 2 866 ± 1,1 | 2 866 ± 1,1 | = | 0,349 |

La lectura cuantitativa de la tabla habilita varias observaciones puntuales. Primero, las funciones F1, F3 y F11 son resueltas hasta su óptimo por ambas ramas (varianza del orden de 1e-10 a 1e-14, atribuible a aritmética de punto flotante), lo cual indica que tanto `base` como `shap` saturan estas instancias con holgura presupuestaria; el controlador no introduce regresión en problemas trivialmente convergentes. Segundo, en las funciones híbridas y compuestas (F6–F12), donde el WO original muestra dificultad sostenida frente al óptimo teórico, ambas ramas exhiben promedios estadísticamente equivalentes. Tercero, **F2** es el único caso con veredicto `−` (`shap` peor) bajo Wilcoxon con p = 0,0154; sin embargo, los promedios `411,6 ± 17` son numéricamente idénticos entre ramas a tres cifras significativas, lo que sugiere que la diferencia ranqueada es marginal en términos de magnitud práctica. Cuarto, **se observa una tendencia descriptiva a menor desviación estándar en `shap`** en varias funciones (F4: 6,9 vs 9,3; F5: 0,20 vs 0,21; F7: 2,5 vs 4,8; F10: 1,4 vs 26), tendencia que se reporta como descriptiva y **no** se evalúa con un test formal de dispersión (Levene o Brown–Forsythe), por lo que no constituye un hallazgo significativo.

Sobre la cuestión de las comparaciones múltiples: al efectuar 12 contrastes simultáneos al nivel nominal α = 0,05, la expectativa de falsos positivos bajo H0 global es de E[FP] = 12 · 0,05 = 0,6, valor cercano al observado (un único `−`). Aplicar una corrección conservadora tipo Bonferroni daría un umbral ajustado de α/12 ≈ 0,0042, bajo el cual el resultado de F2 (p = 0,0154) **no** sería significativo. Se opta por reportar el p-valor crudo y discutirlo abiertamente para preservar transparencia, en lugar de presentar correcciones post hoc que no estaban preregistradas.

#### 4.1.3 Pruebas de normalidad (Shapiro–Wilk)

Para justificar la elección del test no paramétrico de Wilcoxon, se aplicó la prueba de Shapiro–Wilk a cada distribución muestral (n=30 corridas) por función y por rama, a 500 000 FES. La tabla 4.3 resume los p-valores y el dictamen sobre normalidad conjunta (`sí` si ambas ramas pasan al nivel α = 0,05; `no` en caso contrario).

Tabla 4.3. Shapiro–Wilk por función a 500 000 FES.

| Función | SW p base | SW p shap | ¿Normal ambas? | p (Wilcoxon) | H |
|---------|-----------|-----------|----------------|--------------|---|
| F1 | <0,001 | <0,001 | no | 1,000 | = |
| F2 | <0,001 | <0,001 | no | 0,0154 | − |
| F3 | <0,001 | <0,001 | no | 1,000 | = |
| F4 | 0,569 | 0,0645 | sí | 0,626 | = |
| F5 | <0,001 | <0,001 | no | 0,657 | = |
| F6 | <0,001 | <0,001 | no | 0,245 | = |
| F7 | <0,001 | <0,001 | no | 0,440 | = |
| F8 | <0,001 | <0,001 | no | 0,114 | = |
| F9 | <0,001 | <0,001 | no | 0,516 | = |
| F10 | <0,001 | 0,0103 | no | 0,754 | = |
| F11 | <0,001 | 0,00402 | no | 1,000 | = |
| F12 | <0,001 | 0,0467 | no | 0,349 | = |

Únicamente F4 satisface la hipótesis de normalidad en ambas ramas al nivel α = 0,05 (p_base = 0,569; p_shap = 0,0645). Para las 11 funciones restantes, al menos una de las dos ramas rechaza la normalidad, mayoritariamente con p < 0,001. Este patrón es esperable en problemas multimodales y compuestos como los de la suite CEC2022 d=10, donde el `final_fitness` exhibe colas pesadas, mezclas multimodales y, en casos como F1, F3 y F11, distribuciones casi degeneradas en el óptimo. La consecuencia metodológica es directa: **el uso del test no paramétrico de Wilcoxon de rangos con signo apareado está justificado** como contraste principal de calidad, mientras que un test paramétrico tipo t pareado sería formalmente inapropiado para 11 de las 12 funciones bajo el supuesto de normalidad.

#### 4.1.4 Actividad del controlador SHAP

El controlador opera por agente y se activa únicamente cuando `fes_since_improve ≥ 10% MaxFES`, sujeto a la batería de compuertas descrita en la metodología (guardas, cooldowns adaptativos, late, evidencia débil de diversidad). A 500 000 FES, la actividad agregada del controlador sobre 12 funciones × 30 corridas se resume en la tabla 4.4.

Tabla 4.4. Actividad del controlador SHAP a 500 000 FES (12 funciones × 30 corridas).

| Métrica | Valor |
|----------------------------------------|--------------------------------|
| Intervenciones por corrida (media) | 9,90 |
| Total reinit_random (rama A) | 1 957 (54,9% de intervenciones)|
| Total reinit_guided (rama B) | 1 606 (45,1% de intervenciones)|
| Total intervenciones (suma) | 3 563 |
| Top señales dominantes (frecuencia) | safety_signal (1 193); alpha (938); danger_signal (848) |

La media de 9,90 intervenciones por corrida indica que el controlador interviene del orden de **una decena de veces** durante el ciclo completo de optimización, una intensidad consistente con un régimen de "rescate selectivo" más que de "intervención permanente". El reparto random/guided ≈ 55/45 muestra que la compuerta `dominant_share ≥ 0,90` (umbral por encima del cual la rama B amplifica la señal dominante en lugar de reinicializar al azar) **se activa con frecuencia moderadamente alta**, casi a paridad con la rama A. El número total de 3 563 intervenciones a lo largo de 360 corridas constituye también la base muestral del análisis de atribución SHAP de la subsección siguiente (cada intervención registra una explicación de las seis señales del WO).

#### 4.1.5 Atribución SHAP por señal

La contribución de cada una de las seis señales del WO (`alpha`, `beta`, `A`, `R`, `danger_signal`, `safety_signal`) se cuantifica mediante: (i) la **cuota media SHAP** (proporción del valor absoluto de la atribución sobre el total, promediada sobre todas las explicaciones); y (ii) el **número de veces dominante**, es decir, las explicaciones en las que la señal tuvo la mayor magnitud absoluta. La tabla 4.5 reporta ambos indicadores agregados sobre 3 563 explicaciones a 500 000 FES.

Tabla 4.5. Atribución SHAP agregada por señal a 500 000 FES (n = 3 563 explicaciones).

| Señal | Cuota media SHAP | Veces dominante |
|----------------|------------------|-----------------|
| safety_signal | 0,3898 | 1 193 |
| danger_signal | 0,2740 | 848 |
| beta | 0,2044 | 547 |
| alpha | 0,1041 | 938 |
| R | 0,0277 | 37 |
| **A** | **0,0000** | **0** |

Tres hallazgos estructurales emergen con claridad. Primero, **`safety_signal` es la señal dominante en magnitud y frecuencia** (cuota 0,39; 1 193 dominancias, equivalentes al 33,5% del total). Segundo, `danger_signal` ocupa el segundo lugar en cuota (0,27) y se posiciona como contraparte de `safety_signal` en el reparto del estancamiento. Tercero, **`A` no contribuye en ninguna explicación** (cuota exacta 0,0000; cero dominancias en 3 563 casos), constituyendo evidencia robusta y limpia de que esta señal del WO original es **irrelevante** para diagnosticar estancamiento en el régimen MaxFES estudiado. La señal `R` resulta también marginal (cuota 0,028; 37 dominancias, 1,0%). Conjuntamente, las dos señales asociadas a percepción de entorno (`safety_signal` + `danger_signal`) acumulan una cuota media del 66,4% y 2 041 de las 3 563 dominancias (57,3%).

Un análisis transversal por presupuesto (no replicado aquí en tabla numérica pero consolidado en `features_shap_comparativa_cec.png`, ver Sección 4.2) revela un **shift sistemático** en la señal dominante con el MaxFES: a 5 000 FES domina `danger_signal` (cuota ≈ 0,50), correspondiente al régimen de exploración temprana; a 50 000 FES y más, domina `safety_signal` (cuota ≈ 0,39), correspondiente al régimen de explotación/seguridad. Este desplazamiento se interpreta detalladamente en 4.3.

#### 4.1.6 Caso aplicado TMLAP (instancia dura)

La instancia `3.instancia_dura.txt` del problema TMLAP (Telecommunication Multi-Level Allocation Problem, 24 clientes × 8 hubs) se evaluó como caso aplicado combinatorio sin óptimo exacto conocido. El contraste se realiza sobre `final_fitness` (minimización) con n = 30 corridas pareadas por rama a MaxFES = 500 000.

Tabla 4.6. Descriptivos y contraste TMLAP `dura_30_fes500k`.

| Corrida | Best base | Best shap | Avg±Std base | Avg±Std shap | n (b/s) | SW p base | SW p shap | ¿Normal? | p (Wilcoxon) | H |
|------------------|-----------|-----------|--------------|--------------|---------|-----------|-----------|----------|--------------|---|
| dura_30_fes500k | 272 | 273 | 275,3 ± 1,8 | 275,0 ± 1,6 | 30/30 | 0,259 | 0,00368 | no | 0,58 | = |

La Shapiro–Wilk reporta normalidad en `base` (p = 0,259) pero no en `shap` (p = 0,00368), por lo que la prueba apropiada para el contraste apareado es nuevamente Wilcoxon. El veredicto es `=` con p = 0,58, en línea con lo observado en CEC2022 a presupuestos comparables. Numéricamente, `shap` presenta un promedio levemente menor (275,0 vs 275,3) y una desviación estándar también menor (1,6 vs 1,8); ambas diferencias son **descriptivas** y no constituyen evidencia significativa, pero son consistentes con la tendencia a menor varianza observada en CEC. El (W|T|L) agregado sobre la única corrida TMLAP evaluada es `(0|1|0)`.

#### 4.1.7 Costo computacional del controlador

El costo de una explicación SHAP exacta sobre las seis señales del WO requiere evaluar 2^6 = 64 coaliciones; cada coalición simula al agente k = 3 pasos, lo que arroja un costo de 192 FES por explicación. Con una media empírica de 9,90 intervenciones por corrida (sección 4.1.4), el costo total estimado por corrida es de 9,90 × 192 ≈ 1 901 FES. Sobre el presupuesto de 500 000 FES, esto representa el **0,38%** del cómputo total, valor que se redondea a ≈ 0,4% para la comunicación. Este sobrecosto es despreciable frente al presupuesto disponible y demuestra que el régimen "SHAP exacto in-ejecución" es operativamente viable bajo el protocolo MaxFES.

### 4.2 Visualización de resultados

A continuación se enumeran las figuras producidas por el pipeline de análisis, junto con la lectura específica que aporta cada una. Las figuras se referencian por su ruta relativa al proyecto y deben consultarse en el directorio `experiments/presentacion/`.

**Figura 4.1 — Tabla por función con óptimo y p-valor a 500 000 FES.**
Ruta: `experiments/presentacion/tabla_resumen_fes500000.png`.
Representación visual del contenido de la Tabla 4.2: por cada función F1–F12 se muestra el Best y el Avg±Std de ambas ramas, el óptimo de referencia, el p-valor de Wilcoxon y el dictamen H. La figura facilita la lectura rápida del veredicto agregado (un único `−` en F2; once `=`) y permite identificar las funciones saturadas (F1, F3, F11) frente a aquellas en las que ambas ramas presentan dispersión significativa (F6, F9).

**Figura 4.2 — Boxplots por función a 500 000 FES (panel 3 × 4).**
Ruta: `experiments/presentacion/boxplots_por_funcion.png`.
Panel de doce subgráficas (una por función) con boxplots pareados base vs shap (30 vs 30 corridas). La figura visualiza la mediana, los cuartiles y los outliers de cada distribución, complementando el resumen tabular con la geometría completa de los datos. Permite apreciar visualmente la tendencia a menor dispersión de `shap` en F4, F5, F7 y F10, así como la equivalencia casi exacta en F1, F3, F11 (cajas colapsadas en el óptimo) y la dispersión severa en F6 y F9 atribuible a la dificultad intrínseca de las funciones híbridas y compuestas.

**Figura 4.3 — Normalidad y Wilcoxon CEC2022.**
Ruta: `experiments/presentacion/shapiro_wilcoxon_cec.png`.
Visualización integrada de la Tabla 4.3: p-valores de Shapiro–Wilk por rama (eje izquierdo) y p-valor de Wilcoxon (eje derecho), con líneas de referencia al nivel α = 0,05. Justifica gráficamente la elección del test no paramétrico al exhibir que 11 de 12 funciones violan normalidad en al menos una rama.

**Figura 4.4 — Normalidad y Wilcoxon TMLAP.**
Ruta: `experiments/presentacion/shapiro_wilcoxon_tmlap.png`.
Análogo de la figura 4.3 para el caso aplicado TMLAP `dura_30_fes500k`. Muestra el contraste descrito en la Tabla 4.6: base normal, shap no, Wilcoxon `=` con p = 0,58.

**Figura 4.5 — Reparto random/guided de acciones de rescate.**
Ruta: `experiments/presentacion/acciones_rescate.png`.
Diagrama de barras con la frecuencia agregada de las dos ramas de la acción única bifurcada: 1 957 `reinit_random` (rama A) frente a 1 606 `reinit_guided` (rama B), sobre 3 563 intervenciones. Visualiza el reparto ≈ 55/45 reportado en la subsección 4.1.4 y permite apreciar cómo la compuerta `dominant_share ≥ 0,90` opera con frecuencia comparable a la rama de reinicialización uniforme.

**Figura 4.6 — Contribución SHAP por señal a 500 000 FES.**
Ruta: `experiments/presentacion/features_shap.png`.
Representación gráfica de la Tabla 4.5: cuota media SHAP y veces dominante por señal (`alpha`, `beta`, `A`, `R`, `danger_signal`, `safety_signal`). La figura comunica de manera inmediata los tres hallazgos estructurales: (i) `safety_signal` lidera ambos rankings; (ii) `A` es exactamente cero en cuota y en dominancia; (iii) `danger_signal` y `beta` ocupan el bloque intermedio.

**Figura 4.7 — Shift de señal dominante por presupuesto computacional.**
Ruta: `experiments/presentacion/features_shap_comparativa_cec.png`.
Tres paneles (5k, 50k, 500k FES) con la cuota media SHAP por señal en cada uno. Documenta visualmente el shift `danger → safety`: a 5 000 FES el peso reposa sobre `danger_signal` (cuota ≈ 0,50), mientras que a partir de 50 000 FES el liderazgo pasa a `safety_signal` (cuota 0,39). Esta es una de las figuras más informativas del trabajo desde el punto de vista interpretativo, porque liga el régimen presupuestario con la diagnosis del estancamiento.

**Figura 4.8 — Convergencia CEC2022 (panel 12 funciones).**
Ruta: `experiments/presentacion/convergencia/convergencia_cec_panel.png`.
Panel 3 × 4 con la curva de convergencia media de cada función a 500 000 FES, sombreada con banda ±1 desviación estándar sobre las 30 corridas, para ambas ramas. Permite visualizar la dinámica temporal del estancamiento y el efecto del controlador en el régimen tardío (FES > 50% MaxFES). En la mayoría de funciones las curvas se solapan prácticamente en todo el horizonte, consistente con el veredicto `=` agregado; en F2 se observa la única ventaja sostenida de `base` sobre `shap` que origina el veredicto `−`.

**Figura 4.9 — Convergencia TMLAP `dura_30_fes500k`.**
Ruta: `experiments/presentacion/convergencia/convergencia_tmlap_dura_30_fes500k.png`.
Curva de convergencia media con banda ±std para el caso aplicado. Las dos ramas trazan una dinámica casi indistinguible, terminando en `275,3 ± 1,8` (base) y `275,0 ± 1,6` (shap). El gráfico evidencia que el WO consume el grueso de su presupuesto en la fase intermedia y entra en estancamiento sostenido en la última cuarta parte del cómputo.

**Figura 4.10 — Boxplot TMLAP.**
Ruta: `experiments/presentacion/boxplots_tmlap.png`.
Boxplot pareado base vs shap sobre la corrida `dura_30_fes500k`. Confirma visualmente lo reportado en la Tabla 4.6: cajas solapadas, mediana muy próxima entre ramas, dispersión ligeramente menor en `shap`.

**Figura 4.11 — Tabla TMLAP con Shapiro y Wilcoxon.**
Ruta: `experiments/presentacion/tabla_tmlap.png`.
Versión figurativa de la Tabla 4.6, con resaltado del p-valor de Shapiro de cada rama (subrayando la asimetría: base normal, shap no normal) y el resultado de Wilcoxon (p = 0,58, H = `=`).

### 4.3 Interpretación de resultados

La interpretación se organiza en tres bloques: (i) qué dice y qué no dice el veredicto neutro en calidad, (ii) qué aportes sí quedan confirmados por la evidencia, y (iii) la lectura mecanística que conecta el resultado neutro con la principal línea de trabajo futuro.

#### 4.3.1 Sobre el veredicto neutro en calidad

El balance agregado del contraste de calidad puede resumirse en una sola frase: **`shap ≈ base`**. Sobre los 36 contrastes Wilcoxon con muestras pareadas completas (12 funciones × 3 presupuestos), se observa un único `+`, un único `−` y treinta y cuatro `=`. La hipótesis de trabajo —que la incorporación de un controlador SHAP exacto por agente sobre las seis señales del WO mejora la calidad final de la optimización— **no se confirma** bajo el protocolo experimental adoptado.

Tres precisiones son necesarias para evitar lecturas erróneas de este veredicto. Primero, el caso `−` en F2 a 500 000 FES (p = 0,0154) **debe discutirse con cautela**: el promedio numérico es prácticamente idéntico entre ramas (411,6 ± 17 en ambas a tres cifras significativas), de modo que la diferencia ranqueada es de magnitud trivial; adicionalmente, al ejecutar 12 contrastes simultáneos al 5%, la expectativa nula de falsos positivos es E[FP] = 0,6, y una corrección de Bonferroni (no preregistrada) anularía esta significancia (α/12 ≈ 0,0042). El veredicto `−` se reporta porque corresponde al test ejecutado, pero su peso probatorio es bajo. Segundo, el presupuesto de 5 000 000 FES no es concluyente: con `shap` n = 5 vs `base` n = 30, el pareo efectivo de Wilcoxon tiene poder estadístico bajo y tiende sistemáticamente a `=`; este punto se anota como limitación operativa y se incorpora a la sección de trabajo futuro. Tercero, la tendencia descriptiva a **menor varianza de `shap`** (TMLAP std 1,6 vs 1,8; F4 6,9 vs 9,3; F5 0,20 vs 0,21; F7 2,5 vs 4,8; F10 1,4 vs 26) es consistente con la idea de que el controlador "estabiliza" la trayectoria de búsqueda incluso cuando no mejora el centro de la distribución, pero **no** se testeó con un contraste formal de dispersión y por tanto no constituye un hallazgo significativo.

#### 4.3.2 Aportes confirmados: interpretabilidad y viabilidad

A pesar del veredicto neutro en calidad, dos aportes del trabajo quedan **firmemente confirmados** por la evidencia recogida.

El primero es de **interpretabilidad**. SHAP identifica de manera robusta y consistente cuáles señales del WO pesan en la dinámica del estancamiento: `safety_signal` lidera con cuota media 0,3898 y 1 193 de 3 563 dominancias; `danger_signal` la sigue con 0,2740 y 848 dominancias; el bloque `beta`–`alpha` queda en el rango medio; y, de manera particularmente nítida, **`A` no contribuye en ninguna explicación** (cuota exacta 0,0000; cero dominancias en 3 563 casos). Esta última afirmación es robusta porque se sostiene sobre una muestra de varios miles de explicaciones recogidas en 360 corridas y 12 funciones distintas, y constituye una guía clara para futuras simplificaciones del WO: la señal `A` es información redundante o ruidosa desde el punto de vista del diagnóstico de estancamiento en MaxFES y podría descartarse sin pérdida de capacidad explicativa.

Adicionalmente, el **shift `danger → safety` con el presupuesto** es uno de los hallazgos interpretativos más limpios del trabajo. A 5 000 FES domina `danger_signal` (cuota ≈ 0,50), lo que se interpreta como que el estancamiento temprano se explica por la palanca de exploración/percepción de peligro: los agentes que se atascan en esa fase lo hacen porque su lectura del entorno les indica retirada. A partir de 50 000 FES y hasta 500 000 FES, el liderazgo pasa a `safety_signal` (cuota 0,39), consistente con que el estancamiento tardío se explica por la palanca de explotación/seguridad: los agentes se atascan porque la lectura de seguridad los ha llevado a una región que ya no produce mejora marginal. La consistencia entre CEC2022 a presupuestos comparables y TMLAP (`safety_signal` cuota 0,41 en `dura_30_fes500k`) refuerza que este patrón no es un artefacto del benchmark continuo. El controlador, aunque no mejore el `final_fitness`, sí **observa** y **comunica** una transición funcional del régimen de búsqueda que antes era opaca.

El segundo aporte confirmado es de **viabilidad operativa**. El costo del controlador SHAP exacto es de 192 FES por explicación × 9,90 explicaciones promedio por corrida ≈ 1 901 FES, equivalente al **0,38% del presupuesto a 500 000 FES**. Esta cifra demuestra que el régimen "SHAP exacto in-ejecución por agente" es perfectamente compatible con el protocolo MaxFES y abre la puerta a controladores explicables más sofisticados sin incurrir en sobrecostos prohibitivos. La crítica frecuente a SHAP exacto como "computacionalmente inviable" se neutraliza cuando el conjunto de features es pequeño (seis señales aquí) y el value function es liviano (k = 3 pasos del operador WO).

#### 4.3.3 Lectura mecanística: la amplificación a ciegas y su consecuencia natural

El resultado neutro en calidad amerita una lectura mecanística específica que la evidencia recogida permite formular con precisión.

La acción única bifurcada del controlador opera del siguiente modo: si `dominant_share < 0,90` se ejecuta la rama A (`reinit_random`), que reinicializa al agente uniformemente en el dominio; si `dominant_share ≥ 0,90` se ejecuta la rama B (`reinit_guided`), que amplifica la señal dominante mediante la transformación `s' = b + 2·(s − b)` (donde `b` es el valor neutro de la señal) y luego ejecuta un paso del operador WO desde la posición actual con la señal amplificada. La rama A se ejecutó 1 957 veces (55%) y la rama B 1 606 veces (45%) sobre 3 563 intervenciones.

El problema central de este diseño se hace visible al combinar tres hechos: (i) SHAP marca la señal **más influyente** del estancamiento, no la señal **errónea** ni la señal cuya **corrección** mejoraría el fitness; (ii) la rama B amplifica el desvío de esa señal respecto del neutro **sin controlar la dirección** —duplica el desvío `(s − b)` en magnitud, conservando su signo, pero el signo correcto para mejorar el fitness es un dato que SHAP no provee; (iii) en consecuencia, la rama B **a veces ayuda y a veces empeora**, y en promedio sobre una muestra suficientemente grande **se cancela**.

Este diagnóstico es directamente consistente con la evidencia agregada: shap ≈ base sobre 34 de 36 contrastes; la tendencia descriptiva a menor varianza puede explicarse por la frecuencia moderada de intervenciones que "limpian" trayectorias atascadas (efecto similar al de un reinicio uniforme controlado), mientras que la ausencia de mejora en la media refleja precisamente el carácter no-direccional de la amplificación.

La consecuencia natural —que se desarrolla en detalle en la Sección 6.4 (Trabajo futuro) y se anticipa aquí como cierre interpretativo— es el **refactor direccional**: utilizar el **signo** del valor SHAP de la señal dominante para empujarla **hacia donde mejora el fitness** (contrarrestar la palanca culpable), en lugar de amplificar a ciegas el desvío observado. Bajo esta reformulación, SHAP pasaría de ser un "indicador de peso" a ser un "indicador de dirección de corrección", aprovechando completamente la información estructural que el método ya extrae. El presente trabajo aporta la evidencia empírica que motiva ese siguiente paso: la calidad no mejora porque la acción no es direccional, no porque la interpretabilidad sea insuficiente.

En síntesis, los resultados confirman que la explicabilidad in-ejecución por agente sobre las seis señales del WO es **observable, robusta y operativamente viable**, e identifican con claridad el cuello de botella del diseño actual: la traducción entre **explicación** y **acción**. Este diagnóstico cierra la Sección 4 y articula el puente hacia la discusión de la Sección 5 y las conclusiones de la Sección 6.


## 5. Implantación

La implantación del controlador interpretable basado en SHAP sobre el Walrus Optimizer (WO) se concibió desde el inicio como una pieza de software reproducible, portable y verificable. Esta sección documenta exhaustivamente cuatro aspectos exigidos por la rúbrica: (i) los requerimientos de software y hardware con sus mínimos y recomendados, (ii) el procedimiento de preparación del ambiente desde la clonación del repositorio hasta el primer experimento válido, (iii) la evidencia operativa que avala que esa preparación efectivamente se completó y produjo resultados, y (iv) la documentación de usuario y desarrollador que acompaña al artefacto y que permite a terceros reproducir el estudio sin asistencia del autor. Se siguen las indicaciones del MANUAL_USUARIO.md y del archivo operativo COMO_EJECUTAR.txt, así como la interfaz expuesta por el runner unificado `runners/run_ablation.py` verificada directamente sobre `argparse`.

### 5.1 Requerimientos mínimos y recomendados

#### 5.1.1 Plataforma de software

El intérprete de referencia del proyecto es **Python 3.11 o superior**. Esta cota inferior es estricta porque el código del controlador y del adaptador WO utiliza características introducidas en Python 3.10 (operador `match`, anotaciones de tipo modernas con `|` para uniones) y porque `numpy` y `pandas` en sus versiones recientes ya no publican binarios precompilados para versiones anteriores. La versión mayor probada en el barrido reportado (junio 2026) es Python 3.11.x; versiones 3.12 y 3.13 son compatibles, no se ha probado 3.14 al cierre de esta tesis.

Las dependencias externas están listadas en `requirements.txt` y se instalan en un único comando:

```bash
pip install -r requirements.txt
```

La lista mínima es deliberadamente pequeña, por una decisión de diseño orientada a la reproducibilidad:

- **`numpy`**: aritmética vectorial; soporta toda la dinámica del WO (movimiento, señales, muestreo).
- **`pandas`**: persistencia tabular (lectura y escritura de los CSV de telemetría y de salida).
- **`scipy`**: tests no paramétricos (Wilcoxon signed-rank, Shapiro–Wilk, Anderson–Darling) usados por el módulo de análisis.
- **`opfunu`**: implementación canónica del benchmark CEC 2022; es la **única dependencia imprescindible** para reproducir los experimentos continuos. Sin `opfunu`, la familia `cec2022:*` no funciona, mientras que `tmlap:*` seguiría operativa.

No se utiliza la biblioteca pública `shap` de Lundberg, dado que el controlador implementa **valores de Shapley exactos** sobre un dominio de seis señales (2^6 = 64 coaliciones), lo que evita el cuello de aproximación de KernelSHAP y elimina una dependencia pesada con `scikit-learn`. Tampoco se requiere PyTorch ni TensorFlow porque no hay redes neuronales ni surrogate models en línea.

Los scripts envoltorio del servidor (`run_remote_cec.sh`, `run_remote.sh`) están escritos en **bash POSIX** y requieren un shell tipo Unix para su ejecución idiomática (uso de `nohup`, redirección, `tail -f`). Bajo Windows pueden ejecutarse mediante WSL, Git Bash o equivalentes, aunque la práctica habitual en este trabajo fue desplegarlos sobre el servidor Linux y reservar la máquina Windows local para el smoke testing.

#### 5.1.2 Plataforma de hardware

La huella de memoria del runner es modesta, porque la población se mantiene en una matriz `n_agents x dim` (30 x 10 para CEC; 30 x 24 para la instancia dura del TMLAP), y la telemetría se escribe incrementalmente al disco. El cuello de botella no es la memoria sino el **tiempo de CPU**: el costo dominante es `n_runs x MaxFES` con `MaxFES = 5 x 10^6` como el escenario más exigente.

- **Configuración mínima** (corrida individual de validación): cualquier CPU x86 o ARM moderno y aproximadamente 4 GB de RAM. Suficiente para presupuestos de 5 x 10^3 a 5 x 10^4 evaluaciones, con tiempos de pared del orden de segundos a minutos.
- **Configuración recomendada** (barrido completo): CPU multinúcleo con frecuencia base ≥ 2.5 GHz y entre 8 y 16 GB de RAM. La paralelización dentro del runner es por proceso (cada `python -m runners.run_ablation` ocupa un núcleo), de modo que el servidor remoto se aprovechó lanzando varios procesos independientes sobre nodos o sesiones distintas, uno por familia (CEC) y otro por TMLAP.
- **Almacenamiento**: el barrido CEC 2022 completo (12 funciones x 4 presupuestos x 2 modos x 30 corridas) genera del orden de **2 880 archivos de curva** más los CSV agregados, totalizando una decena de MB. No es restrictivo.

#### 5.1.3 Sistema operativo

El artefacto es **independiente de plataforma** en el código Python. Se probó sobre Linux x86_64 (servidor remoto donde se ejecutaron los barridos definitivos), macOS y Windows 11 (estación de trabajo local). Los únicos elementos sensibles al sistema son:

1. Los scripts shell `*.sh` requieren bash; en Windows se invocan vía WSL o se sustituyen por su llamada subyacente al runner (la cual es 100 % Python y portable).
2. Las rutas se manejan internamente con `pathlib.Path`, sin separadores hardcoded.

### 5.2 Preparación del ambiente

Esta subsección describe el procedimiento operativo necesario para llevar una máquina genérica desde un estado limpio hasta tener un experimento válido corriendo. El detalle exhaustivo está en `COMO_EJECUTAR.txt` (siete secciones operativas) y `MANUAL_USUARIO.md` (sección 4); aquí se condensa la secuencia con anotaciones.

#### 5.2.1 Obtención del código y entorno virtual

```bash
# 1. Obtención del repositorio
git clone <repo-url> wo-shap
cd "wo-shap/W.O Python"

# 2. (Opcional pero recomendado) entorno virtual
python -m venv.venv
source.venv/bin/activate # Linux/macOS
#.venv\Scripts\activate # Windows

# 3. Instalación de dependencias
pip install -r requirements.txt
```

Trabajar dentro de un entorno virtual evita conflictos con instalaciones del sistema y es buena práctica reproducible. La carpeta de trabajo es la que contiene `runners/`, `wo_core/`, `shap_controller/`, `problems/`, `analysis/` y `experiments/`; **todos los comandos posteriores deben ejecutarse desde esta raíz**, dado que el runner añade dicha ruta a `sys.path` para resolver los paquetes locales.

#### 5.2.2 Estructura esperada del proyecto

Tras la instalación, la raíz debe contener las siguientes carpetas y archivos. Esta estructura está documentada en `README.md` y en el manual:

```
W.O Python/
├── wo_core/ Dinámica WO (movimiento, señales, roles, value function)
│ ├── walrus.py apply_wo_movement, iteration_signals, walrus_role_counts
│ ├── agent_sim.py make_value_function_for_agent (simulador de k pasos)
│ ├── fes.py FESBudget, counting_objective
│ ├── halton.py, levy_flight.py, initialization.py, diversity.py
├── shap_controller/ Controlador SHAP por agente
│ ├── controller.py SHAPFitnessController (compuertas, explain_fitness, decide)
│ ├── profiles.py Configuración ÚNICA en fracciones de MaxFES
│ ├── features.py Las 6 señales del WO como features SHAP
│ ├── actions.py reinit_random (Rama A) y reinit_guided (Rama B)
├── problems/ Adaptadores (contrato WOProblem)
│ ├── base.py, factory.py
│ ├── cec2022.py Envuelve opfunu, expone F1..F12
│ ├── tmlap.py Problema de asignación clientes-hubs
├── runners/
│ └── run_ablation.py Runner único base + shap, semillas pareadas
├── analysis/ Post-proceso: tablas, figuras, reportes
├── 1.instancia_simple.txt TMLAP (6c x 3h)
├── 2.instancia_mediana.txt TMLAP (12c x 5h)
├── 3.instancia_dura.txt TMLAP (24c x 8h) ← la instancia usada
└── experiments/ Salidas por configuración
```

La presencia de **`3.instancia_dura.txt`** en la raíz es condición necesaria para la ablación TMLAP. Las otras instancias (`1.instancia_simple.txt`, `2.instancia_mediana.txt`) acompañan al repositorio pero no se ejercitan en el barrido definitivo: las pequeñas resultan triviales para el WO y la grande (`4.instancia_grande.txt`) es operativamente inviable (≈ 25 días de pared estimados para 30 corridas a 5 x 10^5 FES, fuera del alcance temporal de la tesis).

#### 5.2.3 Verificación del ambiente: smoke test

Antes de lanzar cualquier barrido extenso, se realiza un *smoke test* de aproximadamente seis minutos sobre una sola función CEC. Este paso valida cuatro propiedades a la vez: que `opfunu` está instalado, que el runner alcanza la raíz del paquete, que las semillas pareadas producen poblaciones iniciales idénticas en `base` y `shap`, y que los gaps al óptimo son estrictamente positivos (un gap negativo o NaN delataría un bug crítico de signo, escala o clipping):

```bash
python -m runners.run_ablation \
 --problem cec2022:F10 --dim 10 --agents 30 \
 --max-fes 500000 --runs 3 --modes base,shap \
 --output experiments/_smoke_F10
```

Inmediatamente después se ejecuta un chequeo de invariantes con un snippet Python tomado de `COMO_EJECUTAR.txt §6`:

```bash
python -c "
import pandas as pd
b = pd.read_csv('experiments/_smoke_F10/base/values/summary.csv')
s = pd.read_csv('experiments/_smoke_F10/shap/values/summary.csv')
assert (b['gap_to_optimum'] > 0).all and (s['gap_to_optimum'] > 0).all
print('OK')
"
```

Si la aserción `(gap_to_optimum > 0).all` falla, el problema más probable es que un script *legacy* se esté ejecutando en lugar del runner unificado vigente. La regla operativa documentada es: **el único punto de entrada válido es `runners.run_ablation`**. Las carpetas `experiments/ablation_b4/` y sus scripts asociados se eliminaron en mayo de 2026; cualquier residuo que sobreviva debe ignorarse.

#### 5.2.4 Despliegue en servidor remoto

Para el barrido definitivo se utilizó un servidor Linux accedido por SSH. La preparación incluye dos pasos adicionales más allá de la instalación local:

1. **Sincronización del código** mediante `rsync` o `git pull`, asegurando que `requirements.txt`, los `.sh` envoltorios y las instancias TMLAP estén en la raíz del repositorio remoto.
2. **Configuración para sesiones persistentes**: los scripts `run_remote_cec.sh` y `run_remote.sh` invocan al runner unificado a través de `nohup`, redirigiendo `stdout` y `stderr` a un archivo de log (`ablation_cec2022_d10_sweep.log` y `ablation_ablation_b4_dura_30.log`). Esto permite que el experimento sobreviva a la desconexión del cliente SSH; el progreso se monitorea desde cualquier terminal con `tail -f`.

### 5.3 Evidencia de preparación del ambiente

La rúbrica exige evidencia tangible de que el ambiente quedó efectivamente operativo. El proyecto ofrece varias capas de evidencia, descritas a continuación. **Las capturas de pantalla de la sesión SSH —con `nohup` lanzando el barrido y `tail -f` mostrando el avance—** se incorporan como anexos gráficos en el documento final; se recomienda al lector adjuntar una o dos imágenes representativas: (a) la consola con el `pip install -r requirements.txt` finalizado, y (b) el `tail -f ablation_cec2022_d10_sweep.log` mostrando líneas del tipo `run=12/30 seed=... final=... gap=... fes_total=... t=...`.

#### 5.3.1 Logs persistentes

Los dos scripts de barrido emiten todo lo que el runner imprime por `stdout`/`stderr` a un archivo plano en la raíz del repositorio:

- **`ablation_cec2022_d10_sweep.log`** — barrido completo CEC 2022 (12 funciones x 4 presupuestos x 2 modos x 30 corridas). Cada línea informa, para la corrida en curso, la semilla, el `final_fitness`, el `gap_to_optimum`, el conteo de intervenciones (modo `shap`), los FES gastados en SHAP, el `fes_total/MaxFES` y el tiempo de pared. La presencia de estas líneas progresivas demuestra que la ejecución es real y no simulada.
- **`ablation_ablation_b4_dura_30.log`** — barrido TMLAP `3.instancia_dura.txt`, 30 corridas, `MaxFES = 500 000`.

El comando recomendado para monitoreo es `tail -f ablation_cec2022_d10_sweep.log`. La pantalla resultante constituye la captura más informativa para el informe.

#### 5.3.2 Salidas validadas en disco

Tras la finalización de los barridos, el árbol `experiments/` contiene las siguientes configuraciones, todas validadas para la entrega:

```
experiments/
├── cec2022_d10_fes5000/ base/ + shap/ con summary.csv (30 corridas)
├── cec2022_d10_fes50000/ base/ + shap/ con summary.csv (30 corridas)
├── cec2022_d10_fes500000/ base/ + shap/ con summary.csv (30 corridas)
├── cec2022_d10_fes5000000/ base/ + shap/ con summary.csv (30 corridas en base, 5 en shap)
├── dura_30_fes500k/ base/ + shap/ con summary.csv (30 corridas TMLAP)
└── presentacion/ Tablas y figuras agregadas
```

Cada subcarpeta `<config>/<modo>/values/summary.csv` contiene **una fila por corrida**, lo que en el escenario CEC totaliza 12 funciones x 30 corridas = 360 filas por modo y presupuesto (excepto `shap@5M`, con 5 filas, documentado explícitamente como "n=5, no concluyente"). La verificación rápida del conteo se ejecuta como:

```bash
python -c "
import pandas as pd
df = pd.read_csv('experiments/cec2022_d10_fes500000/shap/values/summary.csv')
print(df.groupby('problem').size)
"
```

Cada fila debe arrojar exactamente 30 corridas para los presupuestos 5k, 50k y 500k.

#### 5.3.3 Artefactos derivados del análisis

Los productos derivados que prueban la corrección del flujo completo se hallan en `experiments/presentacion/`:

- **`veredicto.md`** y **`veredicto_tmlap.md`** — resultado del test de Wilcoxon signed-rank pareado, con la notación `(W|T|L)` por presupuesto y por función.
- **`tabla_resumen_fes500000.csv`** — estadísticos por función a `MaxFES = 5 x 10^5`.
- **`tabla_tmlap.csv`** — homólogo para TMLAP `dura_30_fes500k`.
- **`actividad_controlador.csv`** — número de intervenciones y reparto entre `reinit_random`/`reinit_guided`.
- **`features_shap.csv`** — cuotas medias y conteos de dominancia por señal.
- **`shapiro_wilcoxon_cec.csv`** y **`shapiro_wilcoxon_tmlap.csv`** — diagnóstico de normalidad y test pareado.
- Figuras PNG: `convergencia/convergencia_cec_panel.png`, `convergencia_tmlap_dura_30_fes500k.png`, `boxplots_por_funcion.png`, `boxplots_tmlap.png`, `acciones_rescate.png`, `features_shap.png`, `features_shap_comparativa_cec.png`.

La existencia simultánea de los logs, los `summary.csv` con el número correcto de filas y los artefactos de análisis derivados constituye una **cadena de custodia operativa**: ninguno de ellos puede existir si los anteriores no se completaron correctamente.

#### 5.3.4 Validación de invariantes

Tres invariantes se comprueban en frío sobre los `summary.csv`:

1. **Semillas pareadas**: para todo `(MaxFES, problem, run_id)`, la columna `seed` coincide entre `base` y `shap`. Esto se verifica con `merge` sobre las tres claves. Si una semilla difiere, la comparación Wilcoxon pareada quedaría invalidada.
2. **Presupuesto respetado**: la columna `fes_total` ≤ `max_fes` en todas las filas, y `fes_init + fes_search + fes_shap + fes_intervention = fes_total`.
3. **Gap positivo**: `gap_to_optimum > 0` para CEC (`final_fitness` por encima del óptimo conocido); en TMLAP `dura` es NaN porque se invoca con `--no-exact-optimum`.

Estas tres invariantes se verifican manualmente con scripts ad-hoc y constituyen la evidencia más fuerte de que el ambiente y el runner se ejecutaron correctamente.

> **Nota sobre capturas**: la rúbrica solicita evidencia gráfica complementaria. En el documento PDF final se intercalan dos capturas de pantalla: (i) la sesión SSH mostrando `nohup./run_remote_cec.sh --runs 30 &` y la cabecera del log, y (ii) `tail -f ablation_cec2022_d10_sweep.log` con varias líneas de progreso. Se recomienda adjuntar también el listado `ls -lh experiments/cec2022_d10_fes500000/{base,shap}/values/summary.csv` para evidenciar tamaño y fecha de los entregables.

### 5.4 Documentación y manual de usuario

Una de las preocupaciones centrales del diseño fue que la implantación quedara documentada con suficiente granularidad como para que un tercero (un par revisor, un futuro tesista, o el propio autor del trabajo seis meses después) pudiera reproducir el estudio sin asistencia. El proyecto incluye cuatro documentos complementarios.

#### 5.4.1 Inventario de documentos

| Documento | Audiencia | Propósito |
|---|---|---|
| `README.md` | Lector inicial | Estructura, CLI básico, ejemplos canónicos, reproducibilidad |
| `MANUAL_USUARIO.md` | Usuario final | Manual exhaustivo de 11 secciones (este es el documento maestro) |
| `COMO_EJECUTAR.txt` | Operador del servidor | Procedimiento paso a paso de los barridos remotos |
| `Informe_Metodologia.md` | Tesista / revisor | Metodología canónica del método vigente |

#### 5.4.2 Las 11 secciones del MANUAL_USUARIO.md

El manual de usuario se redactó como un documento autocontenido, navegable y convertible a PDF mediante el utilitario `python -m analysis.md_to_pdf MANUAL_USUARIO.md MANUAL_USUARIO.pdf`. Sus once secciones cubren todo el ciclo de vida del software:

1. **Propósito y objetivos** — declaración del problema (mitigar estancamiento y convergencia prematura), del enfoque (controlador interpretable basado en SHAP sobre WO) y de los objetivos específicos relevantes (OE3 trazabilidad, OE4 integración y evaluación).
2. **Qué hace el sistema** — descripción operativa: dos modos pareados (`base` y `shap`); flujo del controlador (detectar estancamiento por agente, explicar con SHAP exacto sobre 64 coaliciones, decidir entre `reinit_random` o `reinit_guided`, reposicionar); criterio de parada `MaxFES`.
3. **Requisitos** — software (Python 3.11+, `numpy`, `pandas`, `scipy`, `opfunu`) y hardware (mínimo / recomendado).
4. **Instalación** — secuencia `pip install -r requirements.txt` y opcionalmente `venv`.
5. **Estructura del proyecto** — listado de carpetas con su responsabilidad.
6. **Uso — ejecutar experimentos** — flags del runner único, tabla de parámetros y ejemplos canónicos.
7. **Configuración del controlador** — tabla de los ocho parámetros del controlador, todos expresados como fracciones de MaxFES (ventana 10 %, guard 5 %, late 95 %, cooldowns 5 %, presupuesto SHAP 5 %, umbral 0.90, amplificación 2.0, 3 pasos Shapley).
8. **Salidas que produce** — qué archivos quedan en disco y qué columnas contienen.
9. **Análisis y figuras** — comandos para `presentation_summary`, `report_html` y `make_diagrams`.
10. **Interpretación de resultados** — guía rápida para leer `gap_to_optimum`, la notación `+/=/−` de Wilcoxon, los conteos `(W|T|L)`, la actividad del controlador.
11. **Solución de problemas** — errores comunes (`opfunu` faltante, ejecución fuera de la raíz, wall-clock excesivo, gaps NaN en TMLAP) con sus remedios.

#### 5.4.3 Comandos representativos verificados

Los tres comandos siguientes constituyen el núcleo operativo del proyecto y son los que un tercero ejecutaría para regenerar los resultados. Sus flags fueron verificados contra el `argparse` actual del runner (`runners/run_ablation.py`, junio 2026):

**Comando 1 — Barrido completo CEC 2022 (12 funciones, 4 presupuestos, 30 corridas):**

```bash
python -m runners.run_ablation \
 --problem cec2022:all --modes base,shap \
 --max-fes 5000,50000,500000,5000000 \
 --runs 30 --dim 10 \
 --output experiments/cec2022_d10
```

Este comando produce ocho subcarpetas `experiments/cec2022_d10/{base,shap}/` para cada presupuesto, cada una con su `summary.csv`, su carpeta de `curves/` y, en el modo `shap`, los tres CSV de telemetría del controlador.

**Comando 2 — Ablación TMLAP `3.instancia_dura.txt` (30 corridas, MaxFES = 5 x 10^5):**

```bash
python -m runners.run_ablation \
 --problem tmlap:3.instancia_dura.txt --modes base,shap \
 --max-fes 500000 --runs 30 \
 --no-exact-optimum --init-mode random \
 --output experiments/dura_30_fes500k
```

Los flags `--no-exact-optimum` e `--init-mode random` son específicos de la instancia dura: la primera evita el backtracking exhaustivo (que sería intratable para 24 clientes y 8 hubs), la segunda salta la búsqueda local inicial (que dominaría el tiempo de pared sin aportar al estudio).

**Comando 3 — Generación de tablas y figuras para la presentación:**

```bash
python -m analysis.presentation_summary --input experiments \
 --out-dir experiments/presentacion --highlight-fes 500000
```

El módulo de análisis produce el veredicto Wilcoxon, las tablas de resumen, los boxplots, los paneles de convergencia y los gráficos de actividad del controlador. El flag `--highlight-fes` selecciona el presupuesto destacado en la presentación (500 000, justificado en los resultados como el punto de operación más informativo).

#### 5.4.4 Columnas clave de `summary.csv`

El archivo `values/summary.csv` es el punto de entrada de todo análisis post-corrida. Sus columnas, verificadas en `run_ablation.py`, son las siguientes:

**Comunes a ambos modos:**

- `run_id`, `seed`, `algorithm` (`WO_base` o `WO_shap`), `problem_spec`, `problem_family`, `problem`, `dim`, `agents`, `max_fes`.
- `initial_fitness`, `final_fitness`, `optimum`, `gap_to_optimum`.
- `elapsed_seconds`, `t_init_seconds`, `t_shap_seconds` (NaN en `base`).
- Telemetría de estancamiento: `stagnation_window` (= `round(0.10 * MaxFES)`), `n_stagnant_at_end`, `mean_fes_since_improve_at_end`, `max_fes_since_improve_at_end`.
- Buckets de FES (suman `fes_total`): `fes_init`, `fes_search`, `fes_shap`, `fes_intervention`.
- `best_position` (cadena con la posición ganadora), `init_mode`.

**Solo en `shap`:**

- `controller_profile`, `interventions`, `shap_explanations`, `shapley_steps` (= 3).
- `n_reinit_random`, `n_reinit_guided` (recuento por rama de la acción).

Los CSV auxiliares del modo `shap` —`controller_events.csv` (una fila por intervención aceptada), `controller_non_events.csv` (candidatos bloqueados con su motivo) y `shap_values.csv` (las seis atribuciones SHAP por explicación y la señal dominante)— permiten reconstruir el comportamiento del controlador con granularidad de iteración.

#### 5.4.5 Scripts envoltorio para servidor

Los archivos `run_remote_cec.sh` y `run_remote.sh` ubicados en la raíz del proyecto son envoltorios delgados que invocan al runner unificado bajo `nohup`. Aceptan flags largos (`--runs`, `--dim`, `--agents`, `--budgets` para CEC; `--runs`, `--max-fes`, `--output` para TMLAP) y también un modo posicional *legacy* por retrocompatibilidad. Su valor agregado es operativo: redirigen el log a un archivo predecible y permiten que la ejecución sobreviva a la desconexión SSH. Para el barrido formal se invocaron así:

```bash
./run_remote_cec.sh --runs 30
./run_remote.sh --runs 30 --max-fes 500000
```

Ambos llaman al mismo runner por debajo, lo que garantiza que **no hay caminos de código duplicados**: el runner es el único punto de entrada autorizado.

#### 5.4.6 Reproducibilidad

La reproducibilidad se garantiza mediante una función de semilla determinista que es **independiente del modo**:

```
seed_run = base_seed + max_fes_idx * 100000 + problem_idx * 1000 + run_idx
```

Esto satisface tres propiedades:

1. Para un mismo `run_id`, los modos `base` y `shap` reciben la misma semilla y por tanto la **misma población inicial**, condición sine qua non del test de Wilcoxon pareado.
2. Corridas distintas de la misma función reciben semillas distintas, evitando réplicas idénticas.
3. Fijando la semilla raíz (`--seed 1234`, default) y manteniendo constantes las versiones de Python y de `numpy`, los resultados son bit-a-bit reproducibles.

Esta política, descrita tanto en el `README.md` como en el `MANUAL_USUARIO.md`, es la base estadística que permite afirmar las conclusiones del estudio (`shap ≈ base` en calidad, dominancia consistente de `safety_signal` en interpretabilidad, ausencia total de contribución de la señal `A`) con la confianza adecuada al rigor exigido por una tesis de pregrado.

---

En conjunto, la combinación de un runner único auditable, una configuración del controlador unificada y declarativa, un conjunto reducido pero suficiente de dependencias externas, una política de semillas pareadas explícita, una documentación de usuario en cuatro niveles complementarios (README, manual exhaustivo, archivo operativo del servidor, metodología canónica) y la persistencia de logs y CSV verificables, hace que la implantación cumpla simultáneamente los criterios de **portabilidad**, **reproducibilidad**, **observabilidad** y **mantenibilidad** exigidos a un artefacto de tesis. Los anexos visuales (capturas de la sesión SSH y del `tail -f` del log) completan la evidencia de que el ambiente se preparó y ejercitó efectivamente para producir los resultados que se discuten en la Sección 6.


## 6. Conclusiones

El presente trabajo se propuso aliviar el estancamiento y la convergencia prematura del Walrus Optimizer (WO) mediante un controlador interpretable basado en valores de Shapley (SHAP) que actúa **en línea** sobre la búsqueda. Las conclusiones de esta investigación se organizan en torno a cuatro ejes complementarios: los éxitos alcanzados, los avances que el trabajo aporta al área, los problemas y riesgos identificados a partir de la evidencia, y las oportunidades de trabajo futuro que se desprenden de manera directa de los hallazgos. El orden es deliberado: éxito → riesgo → oportunidad, en coherencia con el carácter científico del estudio. El resultado de no mejora significativa en calidad no se interpreta como un fracaso, sino como un hallazgo válido que **delimita** el alcance del aporte demostrado (interpretabilidad) y **abre** una línea natural de continuación (refactor direccional del controlador).

### 6.1 Éxitos conseguidos

El primer éxito de esta investigación es haber materializado un **controlador interpretable en línea funcionando** sobre una metaheurística poblacional. El lazo propuesto se ejecuta en cuatro etapas integradas con el flujo del WO —detectar, explicar, decidir y reposicionar— y no consiste en un reporte posterior a la corrida. La detección se efectúa por agente mediante un contador en evaluaciones de la función objetivo (FES), gatilla recién cuando el agente lleva al menos un 10 % del MaxFES sin mejorar su mejor personal; la explicación se realiza por SHAP exacto sobre las seis señales de control del WO; la decisión bifurca una única acción de rescate en función de la dominancia de la señal explicativa; y el reposicionamiento se materializa, según la rama elegida, como reinicio uniforme o como paso del operador WO con amplificación de la señal dominante. Que las cuatro etapas estén integradas y se ejecuten dentro del presupuesto FES constituye la primera contribución verificable del trabajo.

El segundo éxito corresponde al cumplimiento del **Objetivo Específico 3** (interpretabilidad). La atribución SHAP identifica de manera trazable cuál de las seis señales del WO (`alpha`, `beta`, `A`, `R`, `danger_signal`, `safety_signal`) explica el estancamiento de cada agente. A presupuesto de 5·10⁵ FES y sobre 12 funciones con 30 corridas, se recopilaron 3 563 explicaciones cuyo agregado muestra un patrón nítido: `safety_signal` domina tanto en magnitud media (cuota 0,39) como en frecuencia de dominancia (1 193 ocurrencias), seguida por `danger_signal` (cuota 0,27; 848 ocurrencias) y `beta` (cuota 0,20; 547 ocurrencias); `alpha` ocupa un rol secundario (0,10 de cuota); `R` resulta marginal (0,028); y `A` **nunca contribuye** (cuota 0) en ningún experimento, lo que la cataloga empíricamente como señal irrelevante para el diagnóstico del estancamiento. Más aún, la lectura **evoluciona con el presupuesto**: a 5·10³ FES domina `danger_signal` (cuota 0,50), mientras que a 5·10⁴ FES en adelante toma el relevo `safety_signal` (0,39). El controlador, por tanto, observa empíricamente cómo el estancamiento temprano se explica por la señal de exploración/peligro y el estancamiento tardío por la señal de seguridad. Este resultado interpretativo se sostiene también en el caso aplicado TMLAP en su instancia dura, donde `safety_signal` mantiene una cuota de 0,41, consistente con CEC a presupuestos altos. Las figuras `experiments/presentacion/features_shap.png` y `experiments/presentacion/features_shap_comparativa_cec.png` documentan estos hallazgos. La caja negra del WO, en consecuencia, queda abierta a una descripción cuantitativa de su dinámica interna [Han 2024].

El tercer éxito es haber demostrado que la **explicabilidad in-ejecución es viable** en términos de costo computacional. El uso de SHAP **exacto** sobre las 2⁶ = 64 coaliciones —factible por el reducido número de features— consume 192 FES por explicación (64 coaliciones × 3 pasos de la value function simulada). Con aproximadamente 10 intervenciones por corrida (media 9,9), el costo total ronda los 1 900 FES sobre un presupuesto de 5·10⁵, esto es, **0,4 % del presupuesto** total. El WO sigue haciendo el trabajo masivo de optimización y SHAP sólo paga su evaluación cuando se la requiere. La actividad del controlador se distribuye en aproximadamente 1 957 ramas `reinit_random` y 1 606 ramas `reinit_guided` (≈ 3 563 explicaciones), lo que evidencia un controlador **acotado y trazable**.

El cuarto éxito atañe al cumplimiento del **Objetivo Específico 4** (impacto) con una **evaluación rigurosa y honesta**. La comparación se diseñó como ablación **pareada** (mismas semillas para `base` y `shap`, donde la semilla depende exclusivamente del índice de problema, de presupuesto y de corrida, no del modo), lo que habilita un test pareado correctamente fundado. Se aplicó primero Shapiro–Wilk para evaluar la normalidad de las distribuciones de costo final por función y, dado que la mayoría no superó la normalidad (p < 0,05; F4 es la única función con ambos grupos normales en CEC, y en TMLAP la condición `base` es normal con p = 0,259 pero `shap` no con p = 0,0037), se adoptó el test no paramétrico de **Wilcoxon signed-rank** a dos colas con α = 0,05. El benchmark cubrió las 12 funciones de CEC2022 en dimensión 10 a cuatro presupuestos (5·10³, 5·10⁴, 5·10⁵, 5·10⁶) con 30 corridas por configuración, más el caso aplicado TMLAP en su instancia dura (24 clientes × 8 hubs). Los archivos `experiments/presentacion/shapiro_wilcoxon_cec.csv` y `experiments/presentacion/shapiro_wilcoxon_tmlap.csv` documentan el procedimiento completo.

El quinto éxito es operativo pero significativo: un **pipeline reproducible** con **configuración única**. Un sólo runner (`runners/run_ablation.py`) ejecuta la ablación pareada; todos los parámetros del controlador se expresan como **fracciones de MaxFES** (ventana de estancamiento al 10 %, cooldowns y *guard window* al 5 %, presupuesto SHAP al 5 %, gate de etapa tardía al 95 %), de modo que no requiere re-tuning al cambiar el presupuesto. Esto facilita la replicación externa y elimina la confusión entre efecto del método y artefactos de ajuste fino.

### 6.2 Avances en el área

El avance principal del trabajo es haber utilizado XAI **como controlador prescriptivo y en línea**, no como herramienta descriptiva post-hoc. La literatura emplea SHAP mayoritariamente para auditar modelos entrenados después de su uso; las hiper-heurísticas y enfoques adaptativos sí intervienen durante la búsqueda, pero lo hacen sin un mecanismo de interpretabilidad explícito. Esta tesis ocupa precisamente el hueco entre ambos cuerpos de literatura: SHAP se calcula **dentro** de la corrida, sobre las señales que el algoritmo ya gestiona, y su resultado **decide** una acción concreta sobre un agente concreto. La explicación deja de ser un epílogo y se convierte en un componente de decisión.

El segundo avance es de naturaleza empírica y consiste en **caracterizar la dinámica interna del WO**. Antes de este trabajo, las seis señales del Walrus Optimizer eran parámetros funcionales documentados en su paper original [Han et al. 2024] pero sin una atribución cuantitativa sobre **cuáles** explican el estancamiento de los agentes durante la búsqueda. Los resultados aportan esa cuantificación: `safety_signal` y `danger_signal` concentran la explicación; `A` es irrelevante; y la palanca dominante **cambia con el presupuesto**. Este hallazgo no es trivial respecto al diseño del WO, porque sugiere que su régimen agrupativo (gobernado por `safety_signal`) es el que más se ve comprometido cuando la búsqueda pierde tracción a presupuestos medios y altos. Es, hasta donde se conoce, la primera caracterización SHAP de la dinámica interna del Walrus Optimizer.

El tercer avance es la propuesta de un **marco general y reusable**, agnóstico del algoritmo subyacente. Aunque la implementación se ancla al WO, la receta es transferible a cualquier metaheurística cuya iteración esté gobernada por un conjunto pequeño de señales de control: basta con redefinir las features SHAP como las señales internas del algoritmo, conservar la value function que simula k pasos del agente y reutilizar la acción única bifurcada por dominancia. El controlador se desacopla del operador específico y queda como una **capa de control** que cualquier metaheurística poblacional con señales internas explícitas puede instrumentar.

El cuarto avance se refiere al posicionamiento respecto a Črepinšek et al. [2025]. Ese trabajo define con precisión el fenómeno de estancamiento (un algoritmo deja de mejorar su mejor solución durante un período prolongado, medido en iteraciones, tiempo o FES) y propone MsMA, un mecanismo que detecta el estancamiento por FES y reinicia la búsqueda. La presente investigación comparte tanto la unidad de medida (FES) como el criterio operativo (detección por porcentaje de MaxFES sin mejora) pero opera en un nivel distinto: MsMA es **meta-nivel** y no incorpora interpretabilidad, mientras que aquí el control es **intra-corrida y por agente**, además interpretable. Ambas aproximaciones son complementarias y, eventualmente, combinables.

### 6.3 Problemas y/o riesgos identificados

El primer hallazgo crítico es honesto: el controlador **no mejora la calidad de forma significativa**. El veredicto Wilcoxon agregado por presupuesto sobre las 12 funciones de CEC2022 arroja, para shap vs base, (W|T|L) = (0|12|0) a 5·10³ FES, (1|11|0) a 5·10⁴, (0|11|1) a 5·10⁵ y (0|12|0) a 5·10⁶; el único caso "−" corresponde a F2 con p = 0,0154 a 5·10⁵, y la única "+" a una función a 5·10⁴ FES. El caso aplicado TMLAP en su instancia dura a 5·10⁵ FES también empata, con p = 0,58. Globalmente la mayoría de las funciones empata y la hipótesis de mejora **no se confirma**. Los archivos `experiments/presentacion/veredicto.md` y `experiments/presentacion/veredicto_tmlap.md` recogen el detalle.

El segundo problema, y a juicio de este trabajo el más importante, es **mecanístico** y constituye la lectura razonada del hallazgo anterior. La rama B del controlador (`reinit_guided`) amplifica la señal dominante alejándola del neutro al doble mediante la transformación s' = b + 2·(s − b) y, sobre esa señal amplificada, ejecuta un paso del operador WO desde la posición actual del agente. El problema es que **SHAP marca la señal más influyente, no necesariamente la "mala"**: identifica qué palanca pesa más en el desempeño del agente, sin pronunciarse sobre el **sentido** en el que ese peso es perjudicial. Amplificar el desvío respecto al neutro, en consecuencia, **a veces ayuda y a veces empeora**; en promedio sobre las 12 funciones y las 30 corridas, los efectos se cancelan. Esta es, mecanísticamente, la razón por la cual **shap ≈ base**: el lazo de control está mal cerrado porque la atribución no está siendo convertida en una dirección de corrección, sino en una dilatación ciega.

El tercer problema es estructural: las **señales del WO son globales**. Las seis features que ve SHAP son iguales para todos los agentes en cada iteración (son magnitudes temporales del algoritmo, no descriptores del agente). En consecuencia, la atribución SHAP indica **qué** palanca pesa, pero no **dónde** reposicionar el agente que se está rescatando. Sin información posicional o geométrica del agente, la acción de reposicionamiento opera con información incompleta.

El cuarto problema atañe a la modulación: la cuota dominante tiende a **degenerar** hacia 1 conforme la búsqueda avanza, lo que en presencia de un umbral de dominancia **estricto** (0,90) deja a la rama B activándose con mucha frecuencia y a la rama A en proporción reducida. La rigidez del umbral merece recalibración.

El quinto problema es de **alcance estadístico**. La condición `shap` a 5·10⁶ FES sólo cuenta con n = 5 corridas (el costo de SHAP a ese presupuesto es prohibitivo en el hardware disponible), por lo que ese presupuesto **no es concluyente**. Tampoco se aplicaron correcciones por **comparaciones múltiples** (Holm/Bonferroni); el único "−" observado (F2 a 5·10⁵) podría ser un **falso positivo**, dado que con 12 funciones y α = 0,05 se espera al azar aproximadamente un rechazo por presupuesto. Asimismo, al tratarse de dos condiciones (base vs shap), no es aplicable el ranking de Friedman, que requiere al menos tres competidores.

El sexto problema es de **alcance experimental**. El estudio se restringe a dimensión 10 y a un único benchmark continuo (CEC2022). El caso aplicado TMLAP carece de óptimo exacto y, de sus tres instancias, sólo la "dura" ejercita realmente al WO (las pequeñas son triviales y la grande, de aproximadamente 25 días por corrida, es inviable en el cronograma).

### 6.4 Oportunidades y/o propuestas de trabajo futuro

La **oportunidad principal**, que se deriva mecanísticamente del problema central diagnosticado en §6.3, es el **refactor direccional** del controlador. La idea es utilizar el **signo del valor SHAP** de la señal dominante para empujarla **hacia** donde mejora el fitness, en lugar de amplificar a ciegas su desvío respecto al neutro. Operacionalmente, si la value function indica que el aporte SHAP de la señal dominante a la fitness es negativo, debe **contrarrestarse** la palanca (no amplificarla); si es positivo, debe **acentuarse**. Este refactor permite cerrar el lazo de control de forma más principiada: la atribución dejaría de responder sólo a "qué" palanca pesa, para responder además a "en qué dirección" moverla. Es la consecuencia natural del Problema 2 de §6.3 y la siguiente línea de continuación del trabajo.

Una segunda línea, complementaria a la anterior, es enriquecer el conjunto de features con **señales por-agente (locales)**. Las seis señales actuales son globales y no aportan información posicional. Incorporar descriptores geométricos del agente —distancia al `gbest`, distancia al centroide poblacional, hacinamiento, descomposición dimension-wise— habilitaría a SHAP a responder no sólo **qué** palanca pesa, sino **dónde** reposicionar. Combinada con el refactor direccional, esta línea convierte al controlador en un mecanismo verdaderamente prescriptivo en términos de acción concreta sobre el agente.

Una tercera línea es la **recalibración** del umbral de dominancia (0,90) y del factor de amplificación (2,0). Ambos hiperparámetros se fijaron por criterio del guía durante el desarrollo y no se exploraron sistemáticamente. La degeneración de la cuota dominante observada en §6.3 sugiere que un umbral **adaptativo** —en función del progreso de la búsqueda o de la dispersión de las cuotas— podría regular mejor la mezcla `reinit_random`/`reinit_guided`.

Una cuarta línea consiste en **ampliar la evidencia experimental**. Concretamente: incorporar CEC2014 y CEC2017 para completar el protocolo de 72 problemas habitual en la comunidad, escalar a dimensión 20, y completar `shap` a 5·10⁶ FES con las 30 corridas faltantes (hoy n = 5 por costo). Con ello, el veredicto a presupuesto alto dejaría de ser no concluyente.

Una quinta línea es la **comparación contra competidores** estándar (EA4eig, LSHADE, AGSK, APGSK-IMODE) y contra **MsMA** [Črepinšek et al. 2025], que también detecta estancamiento por FES y reinicia, pero a nivel meta y sin XAI. Esta ampliación a más de dos condiciones habilita el ranking de **Friedman** y un análisis más completo del posicionamiento relativo.

Una sexta línea es metodológica: aplicar **correcciones por comparaciones múltiples** (Holm o Bonferroni) y reportar los resultados con tamaño del efecto. Esto disminuye el riesgo de falsos positivos como el de F2 a 5·10⁵ y robustece la lectura estadística.

Una séptima línea es la **generalización del controlador** a otras metaheurísticas (Particle Swarm Optimization, Grey Wolf Optimizer, Differential Evolution), validando la transferibilidad del marco. La hipótesis es que cualquier metaheurística con un conjunto pequeño de señales de control puede instrumentarse con SHAP exacto, dado el costo despreciable demostrado en este trabajo.

Una octava línea es medir explícitamente la **diversidad poblacional** intra-corrida, como indicador de exploración "real", para complementar la varianza inter-corridas observada en este trabajo (donde shap tiende descriptivamente a menor desviación estándar; por ejemplo, TMLAP con desviación estándar 1,6 frente a 1,8 de base, o F5 de CEC con 0,04 frente a 0,12). El análisis de diversidad permitirá pasar de la conjetura descriptiva a un resultado contrastable.

Finalmente, una novena línea consiste en **cerrar más el lazo de control**: utilizar la atribución SHAP no sólo para elegir entre dos ramas discretas (`reinit_random` o `reinit_guided`), sino para **modular continuamente** parámetros del WO durante la corrida, de modo análogo a un control retroalimentado. Esta línea convierte al controlador en una verdadera capa de control adaptativo informado por interpretabilidad, integrando el refactor direccional, las señales locales y la modulación continua en un único mecanismo.

En síntesis, esta tesis demuestra que la explicabilidad puede dejar de ser un epílogo descriptivo para convertirse en un componente activo de control en línea sobre una metaheurística poblacional, con un costo despreciable y a un nivel de detalle por agente. El controlador no mejora significativamente la calidad de las soluciones, pero documenta de manera trazable y reproducible cuál de las seis señales del Walrus Optimizer explica el estancamiento, cómo evoluciona esa atribución con el presupuesto y en qué proporción el controlador interviene. El hallazgo de no mejora en calidad, lejos de ser una limitación cerrada, identifica con precisión mecanística el problema —amplificación a ciegas de la señal dominante— y delimita la siguiente línea de trabajo: convertir la atribución SHAP en una **dirección** de corrección. Es, en términos científicos, el final esperado de una primera iteración honesta del método y el comienzo razonado de la siguiente.


## Bibliografía

- Crawford, B. (2023). *Metaheurísticas y sus aplicaciones a problemas de optimización combinatoria*. Pontificia Universidad Católica de Valparaíso.
- Das, S., Mullick, S. S., & Suganthan, P. N. (2022). Recent advances in differential evolution: An updated survey. *Swarm and Evolutionary Computation*, 70, 101046.
- Han, M., Du, Z., Yuen, K. F., Zhu, H., Li, Y., & Yuan, Q. (2024). Walrus Optimizer: A novel nature-inspired metaheuristic algorithm. *Expert Systems with Applications*, 239, 122413.
- Kalasampath, K., Spoorthi, K. N., Sajeev, S., Kumar, S. S., Ramesh, K., & Nair, R. R. (2025). A literature review on applications of explainable artificial intelligence (XAI). *Heliyon*, 11(3), e35123.
- Somvanshi, S., Kumar, R., Sharma, S., & Singh, M. (2025). Bio-inspired metaheuristics: Recent advances and trends. *Swarm and Evolutionary Computation*, 84, 101492.

## Anexo A — Glosario de siglas

| Sigla | Significado |
|---|---|
| **FES** | Function Evaluations / Evaluaciones de la función objetivo. Unidad de cómputo estándar para comparar metaheurísticas. |
| **MaxFES** | Presupuesto máximo de FES; criterio de parada de las corridas. |
| **WO** | Walrus Optimizer; metaheurística poblacional bio-inspirada propuesta por Han et al. (2024). |
| **SHAP** | SHapley Additive exPlanations; método de explicabilidad basado en valores de Shapley (Lundberg & Lee, 2017). |
| **CEC** | Congress on Evolutionary Computation; serie de competencias y benchmarks estandarizados de optimización. |
| **CEC2022** | Benchmark de 12 funciones de la competencia CEC 2022 Single Objective Bound Constrained. |
| **TMLAP** | Telecommunications Multi-Level Assignment Problem; problema combinatorio aplicado del proyecto (asignación clientes–hubs). |
| **GBest** | Global Best; mejor solución encontrada por la población hasta el momento. |
| **PBest** | Personal Best; mejor solución encontrada por un agente individual. |
| **XAI** | eXplainable Artificial Intelligence; inteligencia artificial explicable. |
| **W \| T \| L** | Wins \| Ties \| Losses; tupla agregada de funciones en las que `shap` gana, empata o pierde frente a `base` según Wilcoxon. |
| **MsMA** | Multistart Meta-Aware; mecanismo de detección de estancamiento y reinicio propuesto por Črepinšek et al. (2025). |
| **PSO** | Particle Swarm Optimization. |
| **GWO** | Grey Wolf Optimizer. |
| **DE** | Differential Evolution. |
| **GA** | Genetic Algorithm. |
| **WOA** | Whale Optimization Algorithm. |
| **LB / UB** | Lower / Upper Bound; límites del dominio de búsqueda. |
| **RNG** | Random Number Generator; generador de números pseudoaleatorios. |

