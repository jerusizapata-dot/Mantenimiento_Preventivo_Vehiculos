# Modelos 2 y 3

Responsable: Manuel Mañana

## Objetivo

Desarrollar y evaluar los modelos predictivos 2 y 3 del Sistema de Apoyo a Decisiones para el Mantenimiento Preventivo de Vehículos.

## Estructura

- `scripts/`: implementación de los modelos 2 y 3.
- `resultados/`: métricas, predicciones y resultados generados.
- `tests/`: pruebas asociadas a los modelos.

## Modelos

- Modelo 2: pendiente de implementación.
- Modelo 3: pendiente de implementación.


# Modelos 2 y 3 — Sistema de Apoyo a Decisiones para el Mantenimiento Preventivo de Vehículos

**Responsable:** Manuel Mañana  git statu
**Rama:** `feature/modelos-2-3-manuel`  
**Estado:** Modelos iniciales entrenados; optimización y validación final pendientes.

## 1. Objetivo

Desarrollar, entrenar y evaluar dos modelos de clasificación binaria para estimar la necesidad de mantenimiento preventivo de vehículos:

- **Modelo 2:** Árbol de decisión (`DecisionTreeClassifier`).
- **Modelo 3:** Bosque aleatorio (`RandomForestClassifier`).

La contribución busca generar predicciones y probabilidades que permitan comparar algoritmos y sirvan como insumos para las reglas de priorización del DSS.

## 2. Conjunto de datos

Se utiliza `datos/raw/vehicle_maintenance_raw.csv`, el mismo archivo empleado por Oscarina para el baseline y el modelo 1.

El conjunto contiene 50,000 registros y una variable objetivo: `need_maintenance`.

- `0`: no requiere mantenimiento.
- `1`: requiere mantenimiento.

La distribución es de aproximadamente 19 % de registros negativos y 81 % positivos.

**Variables predictoras:**

| Variable | Descripción |
|---|---|
| `mileage` | Kilometraje acumulado |
| `vehicle_age` | Antigüedad del vehículo |
| `reported_issues` | Problemas mecánicos reportados |
| `service_history` | Servicios realizados |
| `vehicle_model` | Tipo de vehículo |
| `maintenance_history` | Calidad del historial de mantenimiento |

Se mantienen las mismas seis variables utilizadas por Oscarina para asegurar una comparación homogénea.

## 3. Metodología

### División de datos

Se reutilizan los índices generados por Oscarina:

- Entrenamiento: 40,000 registros (80 %).
- Prueba: 10,000 registros (20 %).

Esto permite comparar los modelos utilizando exactamente las mismas observaciones.

### Preprocesamiento

Los modelos incorporan un pipeline con:

- Imputación por mediana para variables numéricas.
- Imputación por moda para variables categóricas.
- Codificación One-Hot de variables categóricas.
- Ajuste del preprocesamiento exclusivamente con los datos de entrenamiento.

No se requiere estandarización para los algoritmos basados en árboles utilizados.

## 4. Modelos implementados

### Modelo 2 — Árbol de decisión

Archivo: `scripts/modelo_2.py`

Algoritmo: `DecisionTreeClassifier`.

Parámetros iniciales:

- `max_depth=6`
- `min_samples_leaf=30`
- `random_state=42`

El algoritmo construye reglas de decisión a partir de las variables predictoras. Sus parámetros actuales son iniciales y todavía no se han optimizado.

### Modelo 3 — Random Forest

Archivo: `scripts/modelo_3.py`

Algoritmo: `RandomForestClassifier`.

Parámetros iniciales:

- `n_estimators=100`
- `max_depth=12`
- `min_samples_leaf=20`
- `random_state=42`
- `n_jobs=-1`

El modelo combina 100 árboles de decisión para producir predicciones y probabilidades de mantenimiento.

## 5. Evaluación inicial

Se calcularon Accuracy, Precision, Recall, F1-score y matrices de confusión, considerando positiva la clase `need_maintenance = 1`.

| Métrica | Baseline | Regresión logística | Árbol | Random Forest |
|---|---:|---:|---:|---:|
| Accuracy | 0.8100 | 0.7931 | 0.8096 | 0.8098 |
| Precision | 0.8100 | 0.8300 | 0.8109 | 0.8124 |
| Recall | 1.0000 | 0.9363 | 0.9977 | 0.9949 |
| F1-score | 0.8950 | 0.8800 | 0.8946 | 0.8945 |
| Especificidad | 0.0000 | 0.1826 | 0.0079 | 0.0205 |

### Matriz de confusión — Modelo 2

| Clase real | Predice 0 | Predice 1 |
|---|---:|---:|
| 0 | 15 | 1,885 |
| 1 | 19 | 8,081 |

### Matriz de confusión — Modelo 3

| Clase real | Predice 0 | Predice 1 |
|---|---:|---:|
| 0 | 39 | 1,861 |
| 1 | 41 | 8,059 |

### Interpretación inicial

Los dos modelos presentan una sensibilidad elevada, pero identifican muy pocos casos de la clase negativa.

Con el umbral estándar de 0.50, el Modelo 2 clasifica el 99.66 % de los vehículos como necesitados de mantenimiento y el Modelo 3 el 99.20 %.

Estos resultados demuestran la necesidad de evaluar otras configuraciones y umbrales antes de utilizar los modelos para priorizar intervenciones.

## 6. Experimento exploratorio de umbrales

Archivo: `scripts/evaluar_umbrales.py`.

Se subdividieron los 40,000 registros de entrenamiento en 32,000 de desarrollo y 8,000 de validación interna.

El objetivo exploratorio consistió en maximizar la especificidad manteniendo un recall mínimo del 90 %.

| Indicador | Árbol | Random Forest |
|---|---:|---:|
| ROC-AUC | 0.8486 | 0.8469 |
| Average Precision | 0.9612 | 0.9647 |
| Umbral candidato | 0.54 | 0.56 |
| Recall | 0.9569 | 0.9235 |
| Especificidad | 0.1579 | 0.2743 |
| Falsos positivos | 1,280 | 1,103 |
| Falsos negativos | 279 | 496 |

**Estos umbrales son experimentales, no definitivos.** Se seleccionaron y evaluaron en la misma partición de validación, por lo que deben verificarse mediante un procedimiento más robusto antes de utilizarlos.

## 7. Estructura de archivos

```text
integrantes/manuel_manana/
├── README.md
├── scripts/
│   ├── verificar_datos.py
│   ├── modelo_2.py
│   ├── modelo_3.py
│   └── evaluar_umbrales.py
├── resultados/
│   ├── metricas_modelo_2.json
│   ├── metricas_modelo_3.json
│   ├── predicciones_modelo_2.csv
│   ├── predicciones_modelo_3.csv
│   ├── modelo_2_arbol_decision.joblib
│   └── modelo_3_random_forest.joblib
└── tests/
    └── .gitkeep
```

Los resultados se generan durante el entrenamiento. Los archivos `.joblib` se mantienen como artefactos de ejecución mientras se coordina su integración con la aplicación.

## 8. Reproducibilidad

El desarrollo utiliza Python 3.12, con dependencias instaladas mediante `uv`.

Bibliotecas principales:

- pandas
- numpy
- scikit-learn
- joblib
- pytest

Desde la raíz del repositorio, preparar el entorno:

```powershell
uv python install 3.12
uv venv --python 3.12
uv pip install --python .venv pandas numpy scikit-learn joblib pytest
```

Ejecutar la validación:

```powershell
.\.venv\Scripts\python.exe integrantes/manuel_manana/scripts/verificar_datos.py
```

Entrenar los modelos:

```powershell
.\.venv\Scripts\python.exe integrantes/manuel_manana/scripts/modelo_2.py
.\.venv\Scripts\python.exe integrantes/manuel_manana/scripts/modelo_3.py
```

Ejecutar el experimento de umbrales:

```powershell
.\.venv\Scripts\python.exe integrantes/manuel_manana/scripts/evaluar_umbrales.py
```

## 9. Actividades pendientes

- Optimización de hiperparámetros mediante validación cruzada.
- Ajuste y validación de umbrales de decisión.
- Evaluación de sensibilidad, especificidad, ROC-AUC, Average Precision y calibración.
- Desarrollo de pruebas automatizadas.
- Documentación de configuraciones finales.
- Preparación de los modelos para su integración por Laudys.
- Entrega de resultados para la evaluación de Lennen y las reglas de Marlin.

## 10. Limitaciones

Los datos presentan desbalance de clases y los modelos iniciales muestran baja especificidad con el umbral estándar.

Solo se utilizaron seis variables predictoras. No se dispone todavía de validación externa ni de una evaluación operativa del costo de los falsos positivos y negativos.

El conjunto de prueba oficial ya fue examinado durante el desarrollo exploratorio. Por tanto, una evaluación posterior sobre esos mismos registros deberá interpretarse considerando esta limitación.

Las probabilidades generadas no han sido calibradas y no deben asumirse automáticamente como riesgos reales.

## 11. Conclusión provisional

Se implementaron y entrenaron correctamente los modelos 2 y 3 utilizando el conjunto de datos, las variables y los índices de partición compartidos con el modelo 1.

Ambos generan métricas, predicciones y probabilidades utilizables como insumos de evaluación. No obstante, sus versiones iniciales presentan baja capacidad para identificar vehículos que no requieren mantenimiento cuando utilizan el umbral predeterminado.

Los experimentos de umbrales muestran posibilidades de mejorar el equilibrio entre sensibilidad y especificidad, pero todavía se requiere optimización y validación adicional.

La selección final de modelos deberá responder a criterios técnicos, estadísticos y operativos acordados por el equipo del DSS.
