
# Baseline y Modelo 1: Mantenimiento Preventivo de Vehículos

**Responsable:** Oscarina Brazoban  
**Asignatura:** Ciencia de Datos II (INF-8239)  
**Proyecto:** Sistema de Apoyo a Decisiones para el Mantenimiento Preventivo de Vehículos  
**Modalidad:** B - Sistema de Soporte a Decisiones (DSS)

## 1. Objetivo

Desarrollar, entrenar y evaluar un modelo de referencia (baseline) y un primer modelo predictivo para identificar vehículos que requieren mantenimiento preventivo inmediato.

Esta contribución proporciona una base cuantitativa para comparar los modelos desarrollados por el equipo y apoyar la construcción de reglas de decisión para la priorización del mantenimiento vehicular.

## 2. Descripción del conjunto de datos

Se utiliza el conjunto de datos público Vehicle Maintenance Data, disponible en Kaggle.

- Registros originales: 50,000.
- Variables originales: 20.
- Valores faltantes identificados: 0.
- Registros duplicados identificados: 0.
- Variable objetivo: `need_maintenance`.

La variable objetivo presenta dos categorías:

- `0`: el vehículo no requiere mantenimiento inmediato.
- `1`: el vehículo requiere mantenimiento inmediato.

### Distribución de la variable objetivo

| Clase | Registros | Porcentaje |
|---|---:|---:|
| 0 - No requiere mantenimiento | 9,502 | 19.00 % |
| 1 - Requiere mantenimiento | 40,498 | 81.00 % |
| Total | 50,000 | 100.00 % |

Se identifica un desbalance de clases que debe considerarse durante la evaluación de los modelos.

## 3. Variables predictoras

Para esta primera versión se seleccionaron seis variables, siguiendo el diccionario de datos elaborado durante el análisis exploratorio.

| Variable | Descripción |
|---|---|
| `vehicle_model` | Tipo o categoría del vehículo |
| `mileage` | Kilometraje acumulado |
| `vehicle_age` | Antigüedad del vehículo |
| `reported_issues` | Cantidad de problemas mecánicos reportados |
| `maintenance_history` | Calidad del historial de mantenimiento |
| `service_history` | Cantidad de servicios realizados |

Estas variables permiten analizar factores asociados con el uso, la antigüedad y los antecedentes de mantenimiento de los vehículos.

## 4. Metodología

### 4.1 Preparación de datos

El script de entrenamiento utiliza el conjunto de datos original ubicado en:

`datos/raw/vehicle_maintenance_raw.csv`

Se aplican las siguientes operaciones:

1. Estandarización de los nombres de las columnas.
2. Eliminación de registros duplicados.
3. Exclusión de registros con variable objetivo faltante.
4. Filtrado de valores negativos en las variables numéricas seleccionadas.
5. Separación de variables predictoras y variable objetivo.
6. División de los datos en entrenamiento y prueba.

El tratamiento de valores faltantes de las variables predictoras se incorpora al preprocesamiento del modelo de regresión logística para evitar utilizar estadísticas del conjunto de prueba durante el entrenamiento.

### 4.2 División de datos

Se utiliza una partición estratificada para conservar aproximadamente la distribución de la variable objetivo.

| Conjunto | Registros | Porcentaje |
|---|---:|---:|
| Entrenamiento | 40,000 | 80 % |
| Prueba | 10,000 | 20 % |

Parámetros utilizados:

- `test_size=0.20`
- `random_state=42`
- `stratify=y`

Los índices de ambas particiones se guardan en archivos CSV para facilitar la reproducibilidad y la comparación con los modelos desarrollados por otros integrantes del equipo.

## 5. Modelos implementados

### 5.1 Baseline: DummyClassifier

Se implementa un modelo de referencia mediante:

`DummyClassifier(strategy="most_frequent")`

Este modelo predice siempre la clase mayoritaria, sin aprender relaciones entre las variables predictoras.

Su propósito es establecer un desempeño mínimo de referencia para evaluar si los modelos más complejos aportan mejoras.

### 5.2 Modelo 1: Regresión logística

Se implementa un modelo de clasificación binaria mediante:

`LogisticRegression(max_iter=1000, random_state=42)`

El modelo utiliza un pipeline de preprocesamiento con:

- Imputación mediante mediana para variables numéricas.
- Estandarización mediante `StandardScaler`.
- Imputación mediante moda para variables categóricas.
- Codificación mediante `OneHotEncoder(handle_unknown="ignore")`.

La regresión logística permite generar predicciones y estimar probabilidades que pueden servir como insumo para las reglas de decisión del DSS.

## 6. Evaluación de los modelos

Se utilizaron las siguientes métricas:

- **Accuracy:** proporción de predicciones correctas.
- **Precision:** proporción de vehículos realmente necesitados de mantenimiento entre aquellos clasificados como positivos.
- **Recall:** proporción de vehículos que requieren mantenimiento y fueron identificados correctamente.
- **F1-score:** media armónica de precision y recall.
- **Matriz de confusión:** distribución de aciertos y errores por clase.

Las métricas de precision, recall y F1-score presentadas a continuación corresponden a la clase positiva (`need_maintenance = 1`).

### 6.1 Resultados comparativos

| Métrica | Baseline | Regresión logística |
|---|---:|---:|
| Accuracy | 0.8100 | 0.7931 |
| Precision | 0.8100 | 0.8300 |
| Recall | 1.0000 | 0.9363 |
| F1-score | 0.8950 | 0.8800 |

### 6.2 Matriz de confusión del baseline

| | Predicción 0 | Predicción 1 |
|---|---:|---:|
| Real 0 | 0 | 1,900 |
| Real 1 | 0 | 8,100 |

El baseline clasifica todos los vehículos como necesitados de mantenimiento, debido al predominio de la clase positiva.

Por ello, alcanza un recall de 100 %, pero no identifica ningún vehículo de la clase negativa.

### 6.3 Matriz de confusión de la regresión logística

| | Predicción 0 | Predicción 1 |
|---|---:|---:|
| Real 0 | 347 | 1,553 |
| Real 1 | 516 | 7,584 |

El modelo de regresión logística registra:

- 7,584 verdaderos positivos.
- 347 verdaderos negativos.
- 1,553 falsos positivos.
- 516 falsos negativos.

## 7. Interpretación de resultados

La regresión logística presenta una precisión superior a la del baseline, pero obtiene valores inferiores de accuracy, recall y F1-score.

Esto indica que, bajo la configuración inicial, el modelo 1 todavía no supera al modelo de referencia en las principales métricas globales evaluadas.

No obstante, la regresión logística logra identificar correctamente algunos vehículos que no requieren mantenimiento, algo que el baseline no consigue.

Desde la perspectiva del DSS, los falsos negativos son especialmente relevantes porque representan vehículos que requieren mantenimiento, pero que el modelo no identifica como tales.

Por tanto, la selección del modelo definitivo no debe depender únicamente de la exactitud, sino también de las consecuencias operativas de cada tipo de error.

## 8. Reproducibilidad

### 8.1 Dependencias

El script requiere Python y las siguientes bibliotecas:

- pandas
- scikit-learn
- joblib

### 8.2 Ejecución

Desde la raíz del repositorio, ejecutar:

    python integrantes/oscarina_brazoban/scripts/entrenar_modelos.py

El script genera los modelos entrenados y los archivos de resultados.

### 8.3 Archivos generados

Dentro de `integrantes/oscarina_brazoban/resultados/` se generan:

- `metricas.json`: resultados de evaluación.
- `indices_train.csv`: índices del conjunto de entrenamiento.
- `indices_test.csv`: índices del conjunto de prueba.
- `baseline.joblib`: modelo de referencia serializado.
- `modelo1_regresion_logistica.joblib`: modelo predictivo serializado.

Los modelos serializados se generan localmente y no se incluyeron en el commit inicial.

Los índices corresponden al orden de los registros del conjunto original utilizado durante el entrenamiento.

## 9. Integración con el proyecto grupal

Esta contribución proporciona:

1. Un modelo de referencia para la comparación de algoritmos.
2. Un primer modelo predictivo basado en regresión logística.
3. Métricas cuantitativas para la evaluación comparativa.
4. Índices reproducibles para entrenamiento y prueba.
5. Código modular para facilitar su integración con los demás componentes.

Para realizar una comparación válida, los modelos desarrollados por otros integrantes deben utilizar el mismo conjunto de datos, las mismas observaciones de prueba y un procedimiento de evaluación equivalente.

Los resultados podrán utilizarse como insumos para el componente de reglas de decisión y la aplicación del DSS.

## 10. Limitaciones y oportunidades de mejora

- Existe desbalance entre las clases de la variable objetivo.
- Se utilizaron seis variables predictoras de las diecinueve disponibles.
- No se realizó optimización de hiperparámetros.
- No se evaluaron diferentes umbrales de clasificación.
- No se aplicó validación cruzada en esta primera versión.
- La evaluación corresponde a una única partición de entrenamiento y prueba.
- No se dispone todavía de validación externa en condiciones reales de operación.
- Se requiere evaluar el costo operativo de los falsos positivos y falsos negativos.

Como mejoras futuras se propone analizar el ajuste de umbrales, incorporar métricas adicionales por clase y comparar los resultados con los demás modelos del proyecto.

## 11. Conclusión

Se implementaron satisfactoriamente un baseline y un primer modelo predictivo para estimar la necesidad de mantenimiento preventivo de vehículos.

Los resultados evidencian la importancia de utilizar modelos de referencia y métricas apropiadas cuando existe desbalance de clases.

Aunque la regresión logística no superó al baseline en accuracy ni F1-score, constituye una alternativa interpretable que permite estudiar las relaciones entre las variables predictoras y apoyar el desarrollo posterior del Sistema de Soporte a Decisiones.

La selección del modelo definitivo deberá fundamentarse en la evaluación comparativa del equipo y en las necesidades operativas del sistema.

---

**Responsable de la contribución:** Oscarina Brazoban  
**Rama de desarrollo:** `feature/oscarina-baseline-modelo1`  
**Commit inicial:** `5c40424`
