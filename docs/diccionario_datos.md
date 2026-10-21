# Documentación del Conjunto de Datos y Diccionario de Datos

**Responsable:** Kendy Báez Santos (Datos y análisis exploratorio)

## 1. Origen y Mecanismo de Obtención del Conjunto de Datos
* **Nombre del Dataset:** Vehicle Maintenance Data (Conjunto de datos público de mantenimiento vehicular).
* **Fuente:** Kaggle (`chavindudulaj/vehicle-maintenance-data`).
* **Mecanismo de obtención reproducible:** Ejecutar desde la raíz del repositorio el comando `py src/descargar_datos.py`, el cual descarga y almacena automáticamente los datos crudos en `datos/raw/vehicle_maintenance_raw.csv`.
* **Ruta del dataset limpio:** Tras ejecutar `py src/realizar_eda.py`, se genera el archivo procesado en `datos/processed/dataset_limpio.csv` listo para el entrenamiento de modelos y las reglas del DSS.

## 2. Diccionario de Datos

| Nombre de la Variable | Tipo de Dato | Descripción | Valores / Rango Permitido | Rol en el Proyecto (DSS) |
| :--- | :--- | :--- | :--- | :--- |
| `vehicle_model` | Categórica (`object`) | Tipo o categoría del vehículo evaluado. | `Car`, `SUV`, `Van`, `Truck`, `Bus` | Segmentación del parque vehicular |
| `mileage` | Numérica (`float64`) | Kilometraje total acumulado recorrido por el vehículo. | Valores numéricos `>= 0` | Variable predictora (desgaste por uso) |
| `vehicle_age` | Numérica (`float64`) | Antigüedad del vehículo desde su año de fabricación. | Años enteros `>= 0` | Variable predictora (desgaste por tiempo) |
| `reported_issues` | Numérica (`float64`) | Cantidad de fallas o problemas mecánicos reportados previamente. | Conteo entero (`0` a `5+`) | Variable predictora y criterio de riesgo |
| `maintenance_history` | Categórica (`object`) | Clasificación de la calidad del historial de mantenimiento previo. | `Good`, `Average`, `Poor` | Variable predictora y regla de priorización |
| `service_history` | Numérica (`int64`) | Cantidad de servicios o mantenimientos preventivos realizados al vehículo. | Conteo entero `>= 1` | Variable predictora de cumplimiento |
| `need_maintenance` | Binaria (`int64`) | Indica si el vehículo requiere mantenimiento preventivo inmediato. | `1` (Requiere), `0` (No requiere) | **Variable Objetivo (Target)** para los modelos |

## 3. Justificación de la Idoneidad del Dataset para el DSS

La selección de este conjunto de datos responde a los criterios técnicos y de negocio de la Modalidad B (Sistema de Soporte a Decisiones):

1. **Pertinencia de Variables:** Incluye de manera explícita las variables críticas del dominio vehicular: kilometraje (`mileage`), antigüedad (`vehicle_age`), historial de fallas (`reported_issues`) y registros de mantenimiento (`maintenance_history`, `service_history`, `need_maintenance`).

2. **Soporte para Modelos Predictivos y Baseline:** Cuenta con una variable objetivo binaria definida (`need_maintenance`) que permite entrenar, evaluar y comparar el baseline y los 3 modelos candidatos requeridos para el equipo.

3. **Capacidad Explicativa y Reglas de Priorización:** A diferencia de datasets con variables anonimizadas, sus atributos tienen significado físico directo para un decisor humano (por ejemplo, un jefe de flota o taller). Esto permite combinar la probabilidad predicha por los modelos con reglas explícitas de negocio (en `reglas_decision/`) para clasificar niveles de riesgo (Alto, Medio, Bajo) y priorizar qué unidades deben ingresar primero a mantenimiento preventivo.

4. **Trazabilidad y Reproducibilidad:** Permite auditar el flujo completo de ingeniería de datos (eliminación de duplicados, corrección de inconsistencias negativas e imputación de valores faltantes) desde `datos/raw/` hacia `datos/processed/`, asegurando un pipeline reproducible.