# Análisis Exploratorio de Datos (EDA) y Hallazgos

**Responsable:** Kendy Báez Santos (Datos y análisis exploratorio)

## 1. Proceso de Limpieza y Auditoría de Datos
Para generar el entregable `datos/processed/dataset_limpio.csv` a partir de `datos/raw/vehicle_maintenance_raw.csv`, se aplicaron las siguientes reglas mediante el script `src/realizar_eda.py`:
1. **Estandarización de columnas:** Conversión de encabezados a minúsculas en formato `snake_case`.
2. **Eliminación de registros duplicados:** Detección y remoción de filas repetidas mediante `drop_duplicates()`.
3. **Corrección de registros inconsistentes:** Filtrado y eliminación de valores físicamente imposibles (valores negativos en `mileage`, `vehicle_age` y `reported_issues`).
4. **Tratamiento de valores faltantes (nulos):** Imputación estadística utilizando la **mediana** para las variables numéricas (`mileage`, `vehicle_age`, `reported_issues`) y la **moda** para las variables categóricas (`maintenance_history`, `vehicle_model`).

## 2. Análisis de Variables Clave y Hallazgos Principales

### A. Análisis de Fallas Reportadas (`reported_issues`) vs. Mantenimiento (`need_maintenance`)
* **Hallazgo:** Es la variable con mayor correlación positiva respecto a la necesidad de mantenimiento preventivo. Los vehículos con 3 o más fallas reportadas presentan una probabilidad drásticamente superior de requerir intervención inmediata (`need_maintenance = 1`).

### B. Análisis de Kilometraje (`mileage`) y Antigüedad (`vehicle_age`)
* **Hallazgo:** La distribución en los diagramas de caja muestra que los vehículos que requieren mantenimiento concentran medianas de kilometraje más altas. De igual forma, a mayor antigüedad del vehículo se observa una tendencia creciente en el promedio de fallas acumuladas.

### C. Análisis del Historial de Mantenimiento (`maintenance_history`)
* **Hallazgo:** Los vehículos con un historial catalogado como deficiente (`Poor`) o regular (`Average`) presentan mayor incidencia de alertas de mantenimiento en comparación con aquellos con historial bueno (`Good`).

## 3. Gráficos del Análisis Exploratorio
A continuación se presentan las visualizaciones generadas sobre kilometraje, antigüedad, fallas reportadas y matriz de correlación:

[![Gráficos del Análisis Exploratorio](reports/graficos_eda.png)]