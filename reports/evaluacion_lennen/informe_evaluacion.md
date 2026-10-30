# Informe de evaluación de modelos

## Objetivo

Comparar el desempeño de cuatro modelos para identificar vehículos que requieren mantenimiento preventivo.

## Resultados comparativos

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC | FN | FP |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 0.8100 | 0.8100 | 1.0000 | 0.8950 | 0.5000 | 0 | 1900 |
| Regresión logística | 0.7931 | 0.8300 | 0.9363 | 0.8800 | 0.8135 | 516 | 1553 |
| Árbol de decisión | 0.8096 | 0.8109 | 0.9977 | 0.8946 | 0.8469 | 19 | 1885 |
| Random Forest | 0.8098 | 0.8124 | 0.9949 | 0.8945 | 0.8458 | 41 | 1861 |

## Análisis de errores

Los falsos negativos representan vehículos que necesitan mantenimiento, pero que el modelo no identifica. Este error es especialmente importante porque puede retrasar una intervención preventiva.

Los falsos positivos representan vehículos señalados para mantenimiento que no lo necesitan según la clasificación real. Pueden generar revisiones o gastos innecesarios.

## Consideraciones y limitaciones

- Las métricas corresponden al conjunto de prueba utilizado durante el entrenamiento.
- El recall es especialmente importante para evaluar la detección de vehículos que necesitan mantenimiento.
- Los resultados apoyan la toma de decisiones, pero no sustituyen la revisión técnica humana.
- La calidad de las predicciones depende de la calidad y representatividad de los datos.

## Archivos generados

- comparacion_modelos.csv
- comparacion_modelos.json
- predicciones_modelos.csv
- analisis_errores.csv