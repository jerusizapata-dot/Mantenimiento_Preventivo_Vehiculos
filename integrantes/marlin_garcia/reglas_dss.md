# Reglas y Lógica de Decisión del DSS

## Objetivo

El Sistema de Soporte a Decisiones (DSS) transforma las predicciones generadas por los modelos de Machine Learning en acciones concretas para la gestión de mantenimiento vehicular.

El objetivo es priorizar los vehículos con mayor riesgo de requerir mantenimiento, optimizando la asignación de recursos y reduciendo el riesgo operacional.

---

# Variables utilizadas

Las reglas de decisión consideran las siguientes variables:

- mileage
- vehicle_age
- reported_issues
- service_history
- probabilidad_mantenimiento

La probabilidad_mantenimiento corresponde a la probabilidad estimada por los modelos predictivos desarrollados durante el proyecto.

---

# Niveles de Riesgo

## Riesgo Crítico

Condiciones:

- probabilidad_mantenimiento >= 0.90
- o reported_issues >= 4

Acción:

- Atención inmediata.
- Prioridad máxima.

---

## Riesgo Alto

Condiciones:

- probabilidad_mantenimiento >= 0.75
- o mileage > 150000

Acción:

- Programar mantenimiento en un máximo de 7 días.

---

## Riesgo Medio

Condiciones:

- probabilidad_mantenimiento >= 0.50
- o vehicle_age > 8 años

Acción:

- Inspección preventiva programada.

---

## Riesgo Bajo

Condiciones:

- probabilidad_mantenimiento < 0.50

Acción:

- Monitoreo periódico.

---

# Justificación

Los modelos desarrollados durante el proyecto presentan valores elevados de recall, lo que permite detectar la mayoría de los vehículos que requieren mantenimiento.

La utilización de niveles de riesgo permite transformar la predicción del modelo en acciones comprensibles para los responsables operativos.

---

# Matriz de Prioridades

| Nivel de Riesgo | Prioridad | Acción |
|-----------------|------------|----------|
| Crítico | 1 | Atención inmediata |
| Alto | 2 | Mantenimiento en menos de 7 días |
| Medio | 3 | Inspección preventiva |
| Bajo | 4 | Monitoreo periódico |