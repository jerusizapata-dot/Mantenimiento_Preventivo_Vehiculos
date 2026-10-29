# Aspectos Éticos, Limitaciones y Green AI

## Riesgos Éticos

### Falsos Positivos

Un vehículo puede ser clasificado como prioritario sin requerir realmente mantenimiento inmediato.

Esto puede generar utilización innecesaria de recursos.

### Falsos Negativos

Aunque son reducidos, algunos vehículos pueden no ser detectados por el modelo.

Por esta razón, las recomendaciones del DSS no sustituyen las inspecciones técnicas.

### Supervisión Humana

Las decisiones finales deben ser validadas por personal especializado.

El DSS funciona como herramienta de apoyo y no como reemplazo del criterio profesional.

---

# Limitaciones

- El sistema depende de la calidad de los datos históricos.
- Nuevos patrones de fallas podrían reducir el desempeño futuro del modelo.
- El comportamiento de los vehículos puede cambiar con el tiempo.

---

# Consideraciones de Green AI

## Reutilización de particiones

Las mismas particiones de entrenamiento y prueba son reutilizadas entre modelos para evitar procesamiento redundante.

## Reutilización de modelos

Los modelos entrenados son almacenados mediante archivos joblib para evitar entrenamientos innecesarios.

## Modelos de complejidad moderada

El proyecto utiliza modelos con costos computacionales razonables:

- Regresión Logística
- Árbol de Decisión
- Random Forest

Esto reduce el consumo de recursos computacionales manteniendo un desempeño adecuado.

---

# Conclusión

La solución propuesta busca equilibrar precisión, interpretabilidad, sostenibilidad computacional y apoyo efectivo a la toma de decisiones.