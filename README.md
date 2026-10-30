# 🚗 SISTEMA DE APOYO A DECISIONES PARA EL MANTENIMIENTO PREVENTIVO DE VEHÍCULOS

### Proyecto académico | Grupo Claude Shaou

---

## 1. Presentación del equipo

**Grupo: Claude Shaou**

**Integrantes:**

1. Laudys Jerusi Zapata
2. Kendry Baez
3. Oscarina Brazoban
4. Manuel Mañana
5. Lennen Santos
6. Marlin Garcia

**Asignatura:** INF-8239 – Ciencia de Datos II  
**Docente:** Edwin Ramón José Nolasco  
**Programa:** Maestría en Ciencia de Datos e Inteligencia Artificial

---

## 2. Presentación del proyecto

El grupo Claude Shaou presenta el proyecto *Sistema de Apoyo a Decisiones para el Mantenimiento Preventivo de Vehículos*, una propuesta tecnológica orientada a integrar la ciencia de datos, el aprendizaje automático y las reglas de decisión en un entorno de apoyo a la gestión del mantenimiento vehicular.

La iniciativa busca transformar los datos en información útil para identificar posibles necesidades de mantenimiento, evaluar condiciones de los vehículos y establecer criterios de priorización que contribuyan a una toma de decisiones más informada, eficiente y responsable.

## 3. Objetivo general

Desarrollar un Sistema de Apoyo a Decisiones (DSS) que permita estimar la necesidad de mantenimiento preventivo de los vehículos mediante modelos predictivos y reglas de decisión, facilitando su evaluación y priorización.

## 4. Objetivos específicos

- Analizar y preparar los datos relacionados con el mantenimiento vehicular.
- Desarrollar y comparar diferentes modelos de aprendizaje automático.
- Evaluar el desempeño predictivo mediante métricas estadísticas.
- Identificar falsos positivos y falsos negativos.
- Implementar reglas de decisión para clasificar y priorizar los vehículos.
- Desarrollar una aplicación interactiva que facilite la interpretación de los resultados.
- Incorporar criterios de transparencia, responsabilidad y supervisión humana.

## 5. Modelos de aprendizaje automático

El sistema integra cuatro alternativas:

| Modelo | Descripción |
|---|---|
| Baseline | Modelo de referencia |
| Regresión logística | Clasificación mediante regresión logística |
| Árbol de decisión | Clasificación mediante reglas jerárquicas |
| Random Forest | Conjunto de árboles de decisión |

La comparación contempla Accuracy, Precision, Recall, F1-score y ROC-AUC, además del análisis de errores de clasificación.

## 6. Tecnologías utilizadas

- Python
- Pandas y NumPy
- Scikit-learn
- Joblib
- Streamlit
- Pytest
- UV
- Git y GitHub

## 7. Estructura del repositorio

```text
Mantenimiento_Preventivo_Vehiculos/
│
├── aplicacion/
├── datos/
├── integrantes/
├── modelos/
├── reglas_decision/
├── reports/
├── src/
├── tests/
├── docs/
├── evaluacion/
├── 01_documentacion/
├── 02_datos/
├── 03_analisis/
├── 04_modelo_sistema/
├── 05_informe_presentacion/
├── pyproject.toml
├── uv.lock
└── README.md
```

## 8. Instalación y configuración

Clonar el repositorio:

```bash
git clone https://github.com/jerusizapata-dot/Mantenimiento_Preventivo_Vehiculos.git
```

Ingresar al directorio:

```bash
cd Mantenimiento_Preventivo_Vehiculos
```

Instalar las dependencias:

```bash
uv sync
```

## 9. Ejecución de la aplicación

Iniciar el sistema mediante Streamlit:

```bash
uv run streamlit run aplicacion/app.py
```

La aplicación permite introducir los datos del vehículo, evaluar su condición, consultar la comparación de modelos y obtener recomendaciones para apoyar la priorización del mantenimiento.

## 10. Pruebas y validación

Para ejecutar las pruebas automatizadas:

```bash
uv run pytest -q
```

En la última ejecución verificada se obtuvieron **29 pruebas aprobadas**.

La evaluación comparativa de los modelos genera los siguientes archivos:

`reports/evaluacion_lennen/`

- `comparacion_modelos.csv`
- `comparacion_modelos.json`
- `predicciones_modelos.csv`
- `analisis_errores.csv`
- `informe_evaluacion.md`

## 11. Gobernanza y uso responsable

El sistema se concibe como una herramienta de apoyo, no como un sustituto del criterio técnico humano.

Su implementación considera:

- Transparencia en los resultados.
- Calidad y confiabilidad de los datos.
- Identificación de errores predictivos.
- Supervisión humana en las decisiones.
- Uso responsable de los recursos computacionales.
- Consideración de las limitaciones de los modelos.

Las recomendaciones generadas deben ser interpretadas por personal responsable antes de ejecutar cualquier intervención de mantenimiento.

## 12. Alcance y valor del proyecto

Este proyecto representa una aplicación práctica de los conocimientos de ciencia de datos e inteligencia artificial, al integrar preparación de datos, modelado predictivo, evaluación estadística, reglas de decisión y desarrollo de una interfaz interactiva.

La propuesta busca demostrar cómo estas tecnologías pueden contribuir a una gestión vehicular más organizada, preventiva y fundamentada en datos.

---

**CLAUDE SHAOU**  
*Datos que se transforman en decisiones.*

**INF-8239 | Ciencia de Datos II**  
**Maestría en Ciencia de Datos e Inteligencia Artificial**