
from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st


# ============================================================
# RUTAS DEL PROYECTO
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "datos" / "raw" / "vehicle_maintenance_raw.csv"

MODEL_PATHS = {
    "Baseline": ROOT / "integrantes/oscarina_brazoban/resultados/baseline.joblib",
    "Regresión logística": ROOT / "integrantes/oscarina_brazoban/resultados/modelo1_regresion_logistica.joblib",
    "Árbol de decisión": ROOT / "integrantes/manuel_manana/resultados/modelo_2_arbol_decision.joblib",
    "Random Forest": ROOT / "integrantes/manuel_manana/resultados/modelo_3_random_forest.joblib",
}

METRICS_PATHS = {
    "oscarina": ROOT / "integrantes/oscarina_brazoban/resultados/metricas.json",
    "modelo_2": ROOT / "integrantes/manuel_manana/resultados/metricas_modelo_2.json",
    "modelo_3": ROOT / "integrantes/manuel_manana/resultados/metricas_modelo_3.json",
}

FEATURES = [
    "vehicle_model",
    "mileage",
    "vehicle_age",
    "reported_issues",
    "maintenance_history",
    "service_history",
]


# ============================================================
# CONFIGURACIÓN DE STREAMLIT
# ============================================================

st.set_page_config(
    page_title="Mantenimiento Preventivo de Vehículos",
    page_icon="🚗",
    layout="wide",
)


# ============================================================
# CARGA DE DATOS
# ============================================================

@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el conjunto de datos: {DATA_PATH}"
        )

    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_models():
    models = {}

    for name, path in MODEL_PATHS.items():
        if not path.exists():
            raise FileNotFoundError(
                f"No se encontró el modelo {name}: {path}"
            )

        models[name] = joblib.load(path)

    return models


# ============================================================
# CARGA DE MÉTRICAS
# ============================================================

@st.cache_data
def load_metrics():

    for name, path in METRICS_PATHS.items():
        if not path.exists():
            raise FileNotFoundError(
                f"No se encontró el archivo de métricas: {path}"
            )

    with open(METRICS_PATHS["oscarina"], "r", encoding="utf-8") as file:
        metrics_oscarina = json.load(file)

    with open(METRICS_PATHS["modelo_2"], "r", encoding="utf-8") as file:
        metrics_model_2 = json.load(file)

    with open(METRICS_PATHS["modelo_3"], "r", encoding="utf-8") as file:
        metrics_model_3 = json.load(file)

    metrics_data = [
        [
            "Baseline",
            metrics_oscarina["baseline"]["accuracy"],
            metrics_oscarina["baseline"]["precision"],
            metrics_oscarina["baseline"]["recall"],
            metrics_oscarina["baseline"]["f1"],
        ],
        [
            "Regresión logística",
            metrics_oscarina["modelo1_regresion_logistica"]["accuracy"],
            metrics_oscarina["modelo1_regresion_logistica"]["precision"],
            metrics_oscarina["modelo1_regresion_logistica"]["recall"],
            metrics_oscarina["modelo1_regresion_logistica"]["f1"],
        ],
        [
            "Árbol de decisión",
            metrics_model_2["accuracy"],
            metrics_model_2["precision"],
            metrics_model_2["recall"],
            metrics_model_2["f1"],
        ],
        [
            "Random Forest",
            metrics_model_3["accuracy"],
            metrics_model_3["precision"],
            metrics_model_3["recall"],
            metrics_model_3["f1"],
        ],
    ]

    return pd.DataFrame(
        metrics_data,
        columns=[
            "Modelo",
            "Accuracy",
            "Precision",
            "Recall",
            "F1-score",
        ],
    )


# ============================================================
# REGLAS DEL SISTEMA DE APOYO A DECISIONES
# ============================================================

def classify_risk(probability, reported_issues, mileage, vehicle_age):

    if probability >= 0.90 or reported_issues >= 4:
        return (
            "CRÍTICO",
            "Prioridad máxima. Atención inmediata.",
            "Revisar el vehículo de inmediato y documentar la intervención.",
        )

    if probability >= 0.75 or mileage >= 150000:
        return (
            "ALTO",
            "Prioridad alta. Mantenimiento en un máximo de 7 días.",
            "Programar una revisión mecánica prioritaria.",
        )

    if probability >= 0.50 or vehicle_age >= 8:
        return (
            "MEDIO",
            "Prioridad media. Requiere inspección preventiva.",
            "Programar inspección y dar seguimiento al estado del vehículo.",
        )

    return (
        "BAJO",
        "Prioridad baja. Seguimiento periódico.",
        "Mantener el calendario habitual de mantenimiento.",
    )


# ============================================================
# CÁLCULO DE PROBABILIDAD
# ============================================================

def get_probability(model, data):

    probabilities = model.predict_proba(data)[0]
    classes = list(model.classes_)

    if 1 not in classes:
        return 0.0

    positive_index = classes.index(1)

    return float(probabilities[positive_index])


# ============================================================
# APLICACIÓN PRINCIPAL
# ============================================================

def main():

    st.title("🚗 Sistema de Apoyo a Decisiones")
    st.subheader("Mantenimiento Preventivo de Vehículos")

    st.write(
        "Herramienta de apoyo para estimar la necesidad de mantenimiento, "
        "comparar modelos y orientar la priorización de los vehículos."
    )

    # --------------------------------------------------------
    # CARGA DE DATOS Y MODELOS
    # --------------------------------------------------------

    try:
        dataset = load_data()
        models = load_models()

    except Exception as error:
        st.error(f"No fue posible cargar los datos o modelos: {error}")
        st.info(
            "Verifica que existan los datos y que se hayan ejecutado "
            "los scripts de entrenamiento de Oscarina y Manuel."
        )
        return

    vehicle_models = sorted(
        dataset["Vehicle_Model"].dropna().astype(str).unique()
    )

    maintenance_histories = sorted(
        dataset["Maintenance_History"].dropna().astype(str).unique()
    )

    # --------------------------------------------------------
    # 1. DATOS DEL VEHÍCULO
    # --------------------------------------------------------

    st.divider()
    st.header("1. Datos del vehículo")

    with st.form("vehicle_form"):

        col1, col2, col3 = st.columns(3)

        with col1:
            vehicle_model = st.selectbox(
                "Tipo de vehículo",
                vehicle_models,
            )

            mileage = st.number_input(
                "Kilometraje",
                min_value=30001,
                max_value=80000,
                value=55000,
                step=100,
            )

        with col2:
            vehicle_age = st.number_input(
                "Antigüedad (años)",
                min_value=1,
                max_value=10,
                value=5,
                step=1,
            )

            reported_issues = st.number_input(
                "Problemas mecánicos reportados",
                min_value=0,
                max_value=5,
                value=2,
                step=1,
            )

        with col3:
            maintenance_history = st.selectbox(
                "Historial de mantenimiento",
                maintenance_histories,
            )

            service_history = st.number_input(
                "Cantidad de servicios realizados",
                min_value=1,
                max_value=10,
                value=5,
                step=1,
            )

        submitted = st.form_submit_button(
            "Evaluar vehículo",
            type="primary",
            use_container_width=True,
        )

    if not submitted:
        st.info(
            "Introduce los datos del vehículo y pulsa "
            "«Evaluar vehículo» para consultar los resultados."
        )
        return

    # --------------------------------------------------------
    # PREPARACIÓN DE LOS DATOS
    # --------------------------------------------------------

    vehicle_data = pd.DataFrame(
        [[
            vehicle_model,
            mileage,
            vehicle_age,
            reported_issues,
            maintenance_history,
            service_history,
        ]],
        columns=FEATURES,
    )

    # --------------------------------------------------------
    # 2. COMPARACIÓN DE LOS MODELOS
    # --------------------------------------------------------

    st.divider()
    st.header("2. Comparación de los modelos")

    results = {}

    try:
        for name, model in models.items():

            prediction = int(model.predict(vehicle_data)[0])

            probability = get_probability(
                model,
                vehicle_data,
            )

            results[name] = {
                "Predicción": (
                    "Requiere mantenimiento"
                    if prediction == 1
                    else "No requiere mantenimiento"
                ),
                "Probabilidad (%)": round(probability * 100, 2),
                "Clase": prediction,
            }

        comparison = pd.DataFrame(results).T
        comparison.index.name = "Modelo"

        st.table(comparison)

        st.caption(
            "Las probabilidades son estimaciones de cada modelo; "
            "no representan una confirmación mecánica."
        )

    except Exception as error:
        st.error(f"No fue posible comparar los modelos: {error}")
        return

    # --------------------------------------------------------
    # 3. EVALUACIÓN DEL RIESGO MEDIANTE EL DSS
    # --------------------------------------------------------

    st.divider()
    st.header("3. Evaluación del riesgo mediante el DSS")

    selected_model = st.selectbox(
        "Modelo utilizado para las reglas de decisión",
        list(models.keys()),
        index=list(models.keys()).index("Random Forest"),
    )

    selected_probability = (
        results[selected_model]["Probabilidad (%)"] / 100
    )

    risk, priority, recommendation = classify_risk(
        selected_probability,
        reported_issues,
        mileage,
        vehicle_age,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Modelo seleccionado",
            selected_model,
        )

        st.metric(
            "Probabilidad estimada",
            f"{selected_probability * 100:.2f}%",
        )

    with col2:
        st.metric(
            "Nivel de riesgo",
            risk,
        )

        st.metric(
            "Prioridad",
            priority,
        )

    st.subheader("Recomendación del sistema")
    st.success(recommendation)

    st.write("**Criterios considerados:**")
    st.write(f"- Probabilidad estimada: {selected_probability * 100:.2f}%")
    st.write(f"- Problemas reportados: {reported_issues}")
    st.write(f"- Kilometraje: {mileage:,} km")
    st.write(f"- Antigüedad: {vehicle_age} años")

    st.warning(
        "La regla de kilometraje alto está definida en 150,000 km, "
        "pero el conjunto de entrenamiento utilizado llega hasta "
        "80,000 km. Por tanto, esa condición no se activa con los "
        "valores admitidos en este formulario."
    )

    # --------------------------------------------------------
    # 4. MÉTRICAS DE EVALUACIÓN DE LOS MODELOS
    # --------------------------------------------------------

    st.divider()
    st.header("4. Métricas de evaluación de los modelos")

    try:
        metrics = load_metrics()

        st.table(
            metrics.set_index("Modelo").round(4)
        )

        st.caption(
            "Métricas cargadas directamente de los resultados "
            "de entrenamiento documentados por los integrantes "
            "del equipo."
        )

    except Exception as error:
        st.error(
            f"No fue posible cargar las métricas de evaluación: {error}"
        )

    # --------------------------------------------------------
    # 5. CONSIDERACIONES
    # --------------------------------------------------------

    st.divider()
    st.header("5. Consideraciones")

    st.write(
        "- El sistema funciona como apoyo a la decisión y no sustituye "
        "la inspección de un técnico."
    )

    st.write(
        "- La clasificación depende del modelo seleccionado y de las "
        "reglas definidas para el DSS."
    )

    st.write(
        "- Los resultados deben interpretarse considerando los falsos "
        "positivos, falsos negativos y las limitaciones de los datos."
    )

    st.write(
        "- Las recomendaciones deben ser revisadas por personal "
        "responsable antes de tomar decisiones sobre los vehículos."
    )

    st.caption(
        "Proyecto académico INF-8239 | Sistema de Apoyo a Decisiones "
        "para el Mantenimiento Preventivo de Vehículos"
    )


# ============================================================
# EJECUCIÓN
# ============================================================

if __name__ == "__main__":
    main()


