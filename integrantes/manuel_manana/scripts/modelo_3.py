### Random Forest


from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# 1. CONFIGURACIÓN DE RUTAS
# ============================================================

BASE = Path(__file__).resolve().parents[3]
RESULTADOS = Path(__file__).resolve().parents[1] / "resultados"

RESULTADOS.mkdir(parents=True, exist_ok=True)

RUTA_DATOS = BASE / "datos/raw/vehicle_maintenance_raw.csv"

RUTA_INDICES = (
    BASE / "integrantes/oscarina_brazoban/resultados"
)


# ============================================================
# 2. CARGA Y PREPARACIÓN DE LOS DATOS
# ============================================================

df = pd.read_csv(RUTA_DATOS)

df.columns = (
    df.columns.str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

numericas = [
    "mileage",
    "vehicle_age",
    "reported_issues",
    "service_history",
]

categoricas = [
    "vehicle_model",
    "maintenance_history",
]

objetivo = "need_maintenance"
variables = numericas + categoricas

# ============================================================
# Aplicar los mismos criterios de selección que Oscarina
# ============================================================
df = df.drop_duplicates().copy()
df = df.dropna(subset=[objetivo])

for col in numericas:
    df = df[df[col].isna() | (df[col] >= 0)]

if not df[objetivo].isin([0, 1]).all():
    raise ValueError("La variable objetivo debe contener solo 0 y 1.")


# ============================================================
# 3. PARTICIÓN DE ENTRENAMIENTO Y PRUEBA
# ============================================================

# Reutilizar exactamente los índices de Oscarina
indices_train = pd.read_csv(
    RUTA_INDICES / "indices_train.csv"
)["indice_original"]

indices_test = pd.read_csv(
    RUTA_INDICES / "indices_test.csv"
)["indice_original"]

# Comprobar integridad de las particiones
assert indices_train.is_unique
assert indices_test.is_unique
assert set(indices_train).isdisjoint(set(indices_test))
assert set(indices_train).issubset(set(df.index))
assert set(indices_test).issubset(set(df.index))

X_train = df.loc[indices_train, variables]
X_test = df.loc[indices_test, variables]

y_train = df.loc[indices_train, objetivo]
y_test = df.loc[indices_test, objetivo]


# ============================================================
# 4. PREPROCESAMIENTO
# ============================================================

preprocesador = ColumnTransformer([
    (
        "numericas",
        SimpleImputer(strategy="median"),
        numericas,
    ),
    (
        "categoricas",
        Pipeline([
            (
                "imputacion",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "codificacion",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]),
        categoricas,
    ),
])


# ============================================================
# 5. MODELO 3: RANDOM FOREST
# ============================================================

modelo = Pipeline([
    (
        "preprocesamiento",
        preprocesador,
    ),
    (
        "clasificador",
        RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_leaf=20,
            random_state=42,
            n_jobs=-1,
        ),
    ),
])


# ============================================================
# 6. ENTRENAMIENTO
# ============================================================

print("Entrenando Modelo 3 - Random Forest...")

modelo.fit(X_train, y_train)


# ============================================================
# 7. PREDICCIONES Y PROBABILIDADES
# ============================================================

predicciones = modelo.predict(X_test)

indice_clase_positiva = list(
    modelo.named_steps["clasificador"].classes_
).index(1)

probabilidades = modelo.predict_proba(X_test)[
    :, indice_clase_positiva
]


# ============================================================
# 8. EVALUACIÓN
# ============================================================

metricas = {
    "accuracy": float(
        accuracy_score(y_test, predicciones)
    ),
    "precision": float(
        precision_score(
            y_test, predicciones, zero_division=0
        )
    ),
    "recall": float(
        recall_score(
            y_test, predicciones, zero_division=0
        )
    ),
    "f1": float(
        f1_score(
            y_test, predicciones, zero_division=0
        )
    ),
    "matriz_confusion": confusion_matrix(
        y_test,
        predicciones,
        labels=[0, 1],
    ).tolist(),
}


# ============================================================
# 9. GUARDAR MÉTRICAS
# ============================================================

with open(
    RESULTADOS / "metricas_modelo_3.json",
    "w",
    encoding="utf-8",
) as archivo:
    json.dump(metricas, archivo, indent=2)


# ============================================================
# 10. GUARDAR PREDICCIONES
# ============================================================

df_predicciones = pd.DataFrame({
    "indice_original": indices_test.to_numpy(),
    "y_real": y_test.to_numpy(),
    "prediccion": predicciones,
    "probabilidad_mantenimiento": probabilidades,
})

df_predicciones.to_csv(
    RESULTADOS / "predicciones_modelo_3.csv",
    index=False,
)


# ============================================================
# 11. GUARDAR MODELO ENTRENADO
# ============================================================

joblib.dump(
    modelo,
    RESULTADOS / "modelo_3_random_forest.joblib",
)


# ============================================================
# 12. MOSTRAR RESULTADOS
# ============================================================

print("\nMODELO 3 - RANDOM FOREST")
print(f"Entrenamiento: {len(X_train)} registros")
print(f"Prueba: {len(X_test)} registros")

print("\nMÉTRICAS:")
print(json.dumps(metricas, indent=2))

print("\nDISTRIBUCIÓN DE PREDICCIONES:")
print(df_predicciones["prediccion"].value_counts())

print("\nResultados guardados correctamente.")
print(f"Ruta: {RESULTADOS}")
