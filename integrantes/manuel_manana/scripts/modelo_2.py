### Modelo 2: Árbol de Decisión
### Decision Tree Classifier

from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
# ============================================================
# 1. Rutas del proyecto
# ============================================================
BASE = Path(__file__).resolve().parents[3]
RESULTADOS = Path(__file__).resolve().parents[1] / "resultados"
RESULTADOS.mkdir(parents=True, exist_ok=True)
# ============================================================
# 2. Cargar los mismos datos que Oscarina
# ============================================================
df = pd.read_csv(
    BASE / "datos/raw/vehicle_maintenance_raw.csv"
)
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
# 3. Reutilizar la partición de Oscarina
# ============================================================
ruta_indices = (
    BASE / "integrantes/oscarina_brazoban/resultados"
)

indices_train = pd.read_csv(
    ruta_indices / "indices_train.csv"
)["indice_original"]

indices_test = pd.read_csv(
    ruta_indices / "indices_test.csv"
)["indice_original"]

X_train = df.loc[indices_train, variables]
X_test = df.loc[indices_test, variables]
y_train = df.loc[indices_train, objetivo]
y_test = df.loc[indices_test, objetivo]

# ============================================================
# 4. Preprocesamiento aprendido solo desde train
# ============================================================
preprocesador = ColumnTransformer([
    ("numericas", SimpleImputer(strategy="median"), numericas),
    ("categoricas", Pipeline([
        ("imputacion", SimpleImputer(strategy="most_frequent")),
        ("codificacion", OneHotEncoder(handle_unknown="ignore")),
    ]), categoricas),
])

# ============================================================
# 5. Modelo 2: Árbol de Decisión
# ============================================================
modelo = Pipeline([
    ("preprocesamiento", preprocesador),
    ("clasificador", DecisionTreeClassifier(
        max_depth=6,
        min_samples_leaf=30,
        random_state=42,
    )),
])

# ============================================================
# 6. Entrenar el modelo
# ============================================================
modelo.fit(X_train, y_train)

# ============================================================
# 7. Generar predicciones y probabilidades

predicciones = modelo.predict(X_test)
probabilidades = modelo.predict_proba(X_test)[:, 1]

# ============================================================
# 8. Evaluar desempeño
# ============================================================
metricas = {
    "accuracy": float(accuracy_score(y_test, predicciones)),
    "precision": float(precision_score(
        y_test, predicciones, zero_division=0
    )),
    "recall": float(recall_score(
        y_test, predicciones, zero_division=0
    )),
    "f1": float(f1_score(
        y_test, predicciones, zero_division=0
    )),
    "matriz_confusion": confusion_matrix(
        y_test, predicciones, labels=[0, 1]
    ).tolist(),
}

# ============================================================
# 9. Guardar resultados
# ============================================================
with open(
    RESULTADOS / "metricas_modelo_2.json",
    "w",
    encoding="utf-8",
) as archivo:
    json.dump(metricas, archivo, indent=2)

pd.DataFrame({
    "indice_original": indices_test.to_numpy(),
    "y_real": y_test.to_numpy(),
    "prediccion": predicciones,
    "probabilidad_mantenimiento": probabilidades,
}).to_csv(
    RESULTADOS / "predicciones_modelo_2.csv",
    index=False,
)

joblib.dump(
    modelo,
    RESULTADOS / "modelo_2_arbol_decision.joblib",
)

print("MODELO 2 - ÁRBOL DE DECISIÓN")
print(f"Entrenamiento: {len(X_train)} registros")
print(f"Prueba: {len(X_test)} registros")
print(json.dumps(metricas, indent=2))
print("Resultados guardados correctamente.")
