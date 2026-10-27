from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE = Path(__file__).resolve().parents[3]
SALIDA = Path(__file__).resolve().parents[1] / "resultados"
SALIDA.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(BASE / "datos/raw/vehicle_maintenance_raw.csv")
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

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

columnas = numericas + categoricas

df = df.drop_duplicates().copy()
df = df.dropna(subset=[objetivo])

for col in numericas:
    df = df[df[col].isna() | (df[col] >= 0)]

if not df[objetivo].isin([0, 1]).all():
    raise ValueError("La variable objetivo debe contener solo 0 y 1.")

X = df[columnas]
y = df[objetivo]

indices_train, indices_test = train_test_split(
    df.index,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

X_train = X.loc[indices_train]
X_test = X.loc[indices_test]
y_train = y.loc[indices_train]
y_test = y.loc[indices_test]

pd.DataFrame({
    "indice_original": indices_train,
    "particion": "train",
}).to_csv(SALIDA / "indices_train.csv", index=False)

pd.DataFrame({
    "indice_original": indices_test,
    "particion": "test",
}).to_csv(SALIDA / "indices_test.csv", index=False)

preprocesador = ColumnTransformer([
    ("numericas", Pipeline([
        ("imputacion", SimpleImputer(strategy="median")),
        ("escalado", StandardScaler()),
    ]), numericas),
    ("categoricas", Pipeline([
        ("imputacion", SimpleImputer(strategy="most_frequent")),
        ("codificacion", OneHotEncoder(handle_unknown="ignore")),
    ]), categoricas),
])

modelos = {
    "baseline": DummyClassifier(strategy="most_frequent"),
    "modelo1_regresion_logistica": Pipeline([
        ("preprocesamiento", preprocesador),
        ("clasificador", LogisticRegression(
            max_iter=1000,
            random_state=42,
        )),
    ]),
}

resultados = {}

for nombre, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    predicciones = modelo.predict(X_test)

    metricas = {
        "accuracy": float(accuracy_score(y_test, predicciones)),
        "precision": float(precision_score(y_test, predicciones, zero_division=0)),
        "recall": float(recall_score(y_test, predicciones, zero_division=0)),
        "f1": float(f1_score(y_test, predicciones, zero_division=0)),
        "matriz_confusion": confusion_matrix(
            y_test, predicciones, labels=[0, 1]
        ).tolist(),
    }

    resultados[nombre] = metricas
    joblib.dump(modelo, SALIDA / f"{nombre}.joblib")

    print(f"\n{nombre.upper()}")
    print(json.dumps(metricas, indent=2))

with open(SALIDA / "metricas.json", "w", encoding="utf-8") as archivo:
    json.dump(resultados, archivo, indent=2)

print("\nEntrenamiento finalizado.")
print(f"Entrenamiento: {len(X_train)} registros")
print(f"Prueba: {len(X_test)} registros")
print(f"Resultados guardados en: {SALIDA}")
