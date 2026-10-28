
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)


# 1. Rutas y datos
BASE = Path(__file__).resolve().parents[3]

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

variables = numericas + categoricas
objetivo = "need_maintenance"

# Misma preparación de registros que Oscarina
df = df.drop_duplicates().copy()
df = df.dropna(subset=[objetivo])

for col in numericas:
    df = df[df[col].isna() | (df[col] >= 0)]

if not df[objetivo].isin([0, 1]).all():
    raise ValueError("La variable objetivo debe ser binaria.")


# 2. Utilizar solamente los índices de entrenamiento
ruta_indices = (
    BASE / "integrantes/oscarina_brazoban"
    / "resultados/indices_train.csv"
)

indices = pd.read_csv(ruta_indices)["indice_original"]

assert len(indices) == 40000
assert indices.is_unique
assert set(indices).issubset(set(df.index))

X = df.loc[indices, variables]
y = df.loc[indices, objetivo]

# Subdivisión interna 80/20 del entrenamiento
X_desarrollo, X_validacion, y_desarrollo, y_validacion = (
    train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42,
    )
)


# 3. Preprocesamiento
def crear_preprocesador():
    return ColumnTransformer([
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
                    SimpleImputer(
                        strategy="most_frequent"
                    ),
                ),
                (
                    "codificacion",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    ),
                ),
            ]),
            categoricas,
        ),
    ])


# 4. Modelos con parámetros iniciales
modelos = {
    "Modelo 2 - Arbol de Decision": DecisionTreeClassifier(
        max_depth=6,
        min_samples_leaf=30,
        random_state=42,
    ),
    "Modelo 3 - Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1,
    ),
}


# 5. Evaluar un umbral
def calcular_indicadores(y_real, probabilidades, umbral):
    predicciones = (probabilidades >= umbral).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_real, predicciones, labels=[0, 1]
    ).ravel()

    recall = tp / (tp + fn)
    especificidad = tn / (tn + fp)

    return {
        "umbral": float(umbral),
        "recall": float(recall),
        "especificidad": float(especificidad),
        "falsos_positivos": int(fp),
        "falsos_negativos": int(fn),
        "vehiculos_priorizados": int(predicciones.sum()),
    }


# 6. Entrenar y estudiar los umbrales
RECALL_MINIMO = 0.90

for nombre, clasificador in modelos.items():

    print(f"\n{'=' * 60}")
    print(nombre)
    print("=" * 60)

    modelo = Pipeline([
        ("preprocesamiento", crear_preprocesador()),
        ("clasificador", clasificador),
    ])

    modelo.fit(X_desarrollo, y_desarrollo)

    probabilidades = modelo.predict_proba(
        X_validacion
    )[:, 1]

    print(
        "ROC-AUC:",
        round(roc_auc_score(y_validacion, probabilidades), 4)
    )
    print(
        "PR-AUC:",
        round(
            average_precision_score(
                y_validacion, probabilidades
            ),
            4,
        ),
    )

    resultados = []

    for umbral in np.linspace(0.30, 0.95, 66):
        indicadores = calcular_indicadores(
            y_validacion, probabilidades, umbral
        )
        resultados.append(indicadores)

    tabla = pd.DataFrame(resultados)

    # Mostrar algunos umbrales ilustrativos
    muestra = tabla[
        np.isclose(
            tabla["umbral"].to_numpy()[:, None],
            np.array([0.50, 0.60, 0.70, 0.80, 0.90]),
            atol=0.0001,
        ).any(axis=1)
    ]

    print("\nComparacion de umbrales:")
    print(muestra.to_string(index=False))

    # Buscar un candidato con recall >= 90 %
    candidatos = tabla[
        tabla["recall"] >= RECALL_MINIMO
    ].copy()

    if candidatos.empty:
        print("\nNo se encontró un umbral viable.")
    else:
        mejor = candidatos.sort_values(
            ["especificidad", "umbral"],
            ascending=[False, False],
        ).iloc[0]

        print("\nMejor candidato bajo la restriccion:")
        print(
            f"Umbral: {mejor['umbral']:.2f}\n"
            f"Recall: {mejor['recall']:.4f}\n"
            f"Especificidad: {mejor['especificidad']:.4f}\n"
            f"Falsos positivos: "
            f"{int(mejor['falsos_positivos'])}\n"
            f"Falsos negativos: "
            f"{int(mejor['falsos_negativos'])}\n"
            f"Vehiculos priorizados: "
            f"{int(mejor['vehiculos_priorizados'])}"
        )

print("\nExperimento de validacion finalizado.")
