
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ============================================================
# 1. RUTAS Y CONFIGURACIÓN
# ============================================================

BASE = Path(__file__).resolve().parents[3]

CARPETA_MODELOS = BASE / "integrantes/manuel_manana"
RESULTADOS = CARPETA_MODELOS / "resultados"

INDICES = (
    BASE / "integrantes/oscarina_brazoban/resultados"
)

VARIABLES = [
    "mileage",
    "vehicle_age",
    "reported_issues",
    "service_history",
    "vehicle_model",
    "maintenance_history",
]

OBJETIVO = "need_maintenance"

MODELOS = {
    2: "modelo_2_arbol_decision.joblib",
    3: "modelo_3_random_forest.joblib",
}


# ============================================================
# 2. FIXTURES: DATOS COMPARTIDOS ENTRE PRUEBAS
# ============================================================

@pytest.fixture(scope="module")
def datos():
    df = pd.read_csv(
        BASE / "datos/raw/vehicle_maintenance_raw.csv"
    )
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )
    return df


@pytest.fixture(scope="module")
def particiones():
    train = pd.read_csv(INDICES / "indices_train.csv")
    test = pd.read_csv(INDICES / "indices_test.csv")
    return train, test


@pytest.fixture(params=[2, 3])
def resultado_modelo(request):
    numero = request.param

    predicciones = pd.read_csv(
        RESULTADOS / f"predicciones_modelo_{numero}.csv"
    )

    with open(
        RESULTADOS / f"metricas_modelo_{numero}.json",
        encoding="utf-8",
    ) as archivo:
        metricas = json.load(archivo)

    return numero, predicciones, metricas


# ============================================================
# 3. PRUEBAS DE INTEGRIDAD DE DATOS
# ============================================================

def test_dataset_contiene_variables(datos):
    assert len(datos) == 50000

    for variable in VARIABLES + [OBJETIVO]:
        assert variable in datos.columns

    assert datos[OBJETIVO].isin([0, 1]).all()


def test_particiones_son_validas(particiones, datos):
    train, test = particiones

    idx_train = train["indice_original"]
    idx_test = test["indice_original"]

    assert len(idx_train) == 40000
    assert len(idx_test) == 10000

    assert idx_train.is_unique
    assert idx_test.is_unique

    assert set(idx_train).isdisjoint(set(idx_test))

    assert (
        set(idx_train) | set(idx_test)
    ) == set(datos.index)


# ============================================================
# 4. PRUEBAS DE PREDICCIONES
# ============================================================

def test_predicciones_validas(resultado_modelo):
    numero, predicciones, _ = resultado_modelo

    assert len(predicciones) == 10000

    assert predicciones["prediccion"].notna().all()

    assert set(
        predicciones["prediccion"].unique()
    ).issubset({0, 1})


def test_probabilidades_validas(resultado_modelo):
    _, predicciones, _ = resultado_modelo

    probabilidades = predicciones[
        "probabilidad_mantenimiento"
    ]

    assert probabilidades.notna().all()
    assert np.isfinite(probabilidades).all()

    assert probabilidades.between(0, 1).all()


def test_predicciones_usan_indices_correctos(
    resultado_modelo,
    particiones,
    datos,
):
    _, predicciones, _ = resultado_modelo

    _, test = particiones

    indices = test["indice_original"].to_numpy()

    np.testing.assert_array_equal(
        predicciones["indice_original"].to_numpy(),
        indices,
    )

    np.testing.assert_array_equal(
        predicciones["y_real"].to_numpy(),
        datos.loc[indices, OBJETIVO].to_numpy(),
    )


# ============================================================
# 5. PRUEBAS DE MÉTRICAS
# ============================================================

def test_metricas_coinciden(resultado_modelo):
    _, predicciones, metricas = resultado_modelo

    y_real = predicciones["y_real"]
    y_pred = predicciones["prediccion"]

    calculadas = {
        "accuracy": accuracy_score(y_real, y_pred),
        "precision": precision_score(
            y_real, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_real, y_pred, zero_division=0
        ),
        "f1": f1_score(
            y_real, y_pred, zero_division=0
        ),
    }

    for nombre, valor in calculadas.items():
        assert metricas[nombre] == pytest.approx(
            valor, abs=1e-10
        )

    matriz = confusion_matrix(
        y_real, y_pred, labels=[0, 1]
    )

    assert metricas["matriz_confusion"] == matriz.tolist()


# ============================================================
# 6. PRUEBAS DE MODELOS SERIALIZADOS
# ============================================================

@pytest.mark.parametrize("numero", [2, 3])
def test_modelo_guardado_reproduce_predicciones(
    numero,
    datos,
):
    ruta_modelo = RESULTADOS / MODELOS[numero]

    assert ruta_modelo.exists()

    # Cargar únicamente archivos generados localmente
    # por nuestro propio entrenamiento.
    modelo = joblib.load(ruta_modelo)

    predicciones = pd.read_csv(
        RESULTADOS / f"predicciones_modelo_{numero}.csv"
    )

    # Muestra pequeña para evitar repetir toda la evaluación
    muestra = predicciones.head(30)
    indices = muestra["indice_original"].to_numpy()

    X = datos.loc[indices, VARIABLES]

    y_pred = modelo.predict(X)
    prob = modelo.predict_proba(X)

    clases = list(modelo.classes_)
    columna_positiva = clases.index(1)

    np.testing.assert_array_equal(
        y_pred,
        muestra["prediccion"].to_numpy(),
    )

    np.testing.assert_allclose(
        prob[:, columna_positiva],
        muestra["probabilidad_mantenimiento"].to_numpy(),
        rtol=1e-7,
        atol=1e-9,
    )

    assert prob.shape == (30, 2)
    np.testing.assert_allclose(
        prob.sum(axis=1),
        np.ones(30),
        atol=1e-9,
    )
