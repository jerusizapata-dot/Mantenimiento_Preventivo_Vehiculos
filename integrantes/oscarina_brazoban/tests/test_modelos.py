
from pathlib import Path
import json

import joblib
import pandas as pd
import pytest

BASE = Path(__file__).resolve().parents[1]
RESULTADOS = BASE / "resultados"
DATOS = Path(__file__).resolve().parents[3] / "datos" / "raw" / "vehicle_maintenance_raw.csv"

VARIABLES = [
    "mileage",
    "vehicle_age",
    "reported_issues",
    "service_history",
    "vehicle_model",
    "maintenance_history",
]


@pytest.fixture(scope="module")
def datos_prueba():
    df = pd.read_csv(DATOS)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    indices = pd.read_csv(RESULTADOS / "indices_test.csv")
    X_test = df.loc[indices["indice_original"], VARIABLES]

    return X_test


@pytest.mark.parametrize(
    "nombre",
    ["baseline", "modelo1_regresion_logistica"],
)
def test_modelo_genera_predicciones_validas(nombre, datos_prueba):
    modelo = joblib.load(RESULTADOS / f"{nombre}.joblib")
    predicciones = modelo.predict(datos_prueba)

    assert len(predicciones) == len(datos_prueba)
    assert set(predicciones).issubset({0, 1})


def test_particiones_no_comparten_registros():
    train = pd.read_csv(RESULTADOS / "indices_train.csv")
    test = pd.read_csv(RESULTADOS / "indices_test.csv")

    indices_train = set(train["indice_original"])
    indices_test = set(test["indice_original"])

    assert indices_train.isdisjoint(indices_test)
    assert len(indices_train) == 40000
    assert len(indices_test) == 10000


def test_metricas_guardadas():
    with open(RESULTADOS / "metricas.json", encoding="utf-8") as archivo:
        metricas = json.load(archivo)

    for nombre in ["baseline", "modelo1_regresion_logistica"]:
        assert nombre in metricas

        for metrica in ["accuracy", "precision", "recall", "f1"]:
            assert 0 <= metricas[nombre][metrica] <= 1


def test_regresion_logistica_genera_probabilidades(datos_prueba):
    modelo = joblib.load(
        RESULTADOS / "modelo1_regresion_logistica.joblib"
    )

    probabilidades = modelo.predict_proba(datos_prueba.head(10))

    assert probabilidades.shape == (10, 2)
    assert ((probabilidades >= 0) & (probabilidades <= 1)).all()
    assert all(abs(fila.sum() - 1) < 1e-6 for fila in probabilidades)
