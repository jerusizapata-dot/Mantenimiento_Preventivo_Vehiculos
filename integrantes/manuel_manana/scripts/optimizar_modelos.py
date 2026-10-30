r"""Búsqueda inicial de hiperparámetros y umbral para modelos 2 y 3.

Ejecutar desde la raíz del repositorio:
    .\.venv\Scripts\python.exe integrantes/manuel_manana/scripts/optimizar_modelos.py

No utiliza índices de test. No sobrescribe modelos ni métricas iniciales.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

BASE = Path(__file__).resolve().parents[3]
SALIDA = Path(__file__).resolve().parents[1] / "resultados" / "optimizacion"
CARPETA_INDICES = BASE / "integrantes" / "oscarina_brazoban" / "resultados"
NUMERICAS = ["mileage", "vehicle_age", "reported_issues", "service_history"]
CATEGORICAS = ["vehicle_model", "maintenance_history"]
VARIABLES = NUMERICAS + CATEGORICAS
OBJETIVO = "need_maintenance"


def cargar_entrenamiento():
    df = pd.read_csv(BASE / "datos" / "raw" / "vehicle_maintenance_raw.csv")
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df = df.drop_duplicates().dropna(subset=[OBJETIVO]).copy()
    for col in NUMERICAS:
        df = df[df[col].isna() | (df[col] >= 0)]
    if not df[OBJETIVO].isin([0, 1]).all():
        raise ValueError("La variable objetivo debe contener solo 0 y 1")

    train = pd.read_csv(CARPETA_INDICES / "indices_train.csv")["indice_original"]
    test = pd.read_csv(CARPETA_INDICES / "indices_test.csv")["indice_original"]
    if not (len(train) == 40000 and len(test) == 10000):
        raise ValueError("Tamaños distintos de los publicados por Oscarina")
    if (not train.is_unique or not test.is_unique
            or not set(train).isdisjoint(set(test))
            or not set(train).issubset(set(df.index))
            or not set(test).issubset(set(df.index))):
        raise ValueError("Los índices no corresponden a la fuente depurada")
    print(f"Registros train: {len(train)}; test reservado sin usar: {len(test)}")
    return df.loc[train, VARIABLES], df.loc[train, OBJETIVO]


def preprocesador():
    return ColumnTransformer([
        ("numericas", SimpleImputer(strategy="median"), NUMERICAS),
        ("categoricas", Pipeline([
            ("imputacion", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAS),
    ])


def buscar_umbral(y, probabilidades, recall_minimo):
    filas = []
    for umbral in np.linspace(0.0, 1.0, 201):
        etiquetas = (probabilidades >= umbral).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, etiquetas, labels=[0, 1]).ravel()
        recall = tp / (tp + fn) if tp + fn else 0.0
        especificidad = tn / (tn + fp) if tn + fp else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        filas.append({
            "umbral": float(round(umbral, 4)),
            "recall": float(recall),
            "especificidad": float(especificidad),
            "precision": float(precision),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "vehiculos_senalados": int(tp + fp),
        })
    tabla = pd.DataFrame(filas)
    elegibles = tabla.loc[tabla["recall"] >= recall_minimo]
    if elegibles.empty:
        return tabla, None
    mejor = elegibles.sort_values(
        ["especificidad", "recall", "umbral"],
        ascending=[False, False, False],
    ).iloc[0].to_dict()
    return tabla, mejor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteraciones", type=int, default=6,
                        help="Configuraciones aleatorias por algoritmo (por defecto 6)")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--recall-minimo", type=float, default=0.95)
    args = parser.parse_args()
    if not 0 < args.recall_minimo <= 1 or args.iteraciones < 1 or args.folds < 2:
        parser.error("Usa recall en (0,1], iteraciones >= 1 y folds >= 2")

    SALIDA.mkdir(parents=True, exist_ok=True)
    X, y = cargar_entrenamiento()
    # La selección de hiperparámetros usa CV SOLO sobre desarrollo (32k).
    # Validación (8k) se reserva para elegir un umbral provisional.
    X_dev, X_val, y_dev, y_val = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)

    configuraciones = {
        "modelo_2_arbol": (
            DecisionTreeClassifier(random_state=42),
            {
                "clasificador__max_depth": [4, 6, 8, 12, None],
                "clasificador__min_samples_leaf": [10, 30, 60, 100],
                "clasificador__class_weight": [None, "balanced"],
                "clasificador__min_samples_split": [2, 20, 60],
            },
        ),
        "modelo_3_random_forest": (
            RandomForestClassifier(random_state=42, n_jobs=-1),
            {
                "clasificador__n_estimators": [60, 100, 140],
                "clasificador__max_depth": [6, 10, 14, None],
                "clasificador__min_samples_leaf": [10, 30, 60],
                "clasificador__class_weight": [None, "balanced"],
                "clasificador__max_features": ["sqrt", 0.8],
            },
        ),
    }

    resumen = {}
    for nombre, (algoritmo, espacio) in configuraciones.items():
        print(f"\n--- {nombre} ---", flush=True)
        pipeline = Pipeline([
            ("preprocesamiento", preprocesador()),
            ("clasificador", algoritmo),
        ])
        busqueda = RandomizedSearchCV(
            pipeline,
            param_distributions=espacio,
            n_iter=args.iteraciones,
            scoring="roc_auc",  # calidad para ordenar riesgos, independiente del umbral
            cv=cv,
            n_jobs=1,  # evita paralelismo anidado y uso excesivo de RAM en Windows
            random_state=42,
            refit=True,
            verbose=1,
            error_score="raise",
        )
        busqueda.fit(X_dev, y_dev)
        prob = busqueda.best_estimator_.predict_proba(X_val)[:, 1]
        tabla, candidato = buscar_umbral(y_val.to_numpy(), prob, args.recall_minimo)
        tabla.to_csv(SALIDA / f"umbrales_{nombre}.csv", index=False)
        pd.DataFrame(busqueda.cv_results_)[
            ["params", "mean_test_score", "std_test_score", "rank_test_score"]
        ].to_csv(SALIDA / f"busqueda_cv_{nombre}.csv", index=False)

        registro = {
            "n_desarrollo": int(len(X_dev)),
            "n_validacion": int(len(X_val)),
            "roc_auc_cv_desarrollo": float(busqueda.best_score_),
            "roc_auc_validacion": float(roc_auc_score(y_val, prob)),
            "average_precision_validacion": float(average_precision_score(y_val, prob)),
            "mejores_parametros": busqueda.best_params_,
            "recall_minimo_exploratorio": args.recall_minimo,
            "umbral_candidato_validacion": candidato,
            "advertencia": (
                "Umbral elegido y medido sobre la misma validación: cifras optimistas. "
                "El test oficial no se ha usado en este script. No es validación externa."
            ),
        }
        resumen[nombre] = registro
        print("Mejor ROC-AUC CV:", round(busqueda.best_score_, 4))
        print("ROC-AUC validación:", round(registro["roc_auc_validacion"], 4))
        print("Mejores parámetros:", busqueda.best_params_)
        print("Umbral candidato:", candidato)

    with (SALIDA / "resumen_optimizacion.json").open("w", encoding="utf-8") as archivo:
        json.dump(resumen, archivo, indent=2, ensure_ascii=False)
    print("\nArchivos de análisis guardados en:", SALIDA)
    print("No se han modificado las versiones iniciales ni se ha usado el test.")


if __name__ == "__main__":
    main()
