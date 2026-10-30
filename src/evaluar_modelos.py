
from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

# ==========================================================
# 1. RUTAS DEL PROYECTO
# ==========================================================

RAIZ = Path(__file__).resolve().parents[1]

RUTA_DATOS = RAIZ / "datos" / "raw" / "vehicle_maintenance_raw.csv"

RUTA_INDICES = (
    RAIZ / "integrantes" / "oscarina_brazoban"
    / "resultados" / "indices_test.csv"
)

RUTA_MODELOS = {
    "Baseline": (
        RAIZ / "integrantes" / "oscarina_brazoban"
        / "resultados" / "baseline.joblib"
    ),
    "Regresión logística": (
        RAIZ / "integrantes" / "oscarina_brazoban"
        / "resultados" / "modelo1_regresion_logistica.joblib"
    ),
    "Árbol de decisión": (
        RAIZ / "integrantes" / "manuel_manana"
        / "resultados" / "modelo_2_arbol_decision.joblib"
    ),
    "Random Forest": (
        RAIZ / "integrantes" / "manuel_manana"
        / "resultados" / "modelo_3_random_forest.joblib"
    ),
}

RUTA_SALIDA = RAIZ / "reports" / "evaluacion_lennen"
RUTA_SALIDA.mkdir(parents=True, exist_ok=True)


# ==========================================================
# 2. CARGA Y PREPARACIÓN DE LOS DATOS
# ==========================================================

print("Cargando datos e índices de prueba...")

df = pd.read_csv(RUTA_DATOS)

df.columns = (
    df.columns.str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

variables_numericas = [
    "mileage",
    "vehicle_age",
    "reported_issues",
    "service_history",
]

df = df.drop_duplicates()
df = df.dropna(subset=["need_maintenance"])

for columna in variables_numericas:
    df = df[df[columna].isna() | (df[columna] >= 0)]

# Leer los índices de prueba de Oscarina
indices_df = pd.read_csv(RUTA_INDICES)

if "indice_original" not in indices_df.columns:
    raise ValueError(
        "El archivo de índices no contiene la columna indice_original."
    )

indices_test = indices_df["indice_original"].astype(int).to_numpy()

if not set(indices_test).issubset(set(df.index)):
    raise ValueError(
        "Los índices de prueba no coinciden con los registros del dataset."
    )

# Seleccionar exactamente los mismos registros de prueba
df_test = df.loc[indices_test].copy()

variables = [
    "mileage",
    "vehicle_age",
    "reported_issues",
    "service_history",
    "vehicle_model",
    "maintenance_history",
]

columnas_faltantes = [
    columna for columna in variables + ["need_maintenance"]
    if columna not in df_test.columns
]

if columnas_faltantes:
    raise ValueError(
        f"Faltan columnas necesarias: {columnas_faltantes}"
    )

X_test = df_test[variables]
y_test = df_test["need_maintenance"].astype(int)

print(f"Registros de prueba: {len(y_test)}")


# ==========================================================
# 3. EVALUACIÓN DE LOS CUATRO MODELOS
# ==========================================================

resultados = []
predicciones_completas = []
errores = []

for nombre, ruta_modelo in RUTA_MODELOS.items():

    print(f"\nEvaluando: {nombre}")

    if not ruta_modelo.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo: {ruta_modelo}"
        )

    modelo = joblib.load(ruta_modelo)

    y_pred = modelo.predict(X_test)

    # Obtener probabilidades de la clase positiva
    y_score = None

    if hasattr(modelo, "predict_proba"):
        probabilidades = modelo.predict_proba(X_test)

        if hasattr(modelo, "classes_"):
            clases = modelo.classes_
        else:
            clases = modelo.named_steps["clasificador"].classes_

        if 1 in clases:
            posicion_positiva = list(clases).index(1)
            y_score = probabilidades[:, posicion_positiva]

    # Matriz de confusión
    matriz = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )

    tn, fp, fn, tp = matriz.ravel()

    # Métricas
    metricas = {
        "modelo": nombre,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(
            y_test, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_test, y_pred, zero_division=0
        ),
        "f1": f1_score(
            y_test, y_pred, zero_division=0
        ),
        "roc_auc": (
            roc_auc_score(y_test, y_score)
            if y_score is not None and y_test.nunique() == 2
            else None
        ),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }

    resultados.append(metricas)

    # Guardar predicciones
    predicciones = pd.DataFrame({
        "indice_original": df_test.index,
        "modelo": nombre,
        "valor_real": y_test.to_numpy(),
        "prediccion": y_pred,
    })

    if y_score is not None:
        predicciones["probabilidad_mantenimiento"] = y_score

    predicciones_completas.append(predicciones)

    # Identificar falsos negativos
    falsos_negativos = predicciones[
        (predicciones["valor_real"] == 1)
        & (predicciones["prediccion"] == 0)
    ].copy()

    falsos_negativos["tipo_error"] = "Falso negativo"

    # Identificar falsos positivos
    falsos_positivos = predicciones[
        (predicciones["valor_real"] == 0)
        & (predicciones["prediccion"] == 1)
    ].copy()

    falsos_positivos["tipo_error"] = "Falso positivo"

    errores.append(falsos_negativos)
    errores.append(falsos_positivos)

    print(
        f"Accuracy: {metricas['accuracy']:.4f} | "
        f"Precision: {metricas['precision']:.4f} | "
        f"Recall: {metricas['recall']:.4f} | "
        f"F1: {metricas['f1']:.4f}"
    )


# ==========================================================
# 4. GUARDAR RESULTADOS
# ==========================================================

df_resultados = pd.DataFrame(resultados)

df_resultados.to_csv(
    RUTA_SALIDA / "comparacion_modelos.csv",
    index=False,
    encoding="utf-8-sig"
)

with open(
    RUTA_SALIDA / "comparacion_modelos.json",
    "w",
    encoding="utf-8"
) as archivo:
    json.dump(
        resultados,
        archivo,
        indent=4,
        ensure_ascii=False
    )

df_predicciones = pd.concat(
    predicciones_completas,
    ignore_index=True
)

df_predicciones.to_csv(
    RUTA_SALIDA / "predicciones_modelos.csv",
    index=False,
    encoding="utf-8-sig"
)

df_errores = pd.concat(
    errores,
    ignore_index=True
)

df_errores.to_csv(
    RUTA_SALIDA / "analisis_errores.csv",
    index=False,
    encoding="utf-8-sig"
)


# ==========================================================
# 5. GENERAR INFORME DE EVALUACIÓN
# ==========================================================

lineas = [
    "# Informe de evaluación de modelos",
    "",
    "## Objetivo",
    "",
    "Comparar el desempeño de cuatro modelos para identificar "
    "vehículos que requieren mantenimiento preventivo.",
    "",
    "## Resultados comparativos",
    "",
    "| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC | FN | FP |",
    "|---|---:|---:|---:|---:|---:|---:|---:|",
]

for r in resultados:

    auc = (
        f"{r['roc_auc']:.4f}"
        if r["roc_auc"] is not None
        else "No disponible"
    )

    lineas.append(
        f"| {r['modelo']} "
        f"| {r['accuracy']:.4f} "
        f"| {r['precision']:.4f} "
        f"| {r['recall']:.4f} "
        f"| {r['f1']:.4f} "
        f"| {auc} "
        f"| {r['FN']} "
        f"| {r['FP']} |"
    )

lineas.extend([
    "",
    "## Análisis de errores",
    "",
    "Los falsos negativos representan vehículos que necesitan "
    "mantenimiento, pero que el modelo no identifica. "
    "Este error es especialmente importante porque puede retrasar "
    "una intervención preventiva.",
    "",
    "Los falsos positivos representan vehículos señalados para "
    "mantenimiento que no lo necesitan según la clasificación real. "
    "Pueden generar revisiones o gastos innecesarios.",
    "",
    "## Consideraciones y limitaciones",
    "",
    "- Las métricas corresponden al conjunto de prueba utilizado "
    "durante el entrenamiento.",
    "- El recall es especialmente importante para evaluar la "
    "detección de vehículos que necesitan mantenimiento.",
    "- Los resultados apoyan la toma de decisiones, pero no "
    "sustituyen la revisión técnica humana.",
    "- La calidad de las predicciones depende de la calidad y "
    "representatividad de los datos.",
    "",
    "## Archivos generados",
    "",
    "- comparacion_modelos.csv",
    "- comparacion_modelos.json",
    "- predicciones_modelos.csv",
    "- analisis_errores.csv",
])

(RUTA_SALIDA / "informe_evaluacion.md").write_text(
    "\n".join(lineas),
    encoding="utf-8"
)


# ==========================================================
# 6. RESUMEN FINAL
# ==========================================================

print("\n" + "=" * 60)
print("COMPARACIÓN DE LOS CUATRO MODELOS")
print("=" * 60)

print(
    df_resultados[
        [
            "modelo",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "FN",
            "FP",
        ]
    ].to_string(index=False)
)

print("\nEVALUACIÓN FINALIZADA")
print(f"Resultados guardados en: {RUTA_SALIDA}")
