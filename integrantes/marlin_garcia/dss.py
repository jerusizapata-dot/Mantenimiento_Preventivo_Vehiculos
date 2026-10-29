from pathlib import Path
import pandas as pd

# ============================================================
# RUTAS
# ============================================================

BASE = Path(__file__).resolve().parents[2]

DATOS = (
    BASE /
    "datos/raw/vehicle_maintenance_raw.csv"
)

PREDICCIONES = (
    BASE /
    "integrantes/manuel_manana/resultados/"
    "predicciones_modelo_3.csv"
)

RESULTADOS = (
    Path(__file__).resolve().parents[1]
    / "resultados"
)

RESULTADOS.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================================
# CARGA DE DATOS
# ============================================================

df = pd.read_csv(DATOS)

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

pred = pd.read_csv(PREDICCIONES)

# ============================================================
# RECUPERAR VARIABLES ORIGINALES
# ============================================================

variables = [
    "mileage",
    "vehicle_age",
    "reported_issues",
    "service_history",
    "maintenance_history"
]

datos_dss = pred.merge(
    df[variables],
    left_on="indice_original",
    right_index=True,
    how="left"
)

# ============================================================
# REGLAS DSS
# ============================================================

def calcular_riesgo(fila):

    prob = fila["probabilidad_mantenimiento"]

    issues = fila["reported_issues"]

    mileage = fila["mileage"]

    edad = fila["vehicle_age"]

    if (
        prob >= 0.90
        or issues >= 4
    ):
        return "CRITICO"

    if (
        prob >= 0.75
        or mileage >= 150000
    ):
        return "ALTO"

    if (
        prob >= 0.50
        or edad >= 8
    ):
        return "MEDIO"

    return "BAJO"


def prioridad(riesgo):

    niveles = {
        "CRITICO": 1,
        "ALTO": 2,
        "MEDIO": 3,
        "BAJO": 4,
    }

    return niveles[riesgo]


def accion(riesgo):

    acciones = {
        "CRITICO":
        "Atencion inmediata",

        "ALTO":
        "Programar mantenimiento en menos de 7 dias",

        "MEDIO":
        "Inspeccion preventiva",

        "BAJO":
        "Monitoreo periodico",
    }

    return acciones[riesgo]


# ============================================================
# GENERAR RESULTADOS
# ============================================================

datos_dss["nivel_riesgo"] = (
    datos_dss.apply(
        calcular_riesgo,
        axis=1
    )
)

datos_dss["prioridad"] = (
    datos_dss["nivel_riesgo"]
    .apply(prioridad)
)

datos_dss["accion_recomendada"] = (
    datos_dss["nivel_riesgo"]
    .apply(accion)
)

datos_dss = datos_dss.sort_values(
    by="prioridad"
)

# ============================================================
# EXPORTAR
# ============================================================

salida = (
    RESULTADOS /
    "matriz_prioridades.csv"
)

datos_dss.to_csv(
    salida,
    index=False
)

print(
    "DSS ejecutado correctamente."
)

print(
    f"Vehiculos evaluados: "
    f"{len(datos_dss)}"
)

print(
    f"Resultado guardado en: "
    f"{salida}"
)