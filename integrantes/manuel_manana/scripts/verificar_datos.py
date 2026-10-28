
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[3]

ruta_datos = BASE / "datos/raw/vehicle_maintenance_raw.csv"
ruta_indices = (
    BASE / "integrantes/oscarina_brazoban/resultados"
)

df = pd.read_csv(ruta_datos)
df.columns = (
    df.columns.str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

train = pd.read_csv(ruta_indices / "indices_train.csv")
test = pd.read_csv(ruta_indices / "indices_test.csv")

variables = [
    "mileage",
    "vehicle_age",
    "reported_issues",
    "service_history",
    "vehicle_model",
    "maintenance_history",
]

objetivo = "need_maintenance"

assert all(col in df.columns for col in variables + [objetivo])
assert df[objetivo].isin([0, 1]).all()

idx_train = train["indice_original"]
idx_test = test["indice_original"]

assert len(idx_train) == 40000
assert len(idx_test) == 10000
assert idx_train.is_unique and idx_test.is_unique
assert set(idx_train).isdisjoint(set(idx_test))
assert set(idx_train).union(set(idx_test)) == set(df.index)

print("Validación de datos completada correctamente.")
print(f"Total de registros: {len(df)}")
print(f"Entrenamiento: {len(idx_train)}")
print(f"Prueba: {len(idx_test)}")
