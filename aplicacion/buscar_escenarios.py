
from itertools import product
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

model = joblib.load(
    ROOT / "integrantes/manuel_manana/resultados/modelo_3_random_forest.joblib"
)

columns = [
    "vehicle_model",
    "mileage",
    "vehicle_age",
    "reported_issues",
    "maintenance_history",
    "service_history",
]

combinations = product(
    ["Car", "SUV", "Bus", "Truck", "Van", "Motorcycle"],
    [30001, 40000, 55000, 65000, 80000],
    [1, 3, 5, 7, 8, 10],
    [0, 1, 2, 3],
    ["Good", "Average", "Poor"],
    [1, 3, 5, 8, 10],
)

data = pd.DataFrame(list(combinations), columns=columns)

probabilities = model.predict_proba(data)
classes = list(model.classes_)
positive_index = classes.index(1)

data["probabilidad"] = probabilities[:, positive_index]

alto = data[
    (data["probabilidad"] >= 0.75)
    & (data["probabilidad"] < 0.90)
    & (data["reported_issues"] < 4)
]

bajo = data[
    (data["probabilidad"] < 0.50)
    & (data["vehicle_age"] < 8)
]

print("\nESCENARIOS DE RIESGO ALTO:")
print(alto.head(5).to_string(index=False))
print("Total encontrados:", len(alto))

print("\nESCENARIOS DE RIESGO BAJO:")
print(bajo.head(5).to_string(index=False))
print("Total encontrados:", len(bajo))
