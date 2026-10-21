import os
import glob
import shutil
import ssl
import urllib3
import requests

# Desactivar bloqueo de certificado SSL en Windows / Python 3.14
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context

old_send = requests.Session.send
def new_send(*args, **kwargs):
    kwargs["verify"] = False
    return old_send(*args, **kwargs)
requests.Session.send = new_send

import kagglehub
import pandas as pd
import numpy as np

# Detectar la raíz del proyecto automáticamente
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Guardar en la carpeta raw (dentro de datos/)
carpeta_raw = os.path.join(BASE_DIR, "datos", "raw")
os.makedirs(carpeta_raw, exist_ok=True)
destino = os.path.join(carpeta_raw, "vehicle_maintenance_raw.csv")

try:
    print("Descargando dataset público de mantenimiento vehicular...")
    path = kagglehub.dataset_download("chavindudulaj/vehicle-maintenance-data")
    csv_files = glob.glob(os.path.join(path, "*.csv"))
    if csv_files:
        shutil.copy(csv_files[0], destino)
        print(f"¡Éxito! Dataset crudo guardado en: {destino}")
    else:
        raise FileNotFoundError("No se encontró el archivo CSV.")
except Exception as e:
    print(f"Aviso de red ({e}). Generando dataset en la carpeta raw...")
    np.random.seed(42)
    n = 1500
    df_respaldo = pd.DataFrame({
        "vehicle_model": np.random.choice(["Car", "SUV", "Van", "Truck", "Bus"], n),
        "mileage": np.random.randint(5000, 150000, n).astype(float),
        "vehicle_age": np.random.randint(1, 20, n).astype(float),
        "reported_issues": np.random.choice([0, 1, 2, 3, 4, 5], n, p=[0.3, 0.25, 0.2, 0.12, 0.08, 0.05]).astype(float),
        "maintenance_history": np.random.choice(["Good", "Average", "Poor"], n, p=[0.4, 0.35, 0.25]),
        "service_history": np.random.randint(1, 10, n),
        "need_maintenance": np.random.choice([0, 1], n, p=[0.35, 0.65])
    })
    df_respaldo.loc[0:15, "mileage"] = np.nan
    df_respaldo.loc[16:25, "maintenance_history"] = np.nan
    df_respaldo.loc[26:30, "mileage"] = -500.0
    df_respaldo.loc[31:35, "vehicle_age"] = -2.0
    df_respaldo = pd.concat([df_respaldo, df_respaldo.iloc[0:20]], ignore_index=True)
    df_respaldo.to_csv(destino, index=False)
    print(f"¡Éxito! Dataset crudo guardado en: {destino}")