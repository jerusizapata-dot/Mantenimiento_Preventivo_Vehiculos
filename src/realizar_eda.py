import os
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Detectar la raíz del proyecto automáticamente (funciona desde cualquier carpeta)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ruta_raw = os.path.join(BASE_DIR, "datos", "raw", "vehicle_maintenance_raw.csv")
carpeta_processed = os.path.join(BASE_DIR, "datos", "processed")
carpeta_reports = os.path.join(BASE_DIR, "reports")

os.makedirs(carpeta_processed, exist_ok=True)
os.makedirs(carpeta_reports, exist_ok=True)

# 1. Cargar datos crudos desde raw
df = pd.read_csv(ruta_raw)
print(f"Dimensiones originales (en raw): {df.shape[0]} filas, {df.shape[1]} columnas")

# Estandarizar nombres de columnas
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

# Calcular antigüedad si viene como año de modelo
anio_actual = datetime.datetime.now().year
if 'vehicle_age' not in df.columns and 'model_year' in df.columns:
    df['vehicle_age'] = anio_actual - df['model_year']

# 2. Limpiar registros duplicados
duplicados = df.duplicated().sum()
df = df.drop_duplicates()
print(f"Duplicados eliminados: {duplicados}")

# 3. Limpiar registros inconsistentes (valores negativos)
for col in ['mileage', 'vehicle_age', 'reported_issues']:
    if col in df.columns:
        inconsistentes = (df[col] < 0).sum()
        df = df[df[col] >= 0]
        print(f"Inconsistencias eliminadas en {col}: {inconsistentes}")

# 4. Limpiar datos faltantes (nulos)
nulos = df.isnull().sum().sum()
print(f"Valores faltantes corregidos: {nulos}")

for col in df.select_dtypes(include=[np.number]).columns:
    df[col] = df[col].fillna(df[col].median())

for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# 5. Guardar Entregable 1: Dataset limpio
ruta_limpio = os.path.join(carpeta_processed, "dataset_limpio.csv")
df.to_csv(ruta_limpio, index=False)
print(f"¡Éxito! Dataset limpio guardado en: {ruta_limpio} ({df.shape[0]} filas válidas)")

# 6. Crear gráficos del Análisis Exploratorio (Entregable 3)
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

if 'mileage' in df.columns and 'need_maintenance' in df.columns:
    sns.boxplot(data=df, x='need_maintenance', y='mileage', hue='need_maintenance', ax=axes[0, 0], palette='Set2', legend=False)
    axes[0, 0].set_title('Kilometraje vs Necesidad de Mantenimiento')
    axes[0, 0].set_xlabel('Requiere Mantenimiento (0 = No, 1 = Sí)')
    axes[0, 0].set_ylabel('Kilometraje')

if 'vehicle_age' in df.columns and 'reported_issues' in df.columns:
    sns.lineplot(data=df, x='vehicle_age', y='reported_issues', marker='o', ax=axes[0, 1], color='coral')
    axes[0, 1].set_title('Antigüedad del Vehículo vs Promedio de Fallas')
    axes[0, 1].set_xlabel('Antigüedad (Años)')
    axes[0, 1].set_ylabel('Fallas Reportadas')

if 'reported_issues' in df.columns and 'need_maintenance' in df.columns:
    sns.countplot(data=df, x='reported_issues', hue='need_maintenance', ax=axes[1, 0], palette='viridis')
    axes[1, 0].set_title('Fallas Reportadas vs Necesidad de Mantenimiento')
    axes[1, 0].set_xlabel('Número de Fallas Reportadas')
    axes[1, 0].set_ylabel('Cantidad de Vehículos')

cols_clave = [c for c in ['mileage', 'vehicle_age', 'reported_issues', 'service_history', 'need_maintenance'] if c in df.columns]
if len(cols_clave) >= 2:
    sns.heatmap(df[cols_clave].corr(), annot=True, cmap='Blues', fmt=".2f", ax=axes[1, 1])
    axes[1, 1].set_title('Correlación entre Variables Clave')

plt.tight_layout()
ruta_grafico = os.path.join(carpeta_reports, "graficos_eda.png")
plt.savefig(ruta_grafico, dpi=300)
print(f"¡Éxito! Gráficos guardados en: {ruta_grafico}")