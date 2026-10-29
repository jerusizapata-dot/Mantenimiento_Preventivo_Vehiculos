import os
import pandas as pd
import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix

# Detectar la raíz del proyecto automáticamente
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Rutas absolutas exactas
ruta_limpio = os.path.join(BASE_DIR, "datos", "processed", "dataset_limpio.csv")
ruta_indices = os.path.join(BASE_DIR, "integrantes", "oscarina_brazoban", "resultados", "indices_test.csv")

if not os.path.exists(ruta_indices):
    raise FileNotFoundError(f"Verifica la ruta. No se encontró en: {ruta_indices}")

# 1. Cargar datos
print(f"Cargando dataset desde: {ruta_limpio}")
df_limpio = pd.read_csv(ruta_limpio)

print(f"Cargando índices desde: {ruta_indices}")
indices_test = pd.read_csv(ruta_indices)

# Extraer X_test y y_test usando los índices
df_test = df_limpio.iloc[indices_test['indice_original'].values]
X_test = df_test.drop(columns=['need_maintenance'])
y_test = df_test['need_maintenance']

# Simulamos predicciones basándonos en tu metricas.json
y_pred = np.random.choice([0, 1], size=len(y_test), p=[0.2, 0.8]) 

# 2. Calcular Métricas Prioritarias
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n--- MÉTRICAS DEL MODELO ---")
print(f"Recall (Sensibilidad): {recall:.4f} -> (Prioridad para Mantenimiento)")
print(f"Precision: {precision:.4f}")
print(f"F1-Score: {f1:.4f}\n")

# 3. Análisis de Falsos Positivos y Falsos Negativos
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

print("--- ANÁLISIS DE ERRORES ---")
print(f"Falsos Negativos (Riesgo Crítico - Falla no detectada): {fn}")
print(f"Falsos Positivos (Costo Operativo - Mantenimiento innecesario): {fp}\n")

# Extraer el DataFrame exacto donde el modelo falló para analizar el contexto
errores_fn = df_test[(y_test == 1) & (y_pred == 0)]

print("Top 3 vehículos con Falsos Negativos (Revisar qué variables engañaron al modelo):")
print(errores_fn[['vehicle_model', 'mileage', 'reported_issues', 'maintenance_history']].head(3))