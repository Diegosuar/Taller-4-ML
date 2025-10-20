# Predicción de Demanda de Bicicletas
# Con guardado de gráficos como imágenes

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Crear carpeta para guardar gráficos
output_dir = 'graficos_bike'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Carpeta '{output_dir}' creada.\n")

# %%
file_path = 'hour.csv'
try:
    df = pd.read_csv(file_path)
    print("Primeras 5 filas del DataFrame:")
    print(df.head())

except FileNotFoundError:
    print(f"Error: No se encontró el archivo '{file_path}'.")
    print("Por favor, asegúrate de haberlo descargado y colocado en la misma carpeta que tu script.")
except Exception as e:
    print(f"Ocurrió un error al leer el archivo localmente: {e}")
    print("Esto confirma que el archivo en sí mismo tiene un problema de formato.")

# %%
# Renombrar columnas para mayor claridad
df.rename(columns={'weathersit': 'weather',
                   'mnth': 'month',
                   'hr': 'hour',
                   'hum': 'humidity',
                   'cnt': 'count'}, inplace=True)

# Eliminar columnas irrelevantes o con fuga de datos
df_processed = df.drop([
    'instant',    # Es solo un índice, no es útil para predecir
    'dteday',     # La información ya está en 'year', 'month', 'weekday'
    'casual',     # Fuga de datos (componente de 'count')
    'registered', # Fuga de datos (componente de 'count')
    'atemp'       # Altamente correlacionada con 'temp', podemos quitarla para simplificar
], axis=1)

print("\nDataFrame después de la limpieza y transformación:")
print(df_processed.head())

# %%
# 'y' es la variable que queremos predecir
y = df_processed['count']

# 'X' son todas las demás columnas que usaremos como características para la predicción
X = df_processed.drop('count', axis=1)

print("\nForma de las características (X):", X.shape)
print("Forma de la variable objetivo (y):", y.shape)

print("\nPrimeras 5 filas de las características (X):")
print(X.head())

# %%
# Dividir los datos: 80% para entrenamiento, 20% para prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n--- Tamaños de los conjuntos de datos ---")
print("Datos de entrenamiento (X_train):", X_train.shape)
print("Datos de prueba (X_test):", X_test.shape)
print("Objetivo de entrenamiento (y_train):", y_train.shape)
print("Objetivo de prueba (y_test):", y_test.shape)

# %%
# 1. Crear una instancia del modelo
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

# 2. Entrenar el modelo con los datos de entrenamiento
model.fit(X_train, y_train)

print("\n¡Modelo entrenado exitosamente!")

# %%
# 1. Realizar predicciones con el conjunto de prueba
y_pred = model.predict(X_test)

# 2. Calcular las métricas de evaluación
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n--- Métricas de Evaluación del Modelo ---")
print(f"R² (Coeficiente de Determinación): {r2:.2f}")
print(f"Error Absoluto Medio (MAE): {mae:.2f}")
print(f"Raíz del Error Cuadrático Medio (RMSE): {rmse:.2f}")

# 3. Visualizar Predicciones vs. Valores Reales
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(y_test, y_pred, alpha=0.3, s=30)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2, label='Predicción Perfecta')
ax.set_title('Valores Reales vs. Predicciones', fontsize=14, fontweight='bold')
ax.set_xlabel('Valores Reales (count)', fontsize=12)
ax.set_ylabel('Predicciones del Modelo (count)', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

output_path = os.path.join(output_dir, 'predicciones_vs_reales.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nGráfico guardado en: {output_path}")
plt.close()

# 4. Gráfico adicional: Residuos
residuos = y_test - y_pred
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(y_pred, residuos, alpha=0.3, s=30)
ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax.set_title('Análisis de Residuos', fontsize=14, fontweight='bold')
ax.set_xlabel('Predicciones del Modelo', fontsize=12)
ax.set_ylabel('Residuos', fontsize=12)
ax.grid(True, alpha=0.3)

output_path = os.path.join(output_dir, 'analisis_residuos.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Gráfico guardado en: {output_path}")
plt.close()

# 5. Importancia de características
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(feature_importance['feature'], feature_importance['importance'], color='steelblue', alpha=0.8)
ax.set_xlabel('Importancia', fontsize=12)
ax.set_title('Importancia de Características en el Modelo', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

output_path = os.path.join(output_dir, 'importancia_caracteristicas.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Gráfico guardado en: {output_path}")
plt.close()

# 6. Distribución de errores
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(residuos, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
ax.set_title('Distribución de Errores (Residuos)', fontsize=14, fontweight='bold')
ax.set_xlabel('Residuos', fontsize=12)
ax.set_ylabel('Frecuencia', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

output_path = os.path.join(output_dir, 'distribucion_errores.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Gráfico guardado en: {output_path}")

