# %%
import pandas as pd
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
from sklearn.model_selection import train_test_split

# Dividir los datos: 80% para entrenamiento, 20% para prueba
# random_state=42 asegura que la división sea siempre la misma, haciendo el resultado reproducible
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n--- Tamaños de los conjuntos de datos ---")
print("Datos de entrenamiento (X_train):", X_train.shape)
print("Datos de prueba (X_test):", X_test.shape)
print("Objetivo de entrenamiento (y_train):", y_train.shape)
print("Objetivo de prueba (y_test):", y_test.shape)

# %%
from sklearn.ensemble import RandomForestRegressor

# 1. Crear una instancia del modelo
# n_estimators=100 significa que el "bosque" tendrá 100 árboles de decisión.
# random_state=42 es para que los resultados sean reproducibles.
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

# 2. Entrenar el modelo con los datos de entrenamiento
# El comando .fit() es el que inicia el proceso de aprendizaje.
model.fit(X_train, y_train)

print("¡Modelo entrenado exitosamente!")

# %%
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# 1. Realizar predicciones con el conjunto de prueba
y_pred = model.predict(X_test)

# 2. Calcular las métricas de evaluación
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("--- Métricas de Evaluación del Modelo ---")
print(f"R² (Coeficiente de Determinación): {r2:.2f}")
print(f"Error Absoluto Medio (MAE): {mae:.2f}")
print(f"Raíz del Error Cuadrático Medio (RMSE): {rmse:.2f}")

# 3. Visualizar Predicciones vs. Valores Reales
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
plt.title('Valores Reales vs. Predicciones')
plt.xlabel('Valores Reales (count)')
plt.ylabel('Predicciones del Modelo (count)')
plt.grid(True)
plt.show()


