# Flujo de Trabajo para Detección de Fraude
# Incluye: Preprocesamiento, SMOTE y RandomizedSearchCV
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Importaciones de Scikit-Learn
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from scipy.stats import randint, uniform

# Crear carpeta para guardar gráficos
output_dir = 'graficos_fraude'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Carpeta '{output_dir}' creada.\n")

# --- 1. Carga y Preparación de Datos ---
print("--- 1. Cargando y Preprocesando Datos ---")
try:
    df = pd.read_csv('creditcard.csv')
except FileNotFoundError:
    print("Error: El archivo 'creditcard.csv' no se encontró.")
    exit()

# Escalar características 'Time' y 'Amount'
scaler = StandardScaler()
df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df['scaled_time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))
df.drop(['Time', 'Amount'], axis=1, inplace=True)

# Definir características (X) y objetivo (y)
X = df.drop('Class', axis=1)
y = df['Class']

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Datos listos.\n")

# --- 2. Modelo Baseline (Sin SMOTE ni optimización) ---
print("--- 2. Entrenando Modelo Baseline ---")
model_baseline = LogisticRegression(random_state=42, max_iter=1000)
model_baseline.fit(X_train, y_train)
y_pred_baseline = model_baseline.predict(X_test)

print("Reporte del Modelo Baseline:")
print(classification_report(y_test, y_pred_baseline, target_names=['No Fraude', 'Fraude']))


# --- 3. Optimización con RandomizedSearchCV ---
print("\n--- 3. Optimizando con RandomizedSearchCV ---")

# Definir el pipeline
lr_pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('classifier', LogisticRegression(random_state=42, max_iter=5000))
])

# Espacio de búsqueda de hiperparámetros
param_distributions_lr = {
    'smote__k_neighbors': randint(3, 20),
    'classifier__C': uniform(0.1, 20),
    'classifier__solver': ['liblinear']
}

# Configurar RandomizedSearchCV
random_search_lr = RandomizedSearchCV(
    estimator=lr_pipeline,
    param_distributions=param_distributions_lr,
    n_iter=20,
    cv=5,
    scoring='recall',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

print("Iniciando la búsqueda de hiperparámetros...")
random_search_lr.fit(X_train, y_train)

# Mejores resultados
print("\nMejores Hiperparámetros encontrados:")
print(random_search_lr.best_params_)
print(f"\nMejor Recall Score (en validación cruzada): {random_search_lr.best_score_:.4f}\n")

# --- 4. Evaluación Final del Modelo Optimizado ---
print("--- 4. Evaluación del Modelo Optimizado en el Conjunto de Test ---")
best_model_lr = random_search_lr.best_estimator_
y_pred_optimized = best_model_lr.predict(X_test)

print("Reporte de Clasificación (Modelo Optimizado):")
print(classification_report(y_test, y_pred_optimized, target_names=['No Fraude', 'Fraude']))


# --- 5. Visualización Comparativa y Guardado ---
print("\n--- 5. Comparando Matrices de Confusión ---")
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
class_names = ['No Fraude', 'Fraude']

# Matriz 1: Modelo Baseline
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_baseline, display_labels=class_names, cmap='Blues', normalize='true', ax=axes[0])
axes[0].set_title('Matriz de Confusión - Modelo Baseline', fontsize=14)
axes[0].grid(False)

# Matriz 2: Modelo Optimizado con RandomizedSearchCV
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_optimized, display_labels=class_names, cmap='Greens', normalize='true', ax=axes[1])
axes[1].set_title('Matriz de Confusión - Modelo Optimizado', fontsize=14)
axes[1].grid(False)

plt.suptitle('Comparación de Modelos (Normalizado)', fontsize=20)

# Guardar la figura
output_path = os.path.join(output_dir, 'comparacion_modelos.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Gráfico guardado en: {output_path}")

plt.show()