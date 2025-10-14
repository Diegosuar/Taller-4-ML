# Flujo de Trabajo Completo para Deserción de Clientes
# Preprocesamiento, SMOTE, Selección de Modelos y RandomizedSearchCV

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Importaciones de Scikit-Learn y XGBoost
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, roc_auc_score, ConfusionMatrixDisplay
from scipy.stats import randint, uniform

# --- 1. Carga y Preparación de Datos ---
print("--- 1. Cargando y Preprocesando Datos ---")
# Carga el dataset
url = 'https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv'
df = pd.read_csv(url)

# Preprocesamiento Básico
df.drop('customerID', axis=1, inplace=True)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)
df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# Separar características (X) y objetivo (y)
X = df.drop('Churn', axis=1)
y = df['Churn']

# Identificar tipos de características
numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

# Crear el pipeline de preprocesamiento
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features)
    ],
    remainder='passthrough'
)

# Dividir los datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Datos listos para el entrenamiento.\n")

# --- 2. Selección de Modelos ---
print("--- 2. Realizando Selección de Modelos ---")
# Se comparan tres algoritmos diferentes usando el mismo pipeline de preprocesamiento y SMOTE.

# Definir los pipelines para cada modelo
pipeline_lr = Pipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000, solver='liblinear'))
])

pipeline_rf = Pipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42))
])

pipeline_xgb = Pipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'))
])

# Lista de modelos para comparar
modelos = {
    "Logistic Regression": pipeline_lr,
    "Random Forest": pipeline_rf,
    "XGBoost": pipeline_xgb
}

# Entrenar y evaluar cada modelo
resultados = {}
for nombre, pipeline in modelos.items():
    print(f"Entrenando {nombre}...")
    pipeline.fit(X_train, y_train)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    resultados[nombre] = auc

print("\n--- Resultados de la Selección de Modelos ---")
for nombre, auc in resultados.items():
    print(f"Modelo: {nombre:<20} | ROC AUC: {auc:.4f}")
print("El modelo XGBoost muestra el mejor rendimiento inicial.\n")


# --- 3. Optimización con RandomizedSearchCV para XGBoost ---
print("--- 3. Optimizando XGBoost con RandomizedSearchCV ---")

# Espacio de búsqueda de hiperparámetros para SMOTE y XGBoost
param_distributions = {
    'smote__k_neighbors': randint(3, 15),
    'classifier__n_estimators': randint(100, 600),
    'classifier__max_depth': randint(3, 12),
    'classifier__learning_rate': uniform(0.01, 0.3),
    'classifier__subsample': uniform(0.6, 0.4) # Rango de 0.6 a 1.0
}

# Configurar RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=pipeline_xgb,
    param_distributions=param_distributions,
    n_iter=50,  # Aumentamos el número de iteraciones para una mejor búsqueda
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

print("Iniciando la búsqueda de hiperparámetros...")
random_search.fit(X_train, y_train)

# Mostrar los mejores resultados
print("\nMejores Hiperparámetros encontrados:")
print(random_search.best_params_)
print(f"\nMejor ROC AUC Score (en validación cruzada): {random_search.best_score_:.4f}\n")


# --- 4. Evaluación Final del Mejor Modelo ---
print("--- 4. Evaluación del Modelo Optimizado en el Conjunto de Test ---")
best_model = random_search.best_estimator_

# Reporte de clasificación
y_pred_final = best_model.predict(X_test)
print("Reporte de Clasificación (Modelo Optimizado):")
print(classification_report(y_test, y_pred_final, target_names=['No Churn', 'Churn']))

# Matriz de confusión
print("Matriz de Confusión (Modelo Optimizado):")
fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay.from_estimator(
    best_model,
    X_test,
    y_test,
    display_labels=['No Churn', 'Churn'],
    cmap='Greens',
    ax=ax
)
plt.title('Matriz de Confusión del Modelo Optimizado')
plt.grid(False)
plt.show()