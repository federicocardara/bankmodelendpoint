import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import pickle
import numpy as np
import os

# Cargar el dataset
csv = "data/bank-full.csv"
df = pd.read_csv(csv, delimiter=";")

# Excluir las variables no permitidas
df = df.drop(columns=["duration", "contact", "day", "month", "poutcome"])

df['y'] = df['y'].map({'yes': 1, 'no': 0})

# Separar las características y la variable objetivo
X = df.drop(columns=["y"])
y = df["y"]

print(X.columns)

# Definir las columnas categóricas
categorical_cols = X.select_dtypes(include=["object"]).columns

# Crear un transformador para aplicar OneHotEncoding
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(), categorical_cols)
    ])

# Crear un pipeline para cada modelo
models = {
    "Random Forest": Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
    ]),
    "Logistic Regression": Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ]),
    "Support Vector Machine": Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", SVC(kernel='linear'))
    ]),
    "XGBoost": Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'))
    ])
}

# Crear la carpeta 'results' si no existe
os.makedirs('results', exist_ok=True)

# Dividir el dataset en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Probar todos los modelos
best_model = None
best_score = 0
best_model_name = ""

for model_name, model_pipeline in models.items():
    # Entrenar el modelo
    model_pipeline.fit(X_train, y_train)
    
    # Evaluar el modelo
    y_pred = model_pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    accuracy = report["accuracy"]
    print(f"Modelo: {model_name}")
    print(report)
    
    # Comparar con el mejor modelo
    if accuracy > best_score:
        best_score = accuracy
        best_model = model_pipeline
        best_model_name = model_name

# Mostrar el mejor modelo
print(f"\nEl mejor modelo es: {best_model_name} con una precisión de: {best_score:.4f}")

# Guardar el mejor modelo y el preprocesador con Pickle en la carpeta 'results'
with open(f'results/best_model.pkl', 'wb') as model_file:
    pickle.dump(best_model, model_file)

with open(f'results/best_model_encoder.pkl', 'wb') as le_file:
    pickle.dump(preprocessor, le_file)

print("\nModelo entrenado y guardado exitosamente en la carpeta 'results'.")

