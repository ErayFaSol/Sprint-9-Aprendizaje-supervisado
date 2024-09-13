# src/main.py
from preprocessing.preprocess import load_and_clean_data, encode_data
from models.random_forest import train_random_forest
from models.logistic_regression import train_logistic_regression
from evaluation.evaluate import evaluate_model
from sklearn.model_selection import train_test_split
from utils.generate_report import save_report_as_html
# Cargar y limpiar datos
data = load_and_clean_data('datasets/Churn.csv')
data_encoded = encode_data(data)

# Dividir datos en entrenamiento, validación y prueba
X = data_encoded.drop('Exited', axis=1)
y = data_encoded['Exited']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12345, stratify=y)

# Entrenar modelos
rf_model = train_random_forest(X_train, y_train)
lr_model = train_logistic_regression(X_train, y_train)

# Evaluar modelos
print("\nEvaluación del modelo Random Forest:")
rf_accuracy, rf_f1, rf_roc_auc = evaluate_model(rf_model, X_test, y_test, "Random Forest")

print("\nEvaluación del modelo Logistic Regression:")
lr_accuracy, lr_f1, lr_roc_auc = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")

# Generar reporte en Html
save_report_as_html(rf_accuracy, rf_f1, rf_roc_auc, lr_accuracy, lr_f1, lr_roc_auc)