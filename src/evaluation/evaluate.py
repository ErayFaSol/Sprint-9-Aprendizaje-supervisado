# src/evaluation/evaluate.py
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def evaluate_model(model, X_val, y_val, model_name):
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    roc_auc = roc_auc_score(y_val, y_pred)

    print(f"Resultados para {model_name}:")
    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")
    print(f"ROC-AUC Score: {roc_auc}")
    return accuracy, f1, roc_auc

