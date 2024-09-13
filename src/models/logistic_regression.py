# src/models/logistic_regression.py
from sklearn.linear_model import LogisticRegression

def train_logistic_regression(X_train, y_train):
    log_reg = LogisticRegression(max_iter=100000, random_state=12345, class_weight='balanced')
    log_reg.fit(X_train, y_train)
    return log_reg
