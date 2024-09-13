# src/models/random_forest.py
from sklearn.ensemble import RandomForestClassifier

def train_random_forest(X_train, y_train):
    rf = RandomForestClassifier(n_estimators=100, random_state=12345, class_weight='balanced')
    rf.fit(X_train, y_train)
    return rf
