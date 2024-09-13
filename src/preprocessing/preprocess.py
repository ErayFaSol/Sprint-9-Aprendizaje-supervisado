# src/preprocessing/preprocess.py
import pandas as pd

def load_and_clean_data(filepath):
    # Cargar los datos
    data = pd.read_csv(filepath)
    
    # Limpiar datos eliminando columnas irrelevantes y registros nulos
    data_clean = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
    data_clean = data_clean.dropna()
    
    return data_clean

def encode_data(data):
    # One-Hot Encoding para variables categóricas
    data_encoded = pd.get_dummies(data, drop_first=True)
    return data_encoded
