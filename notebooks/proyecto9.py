# %% [markdown]
# # Proyecto 9: Aprendizaje supervisado

# %% [markdown]
# ## Descripcion del proyecto

# %% [markdown]
# Los clientes de Beta Bank se están yendo, cada mes, poco a poco. Los banqueros descubrieron que es más barato salvar a los clientes existentes que atraer nuevos.
# 
# Necesitamos predecir si un cliente dejará el banco pronto. Tú tienes los datos sobre el comportamiento pasado de los clientes y la terminación de contratos con el banco.
# Crea un modelo con el máximo valor F1 posible. 
# 
# Para aprobar la revisión, necesitas un valor F1 de al menos 0.59. Verifica F1 para el conjunto de prueba. 
# Además, debes medir la métrica AUC-ROC y compararla con el valor F1.

# %% [markdown]
# ### Instrucciones del proyecto 

# %% [markdown]
# - Descarga y prepara los datos.  Explica el procedimiento.
# 
# - Examina el equilibrio de clases. Entrena el modelo sin tener en cuenta el desequilibrio. Describe brevemente tus hallazgos.
# 
# - Mejora la calidad del modelo. Asegúrate de utilizar al menos dos enfoques para corregir el desequilibrio de clases. Utiliza conjuntos de entrenamiento y validación para encontrar el mejor modelo y el mejor conjunto de parámetros. Entrena diferentes modelos en los conjuntos de entrenamiento y validación. Encuentra el mejor. Describe brevemente tus hallazgos.
# 
# - Realiza la prueba final.

# %% [markdown]
# ### Descripcion de los datos

# %% [markdown]
# Puedes encontrar los datos en el archivo  /datasets/Churn.csv file. Descarga el conjunto de datos.
# 
# Características
# 
# - RowNumber: índice de cadena de datos
# - CustomerId: identificador de cliente único
# - Surname: apellido
# - CreditScore: valor de crédito
# - Geography: país de residencia
# - Gender: sexo
# - Age: edad
# - Tenure: período durante el cual ha madurado el depósito a plazo fijo de un cliente (años)
# - Balance: saldo de la cuenta
# - NumOfProducts: número de productos bancarios utilizados por el cliente
# - HasCrCard: el cliente tiene una tarjeta de crédito (1 - sí; 0 - no)
# - IsActiveMember: actividad del cliente (1 - sí; 0 - no)
# - EstimatedSalary: salario estimado
# 
# Objetivo
# - Exited: El cliente se ha ido (1 - sí; 0 - no)

# %% [markdown]
# ## Evaluacion del proyecto

# %% [markdown]
# Hemos definido los criterios de evaluación para el proyecto. Lee esto con atención antes de pasar al ejercicio.
# 
# Esto es lo que los revisores buscarán cuando evalúen tu proyecto:
# 
# - ¿Cómo preparaste los datos para el entrenamiento? ¿Procesaste todos los tipos de características?
# - ¿Explicaste los pasos de preprocesamiento lo suficientemente bien?
# - ¿Cómo investigaste el equilibrio de clases?
# - ¿Estudiaste el modelo sin tener en cuenta el desequilibrio de clases?
# - ¿Qué descubriste sobre la investigación del ejercicio?
# - ¿Dividiste correctamente los datos en conjuntos?
# - ¿Cómo trabajaste con el desequilibrio de clases?
# - ¿Utilizaste al menos dos técnicas para corregir el desequilibrio?
# - ¿Realizaste correctamente el entrenamiento, la validación y las pruebas finales del modelo?
# - ¿Qué tan alto es tu valor F1?
# - ¿Examinaste los valores AUC-ROC?
# - ¿Mantuviste la estructura del proyecto y el código limpio?
# Ya tienes las hojas informativas y los resúmenes de capítulos, tienes todo para continuar con el proyecto.
# 
# ¡Buena suerte!

# %% [markdown]
# ## Preparacion de datos

# %%
#Importacion de librerias 
# Importar librerias necesarias para el proyecto
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from IPython.display import HTML

# %%
# Cargar los datos
data = pd.read_csv("datasets/Churn.csv")

# Por ejemplo, podrías considerar eliminar 'RowNumber', 'CustomerId', y 'Surname' ya que probablemente no sean relevantes para la predicción
data_clean = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

# Mostrar datos relevantes del dataframe
display(HTML('<hr>'))
display(HTML('<h1> Clientes de Beta-bank '))
display(data_clean.head())
display(data_clean.info())
display(data_clean.describe())
display(HTML('<hr>'))

# Verificacion de datos nulos 
nulos = data_clean.isnull().sum()
display(HTML('<h2> Verificacion de valores nulos'))
display(nulos)

# Verificación de valores duplicados
duplicados = data_clean.duplicated().sum()
display(HTML('<h2> Verificación de valores duplicados </h2>'))
display(f"Total de valores duplicados: {duplicados}")
# Eliminamos valores duplicados
data_clean = data_clean.dropna()

display(HTML('<hr>'))

comentario = """ 
<h2> Comentario sobre la exploracion inicial del archivo </h2>
<p> Al realizar la preparacion de los datos para poder seguir avanzando en el modelo nos dimos que cuanta que este no presenta valores duplicados, pero si nulos, para este caso se decidio descartarlos
dichos datos nulos abarcan 9% de los datos.
<p> Tambien se econtro columnas categoricas las cuales requieren un manejo para poder funcionar en los modelos del machine learning

"""
display(HTML(comentario))


# %% [markdown]
# ## Balance de clases

# %%
# Examinar balance de clases
class_balance = data_clean['Exited'].value_counts(normalize=True)
print(class_balance)

# %% [markdown]
# Notamos que hay un claro desequilibrio de clases dentro del tu conjunto de datos. La clase mayoritaria (clientes que no se han dado de baja) constituye una gran mayoría de los casos, mientras que la clase minoritaria (clientes que se han dado de baja) representa una porción mucho menor. Esto significa que si un modelo eligiera la clase más común (prediciendo que todos los clientes se quedarán en el banco), alcanzaría una precisión del 79.6% sin haber aprendido realmente a distinguir entre las características que contribuyen a la deserción de clientes.

# %% [markdown]
# ## Modelado sin tener en cuenta el desequilibrio de clases

# %%
# One-Hot Encoding para variables categóricas
data_encoded = pd.get_dummies(data_clean, drop_first=True)

# División de los datos en características y objetivo
X = data_encoded.drop('Exited', axis=1)
y = data_encoded['Exited']

# División en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12345, stratify=y)

# Dividimos el conjunto de entrenamiento en subconjuntos de entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=12345, stratify=y_train)


# %%
# Entrenar un modelo de Regresión Logística
log_reg = LogisticRegression(max_iter=100000, random_state=12345, class_weight= 'balanced')
log_reg.fit(X_train, y_train)

# Evaluacion 
y_pred_val = log_reg.predict(X_val)

accuracy_log_reg = accuracy_score(y_val, y_pred_val)
f1_log_reg = f1_score(y_val, y_pred_val)
roc_auc_log_reg = roc_auc_score(y_val, y_pred_val)

print(f"Accuracy (Regresión Logística): {accuracy_log_reg}")
print(f"F1 Score (Regresión Logística): {f1_log_reg}")
print(f"ROC-AUC Score (Regresión Logística): {roc_auc_log_reg}")

# %%
# Entrenar un modelo de Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=12345, class_weight= 'balanced')
rf.fit(X_train, y_train)

# Predicción y evaluación con el conjunto de prueba
y_pred_rf = rf.predict(X_val)

# Calcular y mostrar el puntaje F1
accuracy_rf = accuracy_score(y_val, y_pred_rf)
f1_rf = f1_score(y_val, y_pred_rf)
roc_auc_rf = roc_auc_score(y_val, y_pred_rf)

print(f"Accuracy (Random Forest): {accuracy_rf}")
print(f"F1 Score (Random Forest): {f1_rf}")
print(f"ROC-AUC Score (Random Forest): {roc_auc_rf}")


# %% [markdown]
# ## Interpretacion de los resultados

# %% [markdown]
# Los resultados indican que el modelo de Random Forest manejan mejor el problema del desequilibrio entre clases lo cual nos sugiere que este es mas adecuado para predecir la desercion de clientes en el banco.
# Ahora con respecto a las evaluaciones:
# 
# Accuracy(Precisión): La precision no es una metrica tan relevante para evaluar el rendimiento en conjuntos de datos desequilibrados, sim embargo tener una precision mas alta en un modelo que en otro sugiere que el mas alto es capaz de hacer mejores predicciones generales, para este caso notamos que la precision de Random Forest es de 0.86 que lo deja en posicion alta comparandolo con el modelo de Regresion logistica
# 
# F1 Score: Un valor alto en F1 nos indica que el modelo tiene un mejor balance entre precision y sensibilidad lo cual para este contexto es importando porque nos preocupa la clase minoritaria. En este caso el modelo de Random Forest nos presenta una diferencia considerable con respecto al modelo de Regresion Logistica, para Random Forest la puntuacion es de 0.60 y Regresion Logistica 0.34.
# 
# ROC-AUC Score: Para esta metrica un valor elevado suguiere que un modelo tiene mejor tasa de verdaderos positivos mientras mantiene baja la tasa de falsos positivos. El modelo de Regresion Logistica obtiene un valor de 0.60 mientras que para Random Forest este valor es mas elevado con 0.73
# 
# En conclusion El modelo Random Forest es el que cumple los criterios minimos especificado para la revision del proyecto; Este modelo se puede seguir Mejorando ajustando los hiperparametros.

# %% [markdown]
# ## Mejorando el modelo Random Forest
# 

# %%
# Ajustar Hiperparametros 
# Definir grilla 
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],  
    'min_samples_split': [2, 5, 10], 
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False],
}


rf = RandomForestClassifier(random_state=12345, class_weight= 'balanced')

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='f1')

grid_search.fit(X_train, y_train)

# %%
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print("Mejores hiperparámetros:", best_params)

# Evaluación del mejor modelo con el conjunto de prueba final
y_pred_test = best_model.predict(X_test)

# Calcular métricas de rendimiento utilizando y_test
accuracy_test = accuracy_score(y_test, y_pred_test)  
f1_test = f1_score(y_test, y_pred_test)              
roc_auc_test = roc_auc_score(y_test, y_pred_test)  

# Imprimir los resultados de la prueba final
print(f"Accuracy en el conjunto de prueba: {accuracy_test}")
print(f"F1 Score en el conjunto de prueba: {f1_test}")
print(f"ROC-AUC Score en el conjunto de prueba: {roc_auc_test}")


# %% [markdown]
# ## Resultados 

# %% [markdown]
# Tras haber evaluado dos modelos los cuales fueron Random Forest y Logistic Regression se considero Random Forest como el modelo mas optimo para este caso y se siguio trabajando sobre este al punto que se realizaron varios ajustes en los hiperparametros y obtuvimos el mejor modelo y podemos notar que aumenta sus puntuaciones de Precision, F1 y ROC AUC.


