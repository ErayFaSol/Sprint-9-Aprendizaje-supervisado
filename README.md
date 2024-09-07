# Aprendizaje Supervisado

Este proyecto tiene como objetivo desarrollar un modelo de machine learning supervisado para predecir si un cliente dejará el banco próximamente. El modelo busca ayudar a la institución financiera a tomar medidas preventivas para evitar la pérdida de clientes.

## Tecnologías utilizadas
- Python
- pandas
- scikit-learn

## Objetivo
Predecir la probabilidad de abandono de clientes (churn) en un banco, basado en su historial de comportamiento y características demográficas.

## Contexto
Los bancos están interesados en minimizar la tasa de abandono de clientes, conocido como churn, ya que adquirir nuevos clientes es mucho más costoso que retener a los actuales. Este proyecto utiliza técnicas de aprendizaje supervisado para identificar patrones que predicen cuándo un cliente podría abandonar el banco, permitiendo que se tomen acciones proactivas para mejorar la retención.

## Descripción del Proyecto
El proyecto se centra en el análisis de un conjunto de datos que contiene información sobre clientes, como su antigüedad en el banco, comportamiento transaccional y datos demográficos. El objetivo es clasificar si el cliente está en riesgo de abandonar el banco o no.

Pasos del proyecto:
1. **Exploración de datos**: Se inspecciona el conjunto de datos para identificar las características más relevantes y las correlaciones con el abandono de clientes.
2. **Preprocesamiento de datos**: Se limpiaron los datos, se manejaron valores faltantes y se aplicaron técnicas de codificación y escalado para preparar los datos para el modelo.
3. **Entrenamiento del modelo**: Se probaron varios modelos supervisados como regresión logística, árboles de decisión y random forest.
4. **Evaluación del modelo**: Se utilizaron métricas como precisión, recall y el puntaje F1 para determinar el rendimiento del modelo.

## Proceso

### Exploración y Preprocesamiento de Datos
Se realizó un análisis exploratorio utilizando pandas para identificar las principales variables que influyen en el churn, como el saldo de la cuenta, la antigüedad del cliente, y el número de productos contratados.

Durante el preprocesamiento, se eliminaron datos redundantes, se manejaron valores faltantes y se realizaron transformaciones para mejorar la calidad de los datos, utilizando técnicas de codificación y normalización de las variables.

### Entrenamiento del Modelo
Se entrenaron varios modelos supervisados para predecir la tasa de abandono:
- **Regresión Logística**: Para una clasificación binaria sencilla.
- **Árboles de Decisión**: Que permitieron identificar las reglas más importantes.
- **Random Forest**: Un modelo más robusto y preciso para evitar el overfitting.

El **Random Forest** fue el modelo más efectivo, logrando una precisión del 88% en el conjunto de datos de prueba.

### Evaluación del Modelo
Las métricas clave utilizadas para evaluar el modelo fueron:
- **Precisión**: 88%
- **Recall**: 0.86
- **Puntaje F1**: 0.87
- **Matriz de confusión**: Indicó un buen equilibrio entre falsos positivos y falsos negativos.

## Resultados
El modelo de **Random Forest** demostró ser el más eficiente para predecir la tasa de abandono de clientes, alcanzando una precisión del 88%. El análisis permitió a la institución financiera identificar grupos de clientes en riesgo, lo que facilita la creación de campañas de retención específicas.

## Conclusiones
El uso de aprendizaje supervisado permitió desarrollar un modelo eficaz para predecir el churn de clientes. La precisión obtenida es suficiente para implementar soluciones preventivas que pueden reducir la tasa de abandono y aumentar la fidelidad de los clientes.

### Futuras mejoras
- Integrar datos adicionales, como información más detallada sobre el comportamiento financiero de los clientes.
- Experimentar con modelos más avanzados como Gradient Boosting o XGBoost para mejorar la precisión.
- Implementar técnicas de optimización de hiperparámetros para afinar el rendimiento del modelo.

### Enlace al proyecto
[Aprendizaje Supervisado](https://github.com/ErayFaSol/Sprint-9-Aprendizaje-supervisado)
