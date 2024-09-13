# Aprendizaje Supervisado

Este proyecto tiene como objetivo predecir si un cliente de Beta Bank dejará el banco en un futuro cercano, basado en sus comportamientos pasados. El banco busca retener a los clientes existentes, ya que es más rentable que atraer nuevos. Se utilizan modelos de aprendizaje supervisado para maximizar la métrica F1 y evaluar el rendimiento mediante AUC-ROC.

## Tecnologías utilizadas
- Python
- pandas
- scikit-learn

## Instalacion y configuracion

1. Clonar el repositorio
2. Instalar las dependencias
   ```
   pip install -r requirements.txt
   ```
3. Ejecuta el script principal
   ``` 
   python src/main.py
   ```
4. La ejecucion creara un archivo llamado *reporte_final.html*

## Reporte final
El proyecto genera un archivo llamado reporte_final.html con la evaluación de los modelos entrenados, incluyendo las métricas de Exactitud, F1 Score, y AUC-ROC. El archivo se encuentra en la raíz del proyecto y puede ser abierto con cualquier navegador.

## Modelos Utilizados
1. *Random Forest:*  Utilizado para mejorar la precisión en la clasificación de clientes.
2. *Logistic Regression:* Modelo lineal utilizado como base para comparación.

## Metricas de evaluacion
- Accuracy: Medida de la proporción de predicciones correctas.
- F1 Score: Mide el balance entre la precisión y el recall.
- ROC-AUC: Evalúa el desempeño del modelo considerando las tasas de verdaderos positivos y falsos positivos.

## Resultados y Conclusiones
El modelo con mejor rendimiento fue Random Forest, superando a la regresión logística en términos de F1 Score y AUC-ROC. Esto sugiere que Random Forest es el más adecuado para predecir la deserción de clientes.

### Enlace al proyecto
[Aprendizaje Supervisado](https://github.com/ErayFaSol/Sprint-9-Aprendizaje-supervisado)
