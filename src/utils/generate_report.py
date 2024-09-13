def save_report_as_html(accuracy_rf, f1_rf, roc_auc_rf, accuracy_lr, f1_lr, roc_auc_lr):
    with open("reporte_final.html", "w") as file:
        # Título del reporte
        file.write(f"<h1>Reporte Final de Evaluación de Modelos</h1>")
        
        # Resultados de Random Forest
        file.write(f"<h2>Resultados de Random Forest</h2>")
        file.write(f"<p>Exactitud: {accuracy_rf}</p>")
        file.write(f"<p>F1 Score: {f1_rf}</p>")
        file.write(f"<p>ROC-AUC Score: {roc_auc_rf}</p>")
        #file.write(f"<p>Mejores Hiperparámetros: {best_params_rf}</p>")
        
        # Resultados de Logistic Regression
        file.write(f"<h2>Resultados de Logistic Regression</h2>")
        file.write(f"<p>Exactitud: {accuracy_lr}</p>")
        file.write(f"<p>F1 Score: {f1_lr}</p>")
        file.write(f"<p>ROC-AUC Score: {roc_auc_lr}</p>")
        #file.write(f"<p>Mejores Hiperparámetros: {best_params_lr}</p>")
        
        # Conclusiones
        file.write(f"<h2>Conclusiones</h2>")
        mejor_modelo = 'Random Forest' if f1_rf > f1_lr else 'Logistic Regression'
        file.write(f"<p>El modelo con mejor rendimiento en términos de F1 Score fue: {mejor_modelo}</p>")
