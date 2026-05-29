# Voila MVP Dashboard

Ejecutar desde la raiz del repositorio:

```bash
voila voila_app/phishing_mvp_dashboard.ipynb --host 0.0.0.0 --port 8501
```

El notebook usa el kernel `phishing_s1`, carga `models/final_model.pkl`, permite usar `data/processed/test.csv` o cargar un CSV, y muestra predicciones, Business Value, resumen MLflow y estado MongoDB.
