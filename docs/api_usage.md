# API REST Sprint 6 - Phishing Detection

Esta API expone el modelo final del proyecto como servicio REST para que el dashboard Streamlit pueda operar en modo API durante el MVP de Sprint 6.

## Ejecutar localmente

Desde la raiz del repositorio:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Variables opcionales:

```bash
MODEL_PATH=models/final_model.pkl
DEFAULT_THRESHOLD=0.09
```

El umbral por defecto se lee primero desde `reports/business_value_summary.csv`. Si no esta disponible, la API puede usar `DEFAULT_THRESHOLD`; si tampoco esta configurado, usa `0.5`.

## GET /health

Ejemplo:

```bash
curl http://127.0.0.1:8000/health
```

Respuesta esperada:

```json
{
  "project": "phishing-detection",
  "api_status": "ok",
  "model_status": "loaded",
  "model_path": "models/final_model.pkl",
  "model_error": null,
  "expected_features_count": 37
}
```

Si el modelo no existe o no puede cargarse por dependencias faltantes, `model_status` indicara `missing` o `error`.

## POST /predict

Formato de entrada:

```json
{
  "records": [
    {
      "having_IP_Address": -1,
      "URL_Length": 1,
      "Shortining_Service": 1,
      "having_At_Symbol": 1,
      "double_slash_redirecting": 1,
      "Prefix_Suffix": -1,
      "having_Sub_Domain": 1,
      "SSLfinal_State": -1,
      "Domain_registeration_length": 1,
      "Favicon": 1,
      "port": 1,
      "HTTPS_token": 1,
      "Request_URL": 1,
      "URL_of_Anchor": 0,
      "Links_in_tags": -1,
      "SFH": -1,
      "Submitting_to_email": 1,
      "Abnormal_URL": 1,
      "Redirect": 0,
      "on_mouseover": 1,
      "RightClick": 1,
      "popUpWidnow": 1,
      "Iframe": 1,
      "age_of_domain": 1,
      "DNSRecord": 1,
      "web_traffic": 1,
      "Page_Rank": -1,
      "Google_Index": -1,
      "Links_pointing_to_page": 0,
      "Statistical_report": 1,
      "url_risk_score": 3,
      "security_score": 4,
      "total_suspicious_count": 8,
      "total_legitimate_count": 19,
      "net_signal_ratio": 11,
      "ssl_traffic_interaction": -1,
      "content_risk_score": 2
    }
  ],
  "threshold": 0.09
}
```

Ejemplo con `curl`:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"records":[{"having_IP_Address":-1,"URL_Length":1,"Shortining_Service":1,"having_At_Symbol":1,"double_slash_redirecting":1,"Prefix_Suffix":-1,"having_Sub_Domain":1,"SSLfinal_State":-1,"Domain_registeration_length":1,"Favicon":1,"port":1,"HTTPS_token":1,"Request_URL":1,"URL_of_Anchor":0,"Links_in_tags":-1,"SFH":-1,"Submitting_to_email":1,"Abnormal_URL":1,"Redirect":0,"on_mouseover":1,"RightClick":1,"popUpWidnow":1,"Iframe":1,"age_of_domain":1,"DNSRecord":1,"web_traffic":1,"Page_Rank":-1,"Google_Index":-1,"Links_pointing_to_page":0,"Statistical_report":1,"url_risk_score":3,"security_score":4,"total_suspicious_count":8,"total_legitimate_count":19,"net_signal_ratio":11,"ssl_traffic_interaction":-1,"content_risk_score":2}],"threshold":0.09}'
```

Respuesta:

```json
{
  "model_status": "loaded",
  "threshold": 0.09,
  "n_records": 1,
  "predictions": [
    {
      "prediction": 1,
      "prediction_label": "phishing",
      "phishing_probability": 0.93
    }
  ]
}
```

## Integracion con dashboard

El dashboard Streamlit consumira esta API en modo API usando:

- `GET {API_BASE_URL}/health`
- `POST {API_BASE_URL}/predict`

El modo local se mantiene como respaldo operativo mientras el despliegue de Sprint 6 se estabiliza.
