# Recomendaciones de negocio - Sprint 5

## Decision

Se recomienda avanzar al Sprint 6 y desplegar el MVP de deteccion de phishing, condicionado a monitoreo de drift, validacion con stakeholder y revision mensual del umbral de decision.

## Evidencia principal

- El modelo final del Sprint 4 alcanza F1 = 0.9479 y AUC-ROC = 0.9918 en test set.
- El criterio de exito del Sprint 1 se cumple: F1 >= 0.85, AUC-ROC >= 0.90, Recall >= 0.85 y Precision >= 0.80.
- La mejora frente al baseline Random Forest es positiva: +0.0087 en F1.
- Con supuestos iniciales de negocio, el umbral operativo debe elegirse maximizando valor esperado, no solo metricas tecnicas.

## Supuestos iniciales

- Beneficio por phishing detectado correctamente (TP): USD 100.
- Costo por sitio legitimo bloqueado (FP): USD -20.
- Costo por phishing no detectado (FN): USD -80.
- Beneficio por sitio legitimo permitido (TN): USD 0.
- Volumen anual referencial: 100,000 URLs evaluadas.

## Acciones recomendadas

1. Validar los costos unitarios con el sponsor antes de fijar el umbral final.
2. Usar el dashboard `dashboard/app.py` en Sprint Review para simular escenarios de costo y volumen.
3. Preparar el despliegue MVP con el paquete `handoff/`.
4. Monitorear recall de phishing como KPI primario de riesgo.
5. Activar revision de drift para variables criticas como `SSLfinal_State`, `URL_of_Anchor`, `web_traffic`, `Prefix_Suffix` y `total_suspicious_count`.

## Riesgos

- Los costos de FP y FN son supuestos hasta que el sponsor los firme.
- El dataset historico puede no cubrir nuevas tecnicas de phishing.
- Algunas variables podrian no estar disponibles o cambiar de definicion en produccion.
- Sin monitoreo, el modelo puede degradarse por data drift o concept drift.
