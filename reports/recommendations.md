# Recomendaciones de negocio - Sprint 5

## Decision

Se recomienda avanzar al Sprint 6 y desplegar el MVP de deteccion de phishing, condicionado a monitoreo de drift, validacion con stakeholder y revision mensual del umbral de decision.

## Evidencia principal

- El modelo final del Sprint 4 alcanza F1 = 0.9479 y AUC-ROC = 0.9918 en test set.
- El criterio de exito del Sprint 1 se cumple: F1 >= 0.85, AUC-ROC >= 0.90, Recall >= 0.85 y Precision >= 0.80.
- La mejora frente al baseline Random Forest es positiva: +0.0087 en F1.
- El KPI economico principal para PB-16 es el ahorro neto esperado: costo sin modelo menos costo con modelo.
- El KPI tecnico de soporte es recall de phishing, porque el falso negativo representa una URL maliciosa no detectada.
- Con supuestos iniciales de negocio, el umbral operativo recomendado es 0.09 y no 0.50: reduce FN de 30 a 3, mantiene recall = 0.9950 y aumenta el ahorro neto anual estimado de USD 24.31M a USD 25.32M.

## Supuestos iniciales

- Costo por phishing no detectado (FN): USD 500.
- Costo por sitio legitimo marcado como phishing (FP): USD 25.
- Costo de revision por alerta (TP + FP): USD 2.
- Costo operativo por URL evaluada: USD 0.50.
- Volumen anual referencial: 100,000 URLs evaluadas.
- Recall minimo operativo para el escenario base: 0.99.

Formula de Business Value:

```text
costo_sin_modelo = phishing_reales * costo_FN
costo_operativo_total = alertas * costo_revision + total_urls * costo_operativo_por_url
costo_con_modelo = FN*costo_FN + FP*costo_FP + alertas*costo_revision + total_urls*costo_operativo_por_url
ahorro_neto = costo_sin_modelo - costo_con_modelo
ROI = ahorro_neto / costo_operativo_total
valor_por_1000_URLs = ahorro_neto / total_urls * 1000
```

## Umbral operativo

El umbral 0.09 se recomienda porque maximiza el ahorro neto bajo el escenario base y cumple el recall minimo de phishing. Frente al umbral 0.50, aumenta falsos positivos de 29 a 90, pero reduce falsos negativos de 30 a 3. En deteccion de phishing esta compensacion es aceptable bajo los supuestos actuales porque el costo de FN es sustancialmente mayor que el costo de FP y revision manual.

La comparacion con 0.12 queda resuelta a favor de 0.09: ambos umbrales mantienen recall alto, pero 0.09 produce mayor ahorro neto en el escenario base al evitar un falso negativo adicional. El umbral 0.12 tiene menor friccion por FP, pero deja pasar 4 phishing frente a 3 en 0.09.

## Acciones recomendadas

1. Validar los costos unitarios con el sponsor antes de fijar el umbral final.
2. Revisar el umbral 0.09 con el sponsor usando ahorro neto, recall de phishing, FP y FN.
3. Preparar el despliegue MVP con el paquete `handoff/`.
4. Monitorear recall de phishing como KPI primario de riesgo y ahorro neto como KPI economico.
5. Activar revision de drift para variables criticas como `SSLfinal_State`, `URL_of_Anchor`, `web_traffic`, `Prefix_Suffix` y `total_suspicious_count`.
6. Usar escenarios conservador, base y severo antes de aprobar cambios de umbral.

## Riesgos

- Los costos de FP y FN son supuestos hasta que el sponsor los firme.
- El ROI depende de costos operativos simplificados y puede variar con integracion, infraestructura, soporte y volumen real.
- El dataset historico puede no cubrir nuevas tecnicas de phishing.
- Algunas variables podrian no estar disponibles o cambiar de definicion en produccion.
- Sin monitoreo, el modelo puede degradarse por data drift o concept drift.
- Accuracy no es suficiente como criterio operativo: puede ocultar falsos negativos criticos en una clase de alto impacto.
