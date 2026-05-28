# Auditoria tecnica de Business Value

Fecha de auditoria: 2026-05-28  
Alcance: revision exclusiva de Business Value en `notebooks/15_business_value.ipynb`, `notebooks/16_dashboard_prototype.ipynb`, `notebooks/17_findings_report.ipynb`, `reports/business_value_summary.csv`, `reports/business_value_thresholds.csv`, `reports/gain_curve.csv`, `reports/recommendations.md`, `reports/final_report.qmd`, `dashboard/app.py` y `src/evaluation.py`.

## Tabla de auditoria

| Elemento | Evidencia encontrada | Estado | Brecha | Correccion recomendada |
|---|---|---|---|---|
| KPI economico principal medido actualmente | `reports/business_value_summary.csv` registra `valor_test_referencia_usd`, `valor_anual_referencia_usd` y `valor_anual_optimo_usd`. `src/evaluation.py` calcula `annual_value`. | Cumple | El KPI principal es valor anual estimado, pero no se explicita como KPI rector unico en los reportes. | Declarar formalmente `Valor anual estimado USD` como KPI economico actual, con su formula y supuestos. |
| Tipo de KPI economico | El proyecto mide valor economico esperado por matriz costo-beneficio y lo anualiza con `annual_volume=100000`. | Parcial | No es ahorro neto, ROI ni costo evitado; se presenta como valor anual estimado. | Renombrar en documentacion como `valor esperado anual`, o incorporar ahorro neto y ROI si PB-16 exige impacto economico comparativo. |
| Formula exacta actual | `business_value_from_counts()` calcula `tp * benefit_tp + fp * cost_fp + fn * cost_fn + tn * benefit_tn`. `evaluate_threshold()` calcula `value_per_case = value / len(y_true)` y `annual_value = value_per_case * annual_volume`. | Cumple | La formula esta en codigo, pero no aparece completa en `reports/final_report.qmd` ni `reports/recommendations.md`. | Incluir la formula explicita en el reporte final y en el notebook de Business Value. |
| Costos TP, FP, FN y TN | `DEFAULT_BUSINESS_ASSUMPTIONS` define `benefit_tp=100.0`, `cost_fp=-20.0`, `cost_fn=-80.0`, `benefit_tn=0.0`, `annual_volume=100000`. Los mismos supuestos aparecen en `reports/recommendations.md` y `reports/final_report.qmd`. | Cumple | Los costos son supuestos no firmados por sponsor, segun `17_findings_report.ipynb` y recomendaciones. | Mantenerlos como escenario base y exigir validacion de sponsor antes de fijar el umbral operativo. |
| Diferenciacion entre falso negativo y falso positivo | `src/evaluation.py` trata phishing como clase positiva. La matriz usa `labels=[LEGITIMATE_LABEL, PHISHING_LABEL]`, por lo que FP equivale a legitimo bloqueado y FN a phishing no detectado. | Cumple | La diferencia tecnica es correcta, pero el dashboard no muestra estos errores ni su costo. | Mostrar FP, FN y su impacto economico en dashboard y reporte ejecutivo. |
| Costo sin modelo | No se encontro calculo de escenario sin modelo en notebooks, CSV, reportes ni `src/evaluation.py`. | No cumple | No existe baseline economico del tipo "permitir todas las URLs" o "bloquear todas". | Calcular `valor_sin_modelo` o `costo_sin_modelo`, por ejemplo con todos los phishing como FN y todos los legitimos como TN. |
| Costo con modelo | `evaluate_threshold()` calcula `value_test`, `value_per_case` y `annual_value` para cada umbral. `business_value_thresholds.csv` contiene esos valores por threshold. | Parcial | Se calcula valor con modelo, no costo total con modelo separado ni costo operativo del sistema. | Reportar `valor_con_modelo` y, si aplica, `costo_operativo_modelo`. |
| Ahorro neto | No se encontro campo `ahorro`, `net_savings`, `valor_incremental` ni resta contra escenario sin modelo. | No cumple | Sin escenario sin modelo no puede calcularse ahorro neto. | Agregar `ahorro_neto = valor_con_modelo - valor_sin_modelo`. |
| ROI | No se encontro calculo de ROI en notebooks, reportes, CSV, dashboard ni `src/evaluation.py`. | No cumple | Falta costo de inversion/operacion y formula de retorno. | Definir costo operativo del sistema y calcular `ROI = ahorro_neto / costo_operativo_modelo`. |
| Ganancia por cada 1000 URLs | `src/evaluation.py` calcula `value_per_case`, pero no lo escala a 1000 URLs. `gain_curve.csv` contiene `contacted_share`, `captured_share` y `lift`, no valor monetario por 1000 URLs. | Parcial | Existe base para calcularlo, pero no se reporta `value_per_1000_urls`. | Agregar `valor_por_1000_urls = value_per_case * 1000` y reportarlo junto con el umbral. |
| Evaluacion de umbrales de decision | `threshold_search()` evalua umbrales de 0.05 a 0.95 con paso 0.01. `business_value_thresholds.csv` contiene resultados por umbral. | Cumple | No se documenta una regla de desempate entre umbrales con mismo valor. | Agregar criterio de desempate: menor FN, menor FP, mayor precision, mayor recall o politica de riesgo aprobada. |
| Justificacion del umbral optimo 0.09 vs 0.5 | `business_value_summary.csv` registra `umbral_optimo=0.09`, `valor_anual_optimo_usd=4962393.16` y `valor_anual_referencia_usd=4651282.05`. `business_value_thresholds.csv` muestra 0.09 con 476 TN, 90 FP, 3 FN, 601 TP y 0.5 con 537 TN, 29 FP, 30 FN, 574 TP. | Parcial | La justificacion es implicita: maximiza valor esperado frente a 0.5, pero hay empate con 0.12, tambien con `annual_value=4962393.16`. No se explica por que se elige 0.09 sobre 0.12. | Documentar que 0.09 reduce FN a 3 frente a 4 en 0.12, a costa de mas FP, o elegir 0.12 si se prioriza menor friccion por FP. |
| Uso del umbral optimo en `dashboard/app.py` | `dashboard/app.py` define `predict_pipeline(df, threshold=0.5)` y el slider tiene `value=0.50`. | No cumple | El dashboard sigue usando 0.5 como default, no el umbral optimo 0.09. | Cambiar default operativo a 0.09 o cargarlo desde `business_value_summary.csv`; mostrar 0.5 solo como referencia. |
| KPIs tecnicos en dashboard | `dashboard/app.py` muestra `Total registros`, `Phishing detectados` y `Sitios legitimos`. No muestra F1, AUC-ROC, Recall ni Precision. | No cumple | `16_dashboard_prototype.ipynb` define KPIs tecnicos, pero la app no los implementa. | Incorporar F1, AUC-ROC, Recall y Precision del modelo final o del lote si el CSV trae `Result`. |
| KPIs economicos en dashboard | `dashboard/app.py` no calcula valor anual, valor por caso, costos TP/FP/FN/TN, ahorro ni ROI. | No cumple | El notebook declara valor anual estimado como KPI, pero la app no lo presenta. | Agregar matriz costo-beneficio editable y calcular valor estimado del lote, valor por 1000 URLs y valor anual. |
| Ajuste de costos TP/FP/FN/TN en dashboard | `16_dashboard_prototype.ipynb` marca "Permite ajustar costos TP/FP/FN/TN" como cubierto, pero `dashboard/app.py` solo permite ajustar el umbral. | No cumple | Hay discrepancia entre checklist del notebook y la implementacion real. | Agregar inputs numericos para TP, FP, FN, TN y volumen anual, o corregir el checklist. |
| Matriz de confusion en dashboard | `16_dashboard_prototype.ipynb` marca matriz de confusion como cubierta. `dashboard/app.py` no calcula ni muestra matriz de confusion. | No cumple | La app no usa `Result` para evaluar predicciones si el CSV trae etiquetas reales. | Mostrar matriz de confusion cuando `Result` exista; ocultarla cuando no haya etiqueta real. |
| Escenarios conservador, base y severo | No se encontraron escenarios conservador/base/severo en notebooks, reportes, CSV, dashboard ni `src/evaluation.py`. | No cumple | El proyecto usa un unico set de supuestos. | Definir tres escenarios con costos distintos de FN/FP/TP y comparar umbral, valor anual, ahorro neto y ROI por escenario. |
| Curva de ganancia | `gain_curve()` genera `contacted_share`, `captured_share` y `lift`; `gain_curve.csv` contiene esos campos. | Parcial | Es una curva de ganancia/lift tecnica, no una curva de ganancia economica monetaria. | Complementar con curva de valor acumulado o ahorro acumulado por porcentaje de URLs revisadas. |
| PB-16: matriz costo-beneficio | `15_business_value.ipynb` implementa matriz costo-beneficio con TP, FP, FN, TN y busqueda de umbral. | Parcial | Falta costo sin modelo, ahorro neto, ROI, ganancia por 1000 URLs, escenarios y regla de desempate. | Completar PB-16 con baseline economico, metricas incrementales y analisis de sensibilidad. |
| PB-17: dashboard prototype | `16_dashboard_prototype.ipynb` define objetivo, wireframe, KPIs y checklist; `dashboard/app.py` existe y predice desde CSV. | Parcial | La implementacion no refleja varios elementos del prototipo: KPIs tecnicos, KPIs economicos, costos editables, matriz de confusion y explicabilidad. | Alinear `dashboard/app.py` con el wireframe minimo de PB-17 o ajustar el alcance declarado. |
| Reporte ejecutivo de Business Value | `reports/final_report.qmd` describe resultados tecnicos, supuestos costo-beneficio y recomendacion de umbral por valor esperado. | Parcial | No incluye formula completa, ahorro neto, ROI, escenarios, valor por 1000 URLs ni empate 0.09/0.12. | Ampliar seccion de valor de negocio con calculos comparativos y sensibilidad. |

## Hallazgos cuantitativos clave

- Umbral de referencia 0.50: `TN=537`, `FP=29`, `FN=30`, `TP=574`, `value_test=54420.0`, `annual_value=4651282.05`.
- Umbral 0.09: `TN=476`, `FP=90`, `FN=3`, `TP=601`, `value_test=58060.0`, `annual_value=4962393.16`.
- Umbral 0.12: `TN=485`, `FP=81`, `FN=4`, `TP=600`, `value_test=58060.0`, `annual_value=4962393.16`.
- Mejora de valor anual de 0.09 frente a 0.50: `311111.11 USD` bajo los supuestos actuales.
- La seleccion de 0.09 es razonable si se prioriza minimizar FN, pero no esta formalmente justificada porque 0.12 empata en valor anual y reduce FP.

## KPI economico principal recomendado para phishing

El KPI economico principal recomendado es `Ahorro neto anual esperado frente a no usar modelo`, complementado por `valor neto por 1000 URLs evaluadas`.

La razon es que en phishing el valor no debe medirse solo como beneficio bruto por TP. Debe compararse contra el escenario operativo sin modelo, porque el costo critico es permitir phishing no detectado. El KPI debe hacer visible el intercambio entre reducir FN y aumentar FP.

## Formula recomendada

Definiciones:

- `P`: cantidad de URLs phishing reales.
- `N`: cantidad de URLs legitimas reales.
- `TP`: phishing detectado correctamente.
- `FP`: legitimo bloqueado incorrectamente.
- `FN`: phishing no detectado.
- `TN`: legitimo permitido correctamente.
- `B_TP`: beneficio por TP.
- `C_FP`: costo por FP.
- `C_FN`: costo por FN.
- `B_TN`: beneficio por TN.
- `C_op`: costo anual de operacion del sistema.

Formula actual de valor con modelo:

```text
valor_con_modelo = TP * B_TP + FP * C_FP + FN * C_FN + TN * B_TN
valor_por_caso = valor_con_modelo / total_urls
valor_anual = valor_por_caso * volumen_anual
```

Formula recomendada para evaluacion economica:

```text
valor_sin_modelo = 0 * B_TP + 0 * C_FP + P * C_FN + N * B_TN
valor_con_modelo = TP * B_TP + FP * C_FP + FN * C_FN + TN * B_TN
ahorro_neto = valor_con_modelo - valor_sin_modelo
ahorro_neto_anual = (ahorro_neto / total_urls) * volumen_anual
valor_por_1000_urls = (valor_con_modelo / total_urls) * 1000
ahorro_por_1000_urls = (ahorro_neto / total_urls) * 1000
ROI = (ahorro_neto_anual - C_op) / C_op
```

Si no existe `C_op`, el ROI debe marcarse como no calculable y no debe reportarse como cumplido.

## Cambios minimos necesarios para corregir Sprint 5

1. Completar `src/evaluation.py` con funciones para `valor_sin_modelo`, `ahorro_neto`, `valor_por_1000_urls`, `ahorro_por_1000_urls` y ROI condicionado a costo operativo.
2. Actualizar `15_business_value.ipynb` para generar esos campos en los CSV y resolver el empate 0.09 vs 0.12 con una regla de riesgo documentada.
3. Agregar escenarios conservador, base y severo con distintas matrices TP/FP/FN/TN y comparar umbral optimo por escenario.
4. Alinear `dashboard/app.py` con PB-17: default de umbral 0.09, KPIs tecnicos, KPIs economicos, costos editables, matriz de confusion si existe `Result` y valor estimado del lote.
5. Corregir `16_dashboard_prototype.ipynb` o la app para que el checklist refleje capacidades reales.
6. Ampliar `reports/final_report.qmd` y `reports/recommendations.md` con formula completa, supuestos, ahorro neto, valor por 1000 URLs, escenarios y decision formal del umbral operativo.
