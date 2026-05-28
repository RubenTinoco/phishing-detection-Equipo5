# Auditoria tecnica de herramientas, trazabilidad y despliegue

Fecha de auditoria: 2026-05-28  
Alcance: revision de evidencias de herramientas, trazabilidad experimental, dashboard, API, contenedorizacion y despliegue. No se modificaron codigo, notebooks ni archivos existentes.

| Herramienta | Evidencia encontrada | Estado | Brecha | Correccion recomendada | Sprint relacionado |
|---|---|---|---|---|---|
| Streamlit | Existe `dashboard/app.py` y contiene una aplicacion Streamlit con carga de CSV, prediccion, slider de umbral, metricas y descarga de resultados. | Cumple | La aplicacion existe como prototipo funcional. | Mantener `dashboard/app.py` como interfaz principal del Sprint 5 y alinear su configuracion con los artefactos finales. | Sprint 5 |
| Streamlit | `dashboard/app.py` importa `streamlit`, pero `streamlit` no aparece en `requirements.txt` ni en `environment.yml`. | No cumple | El ambiente reproducible no instala la dependencia necesaria para ejecutar el dashboard. | Agregar `streamlit` a `requirements.txt` y `environment.yml` con version compatible. | Sprint 5 |
| Streamlit | `notebooks/16_dashboard_prototype.ipynb` documenta el comando `streamlit run dashboard/app.py`. `README.md` no documenta el comando. | Parcial | El comando existe en un notebook, pero no esta centralizado en documentacion operativa del proyecto. | Documentar el comando desde la raiz del repositorio en `README.md` o en una guia de ejecucion. | Sprint 5 |
| Streamlit | `dashboard/app.py` carga el modelo localmente con `joblib.load` desde `models/final_model.pkl`. No se observa consumo de API REST. | Parcial | El dashboard opera acoplado al artefacto local y no valida una integracion dashboard/API. | Mantener modo local para desarrollo y agregar modo API mediante endpoint `/predict` en Sprint 6. | Sprint 5-6 |
| Voila | No se encontro notebook preparado para publicacion con Voila ni configuracion especifica de widgets o renderizado Voila. | No cumple | No existe alternativa ejecutable de dashboard basada en notebook. | Crear un notebook ligero de dashboard ejecutivo compatible con Voila si se requiere alternativa a Streamlit. | Sprint 5 |
| Voila | `voila` no aparece en `requirements.txt` ni en `environment.yml`. | No cumple | El ambiente no permite ejecutar notebooks como aplicacion Voila. | Agregar `voila` a dependencias solo si se adopta formalmente como alternativa. | Sprint 5 |
| Voila | No se encontro comando documentado como `voila notebooks/...ipynb`. | No cumple | Falta instruccion de ejecucion reproducible. | Documentar comando de ejecucion y puerto esperado para el notebook Voila. | Sprint 5 |
| Voila | La arquitectura actual puede admitir Voila como alternativa, porque ya existen notebooks de Business Value y prototipo de dashboard. | Parcial | La alternativa es factible, pero no implementada ni validada. | Adaptar `16_dashboard_prototype.ipynb` o crear un notebook dedicado sin dependencias innecesarias ni celdas exploratorias. | Sprint 5 |
| MLflow | No se encontro `mlflow_config.py`. | No cumple | No existe modulo de configuracion para tracking, experimentos o registro de modelos. | Crear `src/mlflow_config.py` con tracking URI, nombre de experimento y utilidades de inicializacion. | Sprint 3-4 |
| MLflow | `mlflow` no aparece en `requirements.txt` ni en `environment.yml`; tampoco se encontro carpeta `mlruns/`. | No cumple | MLflow no esta instalado ni hay evidencia de tracking local. | Agregar `mlflow` a dependencias y definir si el tracking sera local (`mlruns/`) o remoto. | Sprint 3-4 |
| MLflow | Los notebooks de Sprint 3 y Sprint 4 guardan trazabilidad alternativa en CSV, modelos `.pkl` y figuras; por ejemplo `models/experiments_log.csv`, `baseline_comparison.csv`, `ensemble_comparison.csv` y artefactos `tuned_*.pkl`. | Parcial | Hay trazabilidad manual, pero no registro MLflow de parametros, metricas, artefactos y modelos. | Envolver entrenamientos con `mlflow.start_run`, registrar parametros, metricas, artefactos, modelos y version del modelo final. | Sprint 3-4 |
| MLflow | Sprint 3 entrena baselines y Sprint 4 ejecuta tuning, ensambles y validacion final, pero no se observan llamadas a `mlflow.log_param`, `mlflow.log_metric`, `mlflow.log_artifact` o `mlflow.sklearn.log_model`. | No cumple | No hay trazabilidad experimental estandarizada ni comparacion navegable de runs. | Integrar MLflow en baseline, tuning y modelo final; registrar el candidato final como modelo versionado. | Sprint 3-4 |
| MongoDB | No se encontro `mongodb_utils.py`. | No cumple | No existe capa de persistencia para historicos consultables. | Crear `src/mongodb_utils.py` con conexion, validacion basica y operaciones de escritura/lectura. | Sprint 4-5 |
| MongoDB | No se encontro uso de `pymongo`, `MongoClient`, string de conexion ni variables de entorno asociadas. | No cumple | No hay configuracion de base de datos ni mecanismo de credenciales. | Agregar `pymongo` y variables como `MONGODB_URI`, `MONGODB_DB`, `MONGODB_COLLECTION_*`. | Sprint 4-5 |
| MongoDB | No se observa persistencia en MongoDB de experimentos, metricas, predicciones ni feedback. La evidencia actual se concentra en CSV, reportes y artefactos locales. | No cumple | El historico no es consultable mediante una base documental ni esta preparado para auditoria operativa. | Guardar experimentos, metricas finales, predicciones del dashboard y feedback de usuarios en colecciones separadas. | Sprint 4-5 |
| API REST | No se encontro carpeta `api/`. | No cumple | No existe servicio REST separado del dashboard. | Crear `api/main.py` o estructura equivalente para servir inferencia. | Sprint 6 |
| API REST | No se encontro uso de FastAPI, Flask, `uvicorn` o `gunicorn`. | No cumple | Falta framework de servicio web y dependencias asociadas. | Implementar FastAPI o Flask y agregar dependencias al ambiente reproducible. | Sprint 6 |
| API REST | No se encontro endpoint `/predict`. | No cumple | No hay contrato de inferencia para integracion con dashboard u otros consumidores. | Implementar `/predict` con carga del modelo final, preprocesamiento consistente y respuesta con clase, probabilidad y umbral usado. | Sprint 6 |
| API REST | No se encontro endpoint `/health`. | No cumple | No hay mecanismo minimo para monitoreo o verificacion de disponibilidad. | Implementar `/health` con estado de servicio, disponibilidad del modelo y version del artefacto. | Sprint 6 |
| API REST | No se encontro validacion formal del esquema de entrada para una API. | No cumple | Las entradas no tienen contrato tipado ni validacion de columnas requeridas a nivel de servicio. | Definir esquema Pydantic o validacion equivalente con columnas, tipos, rangos y manejo de errores. | Sprint 6 |
| API REST | `dashboard/app.py` no consume API; usa modelo local. | No cumple | No existe integracion dashboard/API. | Modificar el dashboard para invocar `/predict`, manteniendo modo local opcional para desarrollo. | Sprint 6 |
| Docker | No se encontro `Dockerfile`. | No cumple | No hay imagen reproducible para API o dashboard. | Crear Dockerfile para API y, si aplica, otro para dashboard o una imagen parametrizable por comando. | Sprint 6 |
| Docker | No se encontro `docker-compose.yml` ni `docker-compose.yaml`. | No cumple | No hay orquestacion local para levantar servicios relacionados. | Crear `docker-compose.yml` con servicios `api`, `dashboard` y variables de entorno necesarias. | Sprint 6 |
| Docker | No se encontraron instrucciones de build/run para contenedores. | No cumple | La ejecucion depende del ambiente local y de notebooks. | Documentar `docker build`, `docker run` y `docker compose up` con puertos y volumenes. | Sprint 6 |
| Docker | La arquitectura actual podria contenerizar API y dashboard, pero API no existe y Streamlit no esta declarado como dependencia. | Parcial | La contenerizacion es factible, pero bloqueada por dependencias incompletas y ausencia de API. | Completar dependencias, implementar API y luego contenerizar ambos servicios. | Sprint 6 |
| AWS | No se encontro evidencia tecnica reproducible de despliegue en AWS. Existen menciones generales a despliegue o handoff en reportes, pero no artefactos operativos. | No cumple | No hay guia ejecutable ni infraestructura definida. | Definir estrategia AWS minima: EC2/ECS para contenedores o Lambda para API, segun restricciones del proyecto. | Sprint 6 |
| AWS | No se encontraron instrucciones para EC2, Lambda, S3, ECR o CloudWatch. | No cumple | Falta ruta de despliegue, registro de imagenes, almacenamiento y monitoreo. | Documentar build de imagen, push a ECR, despliegue, variables, logs en CloudWatch y verificacion post-despliegue. | Sprint 6 |
| AWS | No se encontraron variables de entorno ni guia de configuracion para despliegue. | No cumple | No hay separacion entre configuracion local y configuracion productiva. | Definir `.env.example` o tabla de variables: rutas de modelo, API URL, MongoDB URI, MLflow URI y puertos. | Sprint 6 |
| AWS | El despliegue actual es local y centrado en Streamlit; no incluye API REST. | No cumple | No hay arquitectura productiva desacoplada dashboard/API. | Desplegar API y dashboard como servicios separados; el dashboard debe consumir la API publicada. | Sprint 6 |

## Priorizacion de correcciones

1. **Alta prioridad:** completar dependencias reproducibles y documentacion minima de ejecucion de Streamlit.
2. **Alta prioridad:** implementar API REST con `/health`, `/predict` y validacion de esquema.
3. **Alta prioridad:** integrar dashboard con la API y mantener modo local solo como fallback de desarrollo.
4. **Media prioridad:** crear Dockerfile y `docker-compose.yml` para levantar API y dashboard de forma reproducible.
5. **Media prioridad:** incorporar MLflow en Sprint 3 y Sprint 4 para trazabilidad formal de baseline, tuning, ensambles y modelo final.
6. **Media prioridad:** definir despliegue AWS reproducible con ECR, servicio de ejecucion, variables y monitoreo.
7. **Baja prioridad:** incorporar MongoDB como historico consultable de experimentos, predicciones y feedback.
8. **Baja prioridad:** implementar Voila como alternativa al dashboard Streamlit si el entregable lo exige.

## Que debe hacerse primero

Primero debe corregirse el contrato operativo del proyecto: dependencias completas, API REST con validacion de entrada y consumo de la API desde el dashboard. Sin este contrato, Docker y AWS quedarian como empaquetado de una aplicacion todavia acoplada al modelo local.

Despues debe agregarse Docker para reproducir la ejecucion de API y dashboard en un ambiente controlado. Con API y Docker listos, el despliegue AWS puede documentarse y validarse con una ruta concreta.

## Que puede quedar como mejora final

Voila puede quedar como alternativa final porque no es indispensable si Streamlit queda operativo. MongoDB tambien puede incorporarse despues de estabilizar API y dashboard, salvo que el curso exija historico consultable como requisito central. MLflow deberia priorizarse antes de cierre academico si la trazabilidad experimental es evaluada formalmente.

## Mapa correcto

- **MLflow en Sprint 3 y Sprint 4:** registrar modelos baseline, parametros, metricas, artefactos de comparacion, tuning, ensambles y modelo final.
- **MongoDB como capa transversal Sprint 4-5:** persistir historico consultable de experimentos relevantes, metricas finales, predicciones del dashboard y feedback.
- **Voila en Sprint 5 como alternativa de dashboard:** publicar un notebook ejecutivo compatible con Voila solo si se requiere una alternativa notebook-first.
- **API, Docker y AWS en Sprint 6:** implementar servicio REST, contenedores reproducibles, integracion dashboard/API y despliegue cloud con monitoreo basico.
