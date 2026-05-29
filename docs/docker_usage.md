# Docker Sprint 6

Esta configuracion permite ejecutar la API FastAPI y el dashboard Streamlit con Docker Compose. El objetivo es dejar el proyecto preparado para un despliegue posterior en AWS EC2.

## Requisitos

- Docker Desktop o Docker Engine.
- Docker Compose disponible como `docker compose`.

## Construir imagenes

Desde la raiz del repositorio:

```bash
docker compose build
```

## Levantar servicios

```bash
docker compose up
```

Servicios disponibles:

- API: `http://127.0.0.1:8000/health`
- Dashboard: `http://127.0.0.1:8501`

El dashboard queda configurado para consumir la API mediante `API_BASE_URL=http://api:8000` dentro de la red de Compose. El modo local del dashboard sigue disponible como respaldo.

## Detener servicios

```bash
docker compose down
```

## Variables principales

Servicio API:

```text
MODEL_PATH=models/final_model.pkl
DEFAULT_THRESHOLD=0.09
```

Servicio dashboard:

```text
API_BASE_URL=http://api:8000
```

## Preparacion para AWS EC2

Esta contenerizacion separa la API y el dashboard en servicios independientes, expone los puertos esperados y centraliza las variables de ejecucion. Para AWS EC2, el siguiente paso es instalar Docker en la instancia, copiar o clonar el repositorio, construir las imagenes y levantar los servicios con `docker compose up -d`.

## Nota de validacion

Si Docker no esta instalado localmente, la validacion debe realizarse en AWS EC2 o en otra maquina con Docker Desktop o Docker Engine disponible.
