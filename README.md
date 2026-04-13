# phishing-detection-Equipo5

Proyecto académico de Data Analytics – Sprint 1  
Proyecto 7: Detección de Sitios Web Fraudulentos (Phishing)

## Objetivo del proyecto
Desarrollar un sistema automático de detección de sitios web de phishing que permita identificar URLs maliciosas antes de que el usuario interactúe con ellas.

## Sprint 1
Alcance:
- Business Understanding
- Data Understanding

## Estructura base del repositorio
- data/raw
- data/processed
- notebooks
- environment.yml
- .gitignore
- README.md

## Configuración del entorno
conda env create -f environment.yml  
conda activate phishing_s1  
python -m ipykernel install --user --name phishing_s1 --display-name "Python (phishing_s1)"  
jupyter lab

## Flujo de trabajo
- No trabajar directamente en `main`
- Crear ramas por tarea: `feat/PB-XX`
- Subir cambios a GitHub
- Abrir Pull Request para revisión antes del merge
