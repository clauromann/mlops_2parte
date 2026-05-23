# Diseño del pipeline - Tema 8

## Objetivo

El objetivo es crear un pipeline completo en GitLab CI/CD que, partiendo del
dataset crudo del Titanic, procese los datos, entrene un modelo, registre
trazabilidad en W&B y despliegue el modelo en RunPod Serverless en dos entornos:
pruebas (`pre`) y producción (`pro`).

## Criterio de división

El pipeline se divide en etapas siguiendo las responsabilidades principales de
un flujo MLOps:

1. Validar el código con tests.
2. Procesar datos.
3. Entrenar el modelo.
4. Construir la imagen de inferencia.
5. Desplegar en preproducción.
6. Probar el endpoint de preproducción.
7. Desplegar en producción.
8. Probar el endpoint de producción.

Esta división sigue el enfoque visto en clase: un pipeline está formado por
stages y cada stage contiene jobs especializados.

## Calidad del código

El procesamiento del Titanic está dividido en módulos:

- `processing/cleaning.py`: limpieza e imputaciones.
- `processing/features.py`: creación de variables.
- `processing/pipeline.py`: composición del procesamiento completo.
- `processing/process.py`: ejecución del procesado y persistencia en W&B.

Esto evita tener un único script grande y facilita probar funciones concretas.

## Pruebas unitarias

Se incluyen pruebas para dos transformaciones no triviales:

1. Extracción y normalización del título (`Title`) a partir del nombre.
2. Categorización del tamaño familiar (`Family_size_cat`).

Estas pruebas se ejecutan en el primer stage del pipeline. Si fallan, no se
procesan datos, no se entrena y no se despliega nada.

## Trazabilidad en W&B

El procesamiento registra:

- dataset crudo como `titanic_raw_gitlab`;
- dataset procesado como `titanic_processed_gitlab`;
- configuración usada;
- código del proceso.

El entrenamiento registra:

- modelo entrenado como `titanic_knn_gitlab`;
- métricas de evaluación;
- hiperparámetros;
- configuración y código.

## Despliegue en RunPod

El pipeline construye una imagen Docker de inferencia que contiene el handler y
el modelo descargado desde W&B.

Después actualiza dos entornos:

- `pre`: entorno de pruebas.
- `pro`: entorno de producción.

El despliegue a producción se configura como manual para evitar publicar cambios
sin revisar antes el resultado en preproducción.

## Pruebas end-to-end

Los tests end-to-end verifican que el endpoint de RunPod:

- responde en `/health`;
- acepta una petición `/runsync`;
- devuelve una salida con la clave `prediction`;
- devuelve una predicción válida (`0` o `1`).

Esto comprueba la salida real de la API, como pide el enunciado.

## Variables externas

Las credenciales no se guardan en el repositorio. Se configuran como variables
de CI/CD en GitLab:

- `WANDB_API_KEY`
- `RUNPOD_API_KEY`
- `DOCKER_PAT`
- IDs de templates/endpoints de RunPod
- nombre de imagen Docker

Así se mantiene una separación clara entre código y secretos.
