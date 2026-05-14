# Documento de diseño - Lab B Tema 7

## Objetivo

El objetivo del Lab B es modificar el trabajo del Lab A para que el entrenamiento
del modelo no se ejecute localmente, sino dentro de un Pod de RunPod. Además, se
añade un flujo de despliegue para publicar el modelo entrenado en RunPod
Serverless.

En esta implementación, Prefect automatiza el procesamiento de datos, la
construcción de imágenes Docker y el despliegue serverless. El Pod de
entrenamiento se crea manualmente desde la interfaz de RunPod porque la cuenta
usada no autoriza la creación automática de Pods mediante API.


## Herramientas utilizadas

- Prefect: orquestación de workflows.
- W&B: trazabilidad de datasets, modelos, métricas y configuración.
- Docker: empaquetado del entrenamiento y de la inferencia.
- Docker Hub: registro de imágenes Docker.
- RunPod: ejecución remota del entrenamiento y despliegue serverless.

## Diseño general

La solución se divide en tres deployments de Prefect y un paso manual en RunPod:

1. Procesamiento de datos con Prefect.
2. Construcción de la imagen Docker de entrenamiento con Prefect.
3. Ejecución manual del Pod de entrenamiento en RunPod.
4. Despliegue serverless con Prefect.

Esta división sigue un criterio práctico: cada fase tiene una responsabilidad
clara y produce un resultado verificable.

## Deployment 1: procesamiento de datos

Deployment: `Titanic RunPod - Process data`

Archivo: `f01_process_data.py`

Etapas:

1. Descargar `titanic_raw` desde W&B.
2. Aplicar limpieza y feature engineering.
3. Guardar el dataset procesado como `titanic_processed`.

Este paso se mantiene local porque es ligero y genera un artefacto estable que
después consume el entrenamiento remoto.

## Deployment 2: construcción de imagen de entrenamiento

Deployment: `Titanic RunPod - Build training image`

Archivo: `f02_build_training_image.py`

Etapas:

1. Iniciar sesión en Docker Hub.
2. Construir la imagen `clauromann/titanic-runpod-training:latest`.
3. Subir la imagen a Docker Hub.

Este paso es necesario porque RunPod ejecuta contenedores Docker. Por tanto, el
código de entrenamiento debe estar empaquetado en una imagen accesible desde
RunPod.

## Paso manual: entrenamiento en RunPod

El entrenamiento se ejecuta en un Pod creado manualmente desde la interfaz de
RunPod.

Configuración principal del Pod:

- Imagen Docker: `clauromann/titanic-runpod-training:latest`.
- Variables de entorno:
  - `WANDB_API_KEY`
  - `RUNPOD_MANAGE_KEY`

El script que se ejecuta dentro del contenedor es `train/train.py`. Sus etapas
son:

1. Descargar `titanic_processed` desde W&B.
2. Dividir train/test.
3. Entrenar el modelo KNN.
4. Registrar métricas en W&B.
5. Guardar el modelo como artefacto `titanic_knn_runpod`.
6. Intentar detener el Pod al terminar.

Aunque el script intenta detener el Pod, se debe revisar manualmente en RunPod
que no quede activo para evitar consumo innecesario de créditos.

## Deployment 3: despliegue serverless

Deployment: `Titanic RunPod - Serverless deployment`

Archivo: `f04_deploy_serverless.py`

Etapas:

1. Descargar el modelo entrenado desde W&B.
2. Construir la imagen Docker de predicción.
3. Subir la imagen a Docker Hub como `clauromann/titanic-runpod-predict:latest`.
4. Actualizar el template de RunPod.
5. Actualizar el endpoint serverless.

Antes de ejecutar este deployment, se deben crear manualmente en RunPod una
Serverless Template y un Serverless Endpoint siguiendo el procedimiento indicado
por el profesor.

Primero se crea una plantilla placeholder en `Resources > My templates`:

- Type: `Serverless`.
- Compute type: `CPU`.
- Container image: `python:latest`.

La imagen `python:latest` solo sirve como valor temporal. Después, el deployment
de Prefect actualiza programáticamente esa plantilla para que use la imagen real:

```text
clauromann/titanic-runpod-predict:latest
```

Después se crea el endpoint en `Resources > Serverless`:

- Custom deployment.
- Deploy from Docker registry or a template.
- Seleccionar la plantilla creada.
- Endpoint type: `queue`.
- Worker type: `CPU`.
- CPU3 Compute-Optimized con 2 vCPUs.
- Max Workers: 1.
- Active Workers: 0.

Los IDs de la plantilla y del endpoint se copian en `config/runpod.json`:

```json
"template_id": "TU_TEMPLATE_ID",
"endpoint_id": "TU_ENDPOINT_ID"
```
El handler de predicción está en `predict/handler.py`.

## Criterio de división

La división se ha hecho según el tipo de responsabilidad:

- Datos: preparación del dataset y generación del artefacto procesado.
- Infraestructura de entrenamiento: construcción de imagen Docker.
- Ejecución remota: entrenamiento dentro de un Pod de RunPod.
- Servicio de inferencia: despliegue serverless.

Esto evita mezclar en un único script operaciones muy distintas. También mejora
la trazabilidad: en Prefect se puede ver qué fase se ejecutó, y en W&B se puede
ver qué datos, métricas y modelo produjo el proceso.

## Trazabilidad

W&B registra:

- `titanic_raw`: dataset original.
- `titanic_processed`: dataset limpio.
- `titanic_knn_runpod`: modelo entrenado en RunPod.
- Métricas del entrenamiento: accuracy, precision, recall y F1.
- Configuración e hiperparámetros: `n_neighbors`, `test_size`, `random_state` y
  features utilizadas.

El proyecto de W&B usado es:

```text
mlops-titanic-labB-t7
```

## Resultados obtenidos

El entrenamiento se ha ejecutado correctamente dentro de un Pod de RunPod usando
la imagen Docker `clauromann/titanic-runpod-training:latest`. El run de W&B
asociado aparece con `train_job_type = runpod-train`.

Métricas registradas en W&B:

| Métrica | Valor |
| --- | ---: |
| Accuracy | 0.82123 |
| Precision | 0.79365 |
| Recall | 0.72464 |
| F1 | 0.75758 |

El modelo entrenado se ha guardado en W&B como artefacto:

```text
titanic_knn_runpod
```

Después del entrenamiento, el deployment `Titanic RunPod - Serverless deployment`
se ha ejecutado correctamente en Prefect. Este deployment ha construido y subido
la imagen Docker de inferencia:

```text
clauromann/titanic-runpod-predict:latest
```

Además, la plantilla de RunPod se ha actualizado para usar esa imagen y el
endpoint serverless queda preparado para servir predicciones.
