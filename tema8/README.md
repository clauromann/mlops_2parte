# Tema 8 - Pipeline completo de GitLab CI/CD para MLOps

Este proyecto implementa un pipeline completo de GitLab para entrenar y
desplegar un modelo sobre el dataset Titanic.

La idea es automatizar en GitLab lo que en temas anteriores se hacía de forma
manual o con Prefect:

```text
tests -> procesado -> entrenamiento -> imagen Docker -> despliegue pre -> test pre -> despliegue pro -> test pro
```

## Estructura

```text
tema8/
|-- .gitlab-ci.yml
|-- requirements.txt
|-- requirements-dev.txt
|-- src/
|   |-- data/
|   |-- processing/
|   |-- training/
|   |-- predict/
|   `-- utils/runpod/
`-- tests/
    |-- unit/
    `-- e2e/
```

En la raíz del repositorio hay otro archivo `.gitlab-ci.yml` muy pequeño que
solo incluye este pipeline. GitLab necesita que exista un `.gitlab-ci.yml` en la
raíz para detectar el pipeline automáticamente.

## Qué hace cada etapa

### 1. `unit_tests`

Ejecuta pruebas unitarias sobre transformaciones del dataset Titanic.

Se prueban dos métodos no triviales:

- extracción y normalización del título (`Title`);
- categorización del tamaño familiar (`Family_size_cat`).

### 2. `process_data`

Lee el dataset crudo de Titanic, aplica limpieza y feature engineering, y guarda
en W&B:

- artefacto `titanic_raw_gitlab`;
- artefacto `titanic_processed_gitlab`;
- configuración y código del procesamiento.

### 3. `train_model`

Descarga desde W&B el dataset procesado, entrena un modelo KNN y registra:

- métricas `accuracy`, `precision`, `recall` y `f1`;
- configuración e hiperparámetros;
- artefacto `titanic_knn_gitlab`.

### 4. `build_predict_image`

Descarga el modelo desde W&B, lo incluye dentro de una imagen Docker de
inferencia y sube la imagen a Docker Hub.

La imagen se etiqueta con:

```text
$CI_COMMIT_SHORT_SHA
latest
```

Usar el SHA del commit sigue la buena práctica vista en clase: no depender solo
de `latest`.

### 5. `deploy_pre`

Actualiza la template y endpoint de RunPod del entorno de pruebas (`pre`).

### 6. `test_pre`

Ejecuta tests end-to-end contra el endpoint `pre`.

Comprueba:

- que el endpoint responde en `/health`;
- que `/runsync` devuelve una salida con `prediction`.

### 7. `deploy_pro`

Despliega en producción (`pro`). Está configurado como manual en la rama
principal para evitar despliegues accidentales.

### 8. `test_pro`

Ejecuta los mismos tests end-to-end contra el endpoint de producción.

## Variables necesarias en GitLab

Configurar en:

```text
Settings -> CI/CD -> Variables
```

Variables:

| Variable | Uso |
| --- | --- |
| `DOCKERHUB_IMAGE_NAME` | Nombre de imagen, por ejemplo `clauromann/titanic-gitlab-predict` |
| `DOCKERHUB_USER` | Usuario de Docker Hub |
| `DOCKER_PAT` | Token de Docker Hub |
| `RUNPOD_API_KEY` | API key de RunPod |
| `RUNPOD_PRE_TEMPLATE_ID` | Template de RunPod para pre |
| `RUNPOD_PRE_ENDPOINT_ID` | Endpoint de RunPod para pre |
| `RUNPOD_PRO_TEMPLATE_ID` | Template de RunPod para pro |
| `RUNPOD_PRO_ENDPOINT_ID` | Endpoint de RunPod para pro |
| `WANDB_API_KEY` | API key de W&B |
| `WANDB_ENTITY` | Entidad de W&B, por ejemplo `cromangarcia-universidad-loyola` |

Las claves deben configurarse como variables protegidas/enmascaradas cuando sea
posible. No deben subirse al repositorio.

## Preparación en RunPod

Hay que crear dos templates y dos endpoints:

- entorno `pre`;
- entorno `pro`.

Las templates pueden crearse inicialmente con la imagen placeholder:

```text
python:latest
```

El pipeline actualizará después la imagen real usando la API de RunPod.

## Entrega

El entregable principal es el enlace al repositorio de GitLab compartido con:

```text
amc.loyola
```

con rol:

```text
Maintainer
```

o:

```text
Owner
```
