# Lab B Tema 7 - Entrenamiento y despliegue en RunPod

Este proyecto adapta el Lab A del tema 7 para que el entrenamiento del modelo
Titanic se ejecute en un Pod de RunPod y el modelo entrenado se despliegue en
RunPod Serverless.

En esta versión, el Pod de entrenamiento se crea manualmente desde la interfaz
de RunPod, porque la cuenta/API disponible no permite crearlo automáticamente
desde la API. El entrenamiento sigue ocurriendo en RunPod usando la imagen Docker
generada por el flujo de Prefect.

## Qué hace cada parte

1. `Titanic RunPod - Process data`
   - Descarga `titanic_raw` desde W&B.
   - Aplica la limpieza y feature engineering.
   - Guarda `titanic_processed` en W&B.

2. `Titanic RunPod - Build training image`
   - Construye la imagen Docker de entrenamiento.
   - La sube a Docker Hub como `clauromann/titanic-runpod-training:latest`.

3. Pod manual en RunPod
   - Se crea desde la interfaz web de RunPod.
   - Usa la imagen `clauromann/titanic-runpod-training:latest`.
   - El contenedor descarga `titanic_processed`, entrena el modelo y registra
     `titanic_knn_runpod` en W&B.

4. `Titanic RunPod - Serverless deployment`
   - Descarga el modelo entrenado desde W&B.
   - Construye la imagen de predicción.
   - La sube a Docker Hub como `clauromann/titanic-runpod-predict:latest`.
   - Actualiza el template y endpoint de RunPod Serverless.

## Variables necesarias

Completar `.env`:

```env
PREFECT_API_URL=http://127.0.0.1:4200/api
DOCKER_PAT=dckr_pat_YOUR_DOCKERHUB_TOKEN
WANDB_API_KEY=wandb_v1_YOUR_WANDB_KEY
RUNPOD_API_KEY=YOUR_RUNPOD_KEY
```

`RUNPOD_API_KEY` se usa para el despliegue serverless. Para el entrenamiento
manual, también se copia como variable `RUNPOD_MANAGE_KEY` dentro del Pod para
que el script intente detenerlo al finalizar.

## Ejecución hasta entrenamiento

Desde esta carpeta:

```powershell
docker compose up -d
```

Después:

```powershell
python wandb_init.py
python deploy_flows.py
```

El comando `deploy_flows.py` debe quedarse abierto.

En Prefect (`http://localhost:4200`) ejecutar:

1. `Titanic RunPod - Process data`
2. `Titanic RunPod - Build training image`

Después crear manualmente el Pod de entrenamiento en RunPod.

## Crear manualmente el Pod de entrenamiento

En RunPod:

1. Ir a `Pods`.
2. Pulsar `Deploy`.
3. Elegir una configuración pequeña, preferiblemente CPU si la cuenta lo permite.
4. Usar como imagen Docker:

```text
clauromann/titanic-runpod-training:latest
```

5. Añadir variables de entorno:

```env
WANDB_API_KEY=tu_api_key_de_wandb
RUNPOD_MANAGE_KEY=tu_api_key_de_runpod
```

6. Lanzar el Pod.
7. Revisar los logs del Pod.
8. Confirmar que en W&B aparece el run `runpod-train`.
9. Confirmar que existe el artefacto `titanic_knn_runpod`.
10. Comprobar manualmente que el Pod queda detenido para no consumir créditos.

El contenedor ejecuta automáticamente:

```text
python train/train.py
```

## Crear template y endpoint serverless

Antes de ejecutar el despliegue serverless, hay que crear en RunPod una plantilla
placeholder y un endpoint asociado a esa plantilla. Esto es lo indicado por el
profesor: la plantilla se crea primero con una imagen temporal y después nuestro
flow de Prefect reemplaza esa imagen por la imagen real de predicción.

### 1. Crear la plantilla

En RunPod:

1. Ir a `Resources > My templates`.
2. Pulsar `New Template`.
3. Darle un nombre, por ejemplo:

```text
titanic-labB-pre-template
```

4. Configurar:

```text
Type: Serverless
Compute type: CPU
Container image: python:latest
```

La imagen `python:latest` es solo un placeholder. Se puede ignorar la advertencia
porque después Prefect actualizará la plantilla con:

```text
clauromann/titanic-runpod-predict:latest
```

5. Guardar la plantilla.
6. Copiar el ID de la plantilla.

### 2. Crear el endpoint

En RunPod:

1. Ir a `Resources > Serverless`.
2. Pulsar `New Endpoint`.
3. Seleccionar `Custom deployment`.
4. Seleccionar `Deploy from Docker registry or a template`.
5. Elegir la plantilla creada en el paso anterior.
6. Configurar:

```text
Endpoint type: queue
Worker type: CPU
CPU Configuration: Compute-Optimized bajo CPU3
vCPUs: 2
Max Workers: 1
Active Workers: 0
```

7. Crear el endpoint.
8. Copiar el ID del endpoint.


### 3. Copiar IDs en la configuración

Después hay que copiar los IDs reales en `config/runpod.json`:

```json
"template_id": "TU_TEMPLATE_ID",
"endpoint_id": "TU_ENDPOINT_ID"
```

## Despliegue serverless

Cuando el entrenamiento ya haya generado `titanic_knn_runpod` en W&B y los IDs
de RunPod estén configurados, ejecutar en Prefect:

```text
Titanic RunPod - Serverless deployment
```

Este flow construye y sube la imagen:

```text
clauromann/titanic-runpod-predict:latest
```

y actualiza el template y endpoint de RunPod.

## Resultado esperado en W&B

Proyecto:

```text
mlops-titanic-labB-t7
```

Artefactos esperados:

- `titanic_raw`
- `titanic_processed`
- `titanic_knn_runpod`

Runs esperados:

- subida de dataset raw;
- procesamiento de datos;
- entrenamiento en RunPod con métricas `accuracy`, `precision`, `recall` y `f1`.
