# Lab A Tema 7 - Workflow con Prefect y Weights & Biases

Proyecto para el Lab A del tema 7 de MLOps. El objetivo es crear workflows sobre
el dataset Titanic usando Prefect para orquestar etapas y Weights & Biases para
guardar artefactos, métricas, código e hiperparámetros.

## Que contiene

- `f01_monolith.py`: un flujo acoplado que hace todo el proceso completo.
- `f02_decoupled.py`: dos flujos desacoplados unidos por un artefacto procesado.
- `wandb_init.py`: sube el dataset raw de Titanic a W&B como artefacto inicial.
- `deploy_flows.py`: despliega todos los flows de Prefect con una sola ejecución.
- `docker-compose.yml`: levanta Prefect Server, Postgres, Redis y un worker.
- `config/`: parámetros del experimento y nombres de artefactos.
- `docs/diseno_workflows.md`: documento de diseño de la entrega.

## Preparación

Crear un fichero `.env`:

```env
PREFECT_API_URL=http://127.0.0.1:4200/api
WANDB_API_KEY=wandb_v1_TU_API_KEY
```

Instalar dependencias:

```bash
pip install -r requirements.txt
```

## Ejecución recomendada

1. Arrancar Prefect:

```bash
docker compose up -d
```

2. Abrir la interfaz:

```text
http://localhost:4200
```

3. Subir el dataset raw a W&B:

```bash
python wandb_init.py
```

4. Desplegar los flows:

```bash
python deploy_flows.py
```

5. Ejecutar desde la UI de Prefect:

- `Titanic Monolith - Full pipeline`
- `Titanic Decoupled - Process data`
- `Titanic Decoupled - Train model`

En el patrón desacoplado, primero hay que ejecutar `Process data`, porque genera
el artefacto `titanic_processed` que después consume `Train model`.

## Resultado esperado en W&B

El proyecto `mlops-titanic-labA-t7` debe contener:

- Artefacto raw: `titanic_raw`
- Artefacto procesado: `titanic_processed`
- Modelo monolítico: `titanic_knn_monolith`
- Modelo desacoplado: `titanic_knn_decoupled`
- Runs con métricas `accuracy`, `precision`, `recall` y `f1`
- Configuración e hiperparámetros guardados en cada run
