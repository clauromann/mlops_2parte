# Documento de diseño - Lab A Tema 7

## Objetivo

El objetivo del laboratorio es construir workflows de MLOps sobre el dataset
Titanic usando Prefect y Weights & Biases. Se implementan los dos patrones
solicitados en el enunciado:

1. Un único flujo completamente acoplado.
2. Al menos dos flujos desacoplados, relacionados mediante artefactos en W&B.

El punto de partida es el dataset original de Titanic, guardado como artefacto
raw en W&B. El objetivo final es entrenar un modelo, serializarlo y registrarlo
también en W&B, junto con la información necesaria para mantener la trazabilidad:
código, datos, configuración, hiperparámetros y métricas.

## Criterios generales de diseño

Se han seguido los principios vistos en clase:

- Prefect se usa para orquestar el workflow.
- Cada etapa concreta se implementa como `@task`.
- Cada pipeline completo se implementa como `@flow`.
- W&B se usa para registrar artefactos, métricas y configuración.
- Los parámetros se guardan en `config/*.json`, no directamente dentro del código.
- La limpieza del dataset Titanic sigue las operaciones predefinidas en Moodle.

La división en etapas se ha hecho siguiendo un criterio sencillo: cada `task`
representa una operación con una responsabilidad clara. Por ejemplo, descargar
datos, limpiar datos, dividir train/test, entrenar, evaluar o registrar el
modelo. Esto hace que el flujo sea más fácil de entender, depurar y auditar.

## Patrón 1: flujo acoplado

Archivo principal: `f01_monolith.py`

Flow: `Titanic Monolith - Full pipeline`

Etapas:

1. Descargar el dataset raw desde W&B.
2. Aplicar limpieza y feature engineering.
3. Separar train/test con estratificación por `Survived`.
4. Entrenar un modelo KNN.
5. Evaluar con accuracy, precision, recall y F1.
6. Registrar el modelo serializado en W&B.

### Criterio de división del flujo acoplado

En el patrón acoplado todo el ciclo de vida se ejecuta dentro de un único flow.
La división existe solo a nivel de etapas internas (`tasks`), no a nivel de flows.

Este diseño se ha elegido porque es simple y adecuado para una primera versión
del pipeline. Permite ver todo el proceso completo de principio a fin:

```text
raw data -> limpieza -> train/test -> entrenamiento -> evaluación -> modelo
```

La ventaja principal es la sencillez. La desventaja es que si solo queremos
repetir el entrenamiento, también se vuelve a ejecutar la limpieza de datos.

## Patrón 2: flujos desacoplados

Archivo principal: `f02_decoupled.py`

Flows:

1. `Titanic Decoupled - Process data`
2. `Titanic Decoupled - Train model`

### Flow 1: procesado de datos

Etapas:

1. Descargar el dataset raw desde W&B.
2. Aplicar limpieza y feature engineering.
3. Guardar el dataset procesado como artefacto `titanic_processed`.

### Flow 2: entrenamiento

Etapas:

1. Descargar el artefacto `titanic_processed` desde W&B.
2. Separar train/test con estratificación por `Survived`.
3. Entrenar el modelo KNN.
4. Evaluar el modelo.
5. Registrar el modelo serializado en W&B.

### Criterio de división de los flujos desacoplados

La separación entre flows se ha hecho en el punto donde aparece un artefacto
estable y reutilizable: el dataset procesado.

El primer flow produce `titanic_processed`. El segundo flow consume ese artefacto
para entrenar el modelo. Así, ambos flows quedan relacionados mediante W&B.

Este criterio tiene sentido porque la limpieza y el entrenamiento tienen ciclos
de vida distintos:

- Si cambian las reglas de limpieza, hay que volver a ejecutar el flow de procesado.
- Si solo queremos repetir el entrenamiento, podemos reutilizar el dataset ya procesado.
- W&B conserva tanto el dataset raw como el procesado, lo que mejora la trazabilidad.

En resumen, la división desacoplada queda así:

```text
Flow 1: raw data -> limpieza -> titanic_processed
Flow 2: titanic_processed -> train/test -> entrenamiento -> evaluación -> modelo
```

## Limpieza y feature engineering

Se aplican los pasos indicados para el dataset Titanic:

1. Eliminar `Cabin`.
2. Imputar `Embarked` con `S`.
3. Imputar `Fare` con la media.
4. Imputar `Age` con la mediana agrupada por `Sex` y `Pclass`.
5. Convertir `Age` a entero.
6. Extraer `Title` desde `Name`.
7. Agrupar títulos raros en `Rare`.
8. Crear `Family_size`.
9. Eliminar columnas redundantes.
10. Crear `Family_size_cat`.

Tras este proceso, el dataset final tiene 891 filas y 8 columnas.

## Modelo elegido

Se usa `KNeighborsClassifier`. Como Titanic tiene variables numéricas y categóricas, 
el modelo se incluye dentro de un `Pipeline` de scikit-learn:

- `StandardScaler` para `Age` y `Fare`.
- `OneHotEncoder` para variables categóricas.
- `KNeighborsClassifier` como estimador final.

## Trazabilidad

La trazabilidad se garantiza con tres elementos:

- Código: W&B guarda el script del flow ejecutado.
- Datos: W&B guarda artefactos raw, procesados y modelos.
- Configuración: cada run guarda hiperparámetros como `n_neighbors`,
  `test_size`, `random_state` y listas de features.

Esto sigue la idea siguiente: en Machine Learning no basta con guardar el
código, porque el comportamiento del modelo también depende de los datos y de
la configuración usada.

## Resultados obtenidos

Los dos patrones han generado las mismas métricas porque el entrenamiento final
se ha hecho con el mismo dataset procesado, las mismas features, el mismo modelo
KNN y los mismos hiperparámetros.

| Patrón | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| Flujo acoplado / monolítico | 0.82123 | 0.79365 | 0.72464 | 0.75758 |
| Flujos desacoplados | 0.82123 | 0.79365 | 0.72464 | 0.75758 |

En W&B quedan registrados los runs y artefactos principales:

- `titanic_raw`: dataset original.
- `titanic_processed`: dataset procesado por el flujo desacoplado.
- `titanic_knn_monolith`: modelo generado por el flujo acoplado.
- `titanic_knn_decoupled`: modelo generado por el flujo desacoplado.

Estos resultados confirman que ambas estrategias implementan el mismo proceso
de entrenamiento, pero con una organización distinta del workflow.
