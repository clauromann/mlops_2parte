# Introducción al uso de Evidently AI para la Deriva del Dato

**Dataset:** Titanic (crudo) — 12 condiciones de división  
**Asignatura:** MLOPs — Master en Inteligencia Artificial  
**Autora:** Claudia Roman Garcia  
**Fecha:** Mayo 2026

---

## 1. Introducción

Este documento recoge el análisis de **deriva del dato (data drift)** realizado sobre el dataset del Titanic como parte del LabA de la asignatura MLOPs.

La deriva de datos ocurre cuando la distribución estadística de las variables cambia entre el conjunto con el que se entrena un modelo y los datos que llegan en producción. Detectarla a tiempo es fundamental en MLOps, porque **un modelo puede degradarse silenciosamente sin dar error**, simplemente porque los datos ya no son como los que él conoció.

En este ejercicio dividimos un único dataset estático en tres subconjuntos (train, val y test) bajo **12 condiciones distintas**. Como todos los datos provienen de la misma fuente, no esperamos deriva real — pero el ejercicio nos permite ver en qué condiciones puede aparecer **deriva artificial**, causada por el tamaño pequeño de las muestras o por la aleatoriedad del muestreo.

---

## 2. Metodología

### 2.1 Preprocesado del dataset (`data/titanic/titanic-dataset.csv`)

Se siguen exactamente los pasos de `data_cleaning_feature_engineering_steps.md`:

1. **Eliminar columna `Cabin`** — más del 77% de valores nulos.
2. **Imputar `Embarked` con `'S'`** — puerto más frecuente (Southampton).
3. **Imputar `Fare` con la media** de la columna.
4. **Imputar `Age` con la mediana** agrupando por `Sex` y `Pclass`.
5. **Convertir `Age` a entero.**
6. **Extraer el título (`Title`)** del campo `Name` y agrupar los raros en `'Rare'`.
7. **Crear `Family_size`** = `SibSp + Parch + 1`, categorizada en `Alone / Small / Large`.
8. **Eliminar columnas redundantes:** `Name`, `SibSp`, `Parch`, `Ticket`, `PassengerId`.

El dataset final tiene **8 columnas** y **891 filas**:

| Tipo | Columnas |
|------|----------|
| Numéricas | `Age`, `Fare` |
| Categóricas | `Pclass`, `Sex`, `Embarked`, `Title`, `Family_size_cat`, `Survived` |

---

### 2.2 Las 12 condiciones de división

Se combinan los siguientes parámetros (2 × 3 × 2 = **12 condiciones**):

| Parámetro | Valores |
|-----------|---------|
| Estratificación por `Survived` | Sí / No |
| Proporciones (train/val/test) | 60/20/20 · 90/5/5 · 98/1/1 |
| Semilla aleatoria | 42 · 123 |

La división se realiza en **dos pasos** con `train_test_split` de scikit-learn:
1. Separar `train` del resto (`val + test`).
2. Separar `val` de `test` dentro del resto.

Estratificar garantiza que la proporción de supervivientes (`Survived`) sea igual en los tres subconjuntos.

---

### 2.3 Análisis con Evidently AI

Para cada condición se generan **dos reports HTML** (`val vs train` y `test vs train`) usando `DataDriftPreset` + `DataSummaryPreset`.

Evidently aplica automáticamente el test estadístico más adecuado según el tipo de columna:
- **Numéricas** → Wasserstein distance o Kolmogorov-Smirnov
- **Categóricas** → Chi-cuadrado o Jensen-Shannon divergence

Una columna se considera con **deriva** cuando el p-valor o la distancia supera el umbral por defecto (0.05).

La métrica principal es la **fracción de columnas con drift** = columnas con drift / total (8).

---

## 3. Tabla resumen de resultados

> 🟡 Celdas en amarillo → drift detectado · 🔴 Fracción ≥ 0.20

| Cond. | Estrat. | Split (tr/va/te) | Semilla | Drift VAL (n/N) | Frac. VAL | Drift TEST (n/N) | Frac. TEST |
|:-----:|:-------:|:----------------:|:-------:|:---------------:|:---------:|:----------------:|:----------:|
| 01 | No | 60/20/20 | 42  | 0/8 | 0.000 | 0/8 | 0.000 |
| 02 | No | 60/20/20 | 123 | 0/8 | 0.000 | 1/8 | **0.125** |
| 03 | No | 90/5/5   | 42  | 1/8 | **0.125** | 0/8 | 0.000 |
| 04 | No | 90/5/5   | 123 | 0/8 | 0.000 | 1/8 | **0.125** |
| 05 | No | 98/1/1   | 42  | 1/8 | **0.125** | 0/8 | 0.000 |
| 06 | No | 98/1/1   | 123 | 0/8 | 0.000 | 1/8 | **0.125** |
| 07 | Sí | 60/20/20 | 42  | 0/8 | 0.000 | 1/8 | **0.125** |
| 08 | Sí | 60/20/20 | 123 | 1/8 | **0.125** | 1/8 | **0.125** |
| 09 | Sí | 90/5/5   | 42  | 0/8 | 0.000 | 0/8 | 0.000 |
| 10 | Sí | 90/5/5   | 123 | 0/8 | 0.000 | 1/8 | **0.125** |
| 11 | Sí | 98/1/1   | 42  | 0/8 | 0.000 | 0/8 | 0.000 |
| 12 | Sí | 98/1/1   | 123 | 2/8 | 🔴 **0.250** | 0/8 | 0.000 |

---

## 4. Discusión de resultados

### 4.1 Nivel general de deriva

En todas las condiciones la deriva es muy baja: entre **0 y 2 columnas de 8** presentan drift (fracciones de 0.000 a 0.250). Esto es exactamente lo esperado cuando dividimos un único dataset estático: train, val y test provienen de la misma población y no hay cambio real en los datos.

El drift detectado es puramente estadístico — consecuencia del pequeño tamaño de las muestras, no de un cambio real en el fenómeno.

---

### 4.2 Efecto de las proporciones de división

**60/20/20** → Val y test tienen ~178 filas. Con ese tamaño las distribuciones se aproximan bien a las del total y raramente se detecta drift. Es la condición **más estable**.

**90/5/5** → Val y test tienen solo ~45 filas. La varianza muestral aumenta y pequeñas desviaciones aleatorias pueden superar el umbral del test estadístico, generando **falsos positivos** (cond03, cond04, cond07, cond08, cond10).

**98/1/1** → Val y test tienen ~9 filas. La varianza es extrema. Cualquier diferencia aleatoria puede parecer drift. La condición 12 detecta drift en **2/8 columnas (25%)**, el valor más alto de todo el experimento. Con tan pocas filas, los tests estadísticos no son fiables.

> **Conclusión parcial:** a menor tamaño de muestra, mayor riesgo de detectar drift artificial.

---

### 4.3 Efecto de la estratificación

La estratificación garantiza que `Survived` mantenga su proporción en todos los subconjuntos. Con splits amplios (60/20/20) apenas hay diferencia, porque la ley de los grandes números ya garantiza proporciones similares por muestreo simple.

Con splits muy pequeños (98/1/1), la estratificación ayuda a controlar la distribución de `Survived`, pero **no puede controlar el resto de columnas** (`Age`, `Fare`, `Title`...), por lo que el drift puede seguir apareciendo en ellas. Comparando cond11 vs cond12 (misma estrat+proporción, diferente semilla):

- cond11 (seed=42): **0 drift**
- cond12 (seed=123): **2 drift (0.250)**

Incluso con estratificación, la semilla importa mucho cuando las muestras son tan pequeñas.

---

### 4.4 Efecto de la semilla aleatoria

La semilla controla qué filas concretas van a cada subconjunto. Comparando pares con la misma configuración pero diferente semilla se observan diferencias de hasta 1–2 columnas con drift:

| Par | Diferencia en drift VAL | Diferencia en drift TEST |
|-----|:-----------------------:|:------------------------:|
| cond01 vs cond02 (No, 60/20/20) | 0 vs 0 | 0 vs 1 |
| cond03 vs cond04 (No, 90/5/5)   | 1 vs 0 | 0 vs 1 |
| cond11 vs cond12 (Sí, 98/1/1)   | 0 vs 2 | 0 vs 0 |

Esto demuestra que el drift detectado es **fruto del azar del muestreo**, no de un cambio real. En la práctica, si detectamos drift con una semilla pero no con otra sobre los mismos datos, debemos sospechar que el tamaño de muestra es insuficiente para un diagnóstico fiable.

---

### 4.5 Val vs Test

No hay una tendencia sistemática de que val tenga más o menos drift que test. En varias condiciones uno tiene drift y el otro no (ej. cond03: val=0.125, test=0.000; cond07: val=0.000, test=0.125). Esto confirma que el drift observado es **aleatorio, no estructural**.

---

## 5. Conclusión

Al dividir un dataset estático, la deriva detectada por Evidently AI es **prácticamente nula** bajo condiciones razonables (split 60/20/20, con o sin estratificación). Los subconjuntos son representativos del total y no hay cambio real en los datos — que es la situación óptima para entrenar y evaluar un modelo.

El drift aparece principalmente cuando las muestras son muy pequeñas (proporciones 98/1/1 o 90/5/5) y es de naturaleza aleatoria, no real. Varía con la semilla porque depende de cuáles filas concretas caen en cada subconjunto.

**La estratificación es siempre recomendable**, especialmente con datasets pequeños o desbalanceados: garantiza que `Survived` esté bien representada en todos los subconjuntos, lo que mejora la calidad de la evaluación del modelo.

En producción real, el análisis de drift cobra su verdadero sentido cuando se comparan datos de entrenamiento con datos reales recogidos con el paso del tiempo. Ahí sí puede haber deriva real (cambios en el comportamiento de los usuarios, el contexto social, el mercado...) y Evidently AI sería la herramienta adecuada para detectarla y actuar antes de que el modelo se degrade.

---

## Anexo: Estructura de ficheros

```
labA/
├── data/titanic/
│   ├── titanic-dataset.csv
│   └── data_cleaning_feature_engineering_steps.md
├── reports/
│   ├── cond01_nostrat_60-20-20_seed42_report_VAL.html
│   ├── cond01_nostrat_60-20-20_seed42_report_TEST.html
│   └── ... (24 reports en total)
├── titanic_drift_analysis.py   ← script principal
├── requirements.txt
├── .venv/
├── LabA_Deriva_Datos_Titanic.docx
└── LabA_Deriva_Datos_Titanic.md  ← este fichero
```

## Anexo: Cómo ejecutar el script

```bash
# Desde la carpeta labA con el entorno activado:
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python titanic_drift_analysis.py
```

Los 24 reports HTML se generarán automáticamente en `reports/`.
