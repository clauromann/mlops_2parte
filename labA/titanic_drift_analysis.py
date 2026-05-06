# =============================================================
#  0. IMPORTACIONES
# =============================================================
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from evidently import Dataset, DataDefinition, Report
from evidently.presets import DataDriftPreset, DataSummaryPreset


# =============================================================
#  1. RUTAS
# =============================================================
# Todas las rutas son relativas a la carpeta labA
LABА_DIR    = Path(__file__).parent          # carpeta labA
DATA_PATH   = LABА_DIR / "data" / "titanic" / "titanic-dataset.csv"
REPORTS_DIR = LABА_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


# =============================================================
#  2. CARGA Y PREPROCESADO DEL DATASET
# =============================================================
# Seguimos exactamente los pasos de:
#   data/titanic/data_cleaning_feature_engineering_steps.md

def preprocess(path: Path) -> pd.DataFrame:
    """
    Carga el CSV crudo del Titanic y aplica limpieza + feature engineering.

    Pasos (según data_cleaning_feature_engineering_steps.md):
      1. Eliminar columna Cabin        (>77% nulos)
      2. Imputar Embarked con 'S'      (puerto más frecuente)
      3. Imputar Fare con la media     (hay 1 nulo en el dataset de test de Kaggle)
      4. Imputar Age con mediana       (agrupando por Sex y Pclass)
      5. Convertir Age a entero
      6. Extraer Title desde Name
      7. Consolidar títulos raros en 'Rare'
      8. Crear Family_size = SibSp + Parch + 1
      9. Eliminar columnas redundantes (Name, Parch, SibSp, Ticket)
     10. Categorizar Family_size en Alone / Small / Large
    """
    df = pd.read_csv(path)

    # Paso 1: Eliminar Cabin (demasiados valores nulos)
    df = df.drop(columns=["Cabin"], errors="ignore")

    # Paso 2: Imputar Embarked con la moda ('S' = Southampton)
    df["Embarked"] = df["Embarked"].fillna("S")

    # Paso 3: Imputar Fare con la media de la columna
    df["Fare"] = df["Fare"].fillna(df["Fare"].mean())

    # Paso 4: Imputar Age con la mediana agrupando por Sex y Pclass
    #   Usamos transform para rellenar dentro de cada grupo
    df["Age"] = df.groupby(["Sex", "Pclass"])["Age"].transform(
        lambda x: x.fillna(x.median())
    )

    # Paso 5: Convertir Age a entero
    df["Age"] = df["Age"].astype(int)

    # Paso 6: Extraer el título (Mr, Mrs, Miss, etc.) del campo Name
    df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()

    # Paso 7: Consolidar títulos poco frecuentes en 'Rare'
    rare_titles = ["Lady", "Countess", "Capt", "Col", "Don", "Dr",
                   "Major", "Rev", "Sir", "Jonkheer", "Dona"]
    df["Title"] = df["Title"].replace(rare_titles, "Rare")
    df["Title"] = df["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})

    # Paso 8: Crear columna Family_size
    df["Family_size"] = df["SibSp"] + df["Parch"] + 1

    # Paso 9: Eliminar columnas ya no necesarias
    df = df.drop(columns=["Name", "Parch", "SibSp", "Ticket", "PassengerId"],
                 errors="ignore")

    # Paso 10: Categorizar Family_size
    def categorize_family(n):
        if n == 1:
            return "Alone"
        elif n <= 4:
            return "Small"
        else:
            return "Large"

    df["Family_size_cat"] = df["Family_size"].apply(categorize_family)
    df = df.drop(columns=["Family_size"])  # ya no necesitamos el número

    return df


# Cargar y preprocesar el dataset
df_full = preprocess(DATA_PATH)

print("=" * 60)
print("  Dataset cargado y preprocesado")
print("=" * 60)
print(f"  Filas: {len(df_full)}")
print(f"  Columnas: {list(df_full.columns)}")
print(f"\n  Distribución de Survived (variable objetivo):")
print(df_full["Survived"].value_counts(normalize=True).round(3).to_string())
print()


# =============================================================
#  3. ESQUEMA PARA EVIDENTLY AI
# =============================================================
# Debemos indicar a Evidently qué columnas son numéricas y cuáles
# categóricas para que aplique el test estadístico correcto:
#   - Numéricas  → Wasserstein distance o Kolmogorov-Smirnov
#   - Categóricas → Chi-cuadrado o Jensen-Shannon divergence

NUMERICAL_COLS   = ["Age", "Fare"]
CATEGORICAL_COLS = ["Pclass", "Sex", "Embarked", "Title",
                    "Family_size_cat", "Survived"]

schema = DataDefinition(
    numerical_columns=NUMERICAL_COLS,
    categorical_columns=CATEGORICAL_COLS,
)


# =============================================================
#  4. FUNCIÓN PARA DIVIDIR EL DATASET (reutilizable)
# =============================================================

def split_dataset(df, train_r, val_r, test_r, stratify, seed):
    """
    Divide df en train / val / test con las proporciones indicadas.

    Parámetros:
      df        : DataFrame completo (ya preprocesado)
      train_r   : fracción para train  (ej. 0.60)
      val_r     : fracción para val    (ej. 0.20)
      test_r    : fracción para test   (ej. 0.20)
      stratify  : bool — si True, mantiene proporción de 'Survived'
      seed      : semilla para reproducibilidad

    Proceso (siempre en 2 pasos):
      1. Separar train del resto (val+test)
      2. Separar val de test dentro del resto

    Retorna:
      df_train, df_val, df_test
    """
    strat_col = df["Survived"] if stratify else None

    # Paso 1: train vs (val + test)
    df_train, df_rest = train_test_split(
        df,
        test_size=val_r + test_r,
        random_state=seed,
        stratify=strat_col,
    )

    # Paso 2: val vs test (dentro del resto)
    test_fraction = test_r / (val_r + test_r)
    strat_rest = df_rest["Survived"] if stratify else None

    df_val, df_test = train_test_split(
        df_rest,
        test_size=test_fraction,
        random_state=seed,
        stratify=strat_rest,
    )

    return df_train, df_val, df_test


# =============================================================
#  5. FUNCIÓN PARA GENERAR UN REPORT DE EVIDENTLY (reutilizable)
# =============================================================

def generate_drift_report(df_reference, df_current, schema, output_path):
    """
    Genera un report HTML de deriva usando Evidently AI.

    Evidently compara df_current contra df_reference y detecta
    si la distribución de cada columna ha cambiado significativamente.

    Usamos dos presets:
      - DataDriftPreset  : detecta deriva en todas las columnas
      - DataSummaryPreset: estadísticas descriptivas comparadas

    Parámetros:
      df_reference : conjunto de referencia (train)
      df_current   : conjunto a comparar (val o test)
      schema       : DataDefinition con tipos de columnas
      output_path  : ruta del fichero HTML de salida
    """
    # Convertir DataFrames al formato que entiende Evidently
    ev_reference = Dataset.from_pandas(df_reference, data_definition=schema)
    ev_current   = Dataset.from_pandas(df_current,   data_definition=schema)

    # Crear el report con include_tests=True para añadir tests automáticos
    report = Report(
        metrics=[
            DataDriftPreset(),   # Detecta deriva columna por columna
            DataSummaryPreset(), # Estadísticas descriptivas comparadas
        ],
        include_tests=True,
    )

    # Ejecutar el análisis
    result = report.run(
        reference_data=ev_reference,
        current_data=ev_current,
    )

    # Guardar como HTML
    result.save_html(str(output_path))
    return result


# =============================================================
#  6. DEFINICIÓN DE LAS 12 CONDICIONES
# =============================================================

conditions = []
for stratify in [False, True]:
    for (tr, va, te) in [(0.60, 0.20, 0.20),
                         (0.90, 0.05, 0.05),
                         (0.98, 0.01, 0.01)]:
        for seed in [42, 123]:
            conditions.append({
                "stratify": stratify,
                "train":    tr,
                "val":      va,
                "test":     te,
                "seed":     seed,
            })

print(f"Total de condiciones a ejecutar: {len(conditions)}")
print()


# =============================================================
#  7. BUCLE PRINCIPAL
# =============================================================

summary = []  # Para guardar los resultados de drift de cada condición

for i, cond in enumerate(conditions, start=1):

    # Nombre descriptivo y trazable para los ficheros
    # Formato: condXX_strat/nostrat_TR-VA-TE_seedYYY
    strat_tag = "strat" if cond["stratify"] else "nostrat"
    split_tag = (f"{int(cond['train']*100)}-"
                 f"{int(cond['val']*100)}-"
                 f"{int(cond['test']*100)}")
    name = f"cond{i:02d}_{strat_tag}_{split_tag}_seed{cond['seed']}"

    print(f"{'='*60}")
    print(f"  Condición {i:02d}: {strat_tag} | {split_tag} | seed={cond['seed']}")
    print(f"{'='*60}")

    # ── Dividir el dataset ────────────────────────────────────
    df_train, df_val, df_test = split_dataset(
        df_full,
        cond["train"], cond["val"], cond["test"],
        cond["stratify"], cond["seed"],
    )
    print(f"  Tamaños → train: {len(df_train)} | val: {len(df_val)} | test: {len(df_test)}")
    print(f"  % Survived → train: {df_train['Survived'].mean():.3f} | "
          f"val: {df_val['Survived'].mean():.3f} | "
          f"test: {df_test['Survived'].mean():.3f}")

    # ── Report VAL vs TRAIN ───────────────────────────────────
    html_val = REPORTS_DIR / f"{name}_report_VAL.html"
    generate_drift_report(df_train, df_val, schema, html_val)
    print(f"  [OK] Report VAL  → {html_val.name}")

    # ── Report TEST vs TRAIN ──────────────────────────────────
    html_test = REPORTS_DIR / f"{name}_report_TEST.html"
    generate_drift_report(df_train, df_test, schema, html_test)
    print(f"  [OK] Report TEST → {html_test.name}")

    print()

print("=" * 60)
print(f"  ¡Completado! {len(list(REPORTS_DIR.glob('*.html')))} reports generados en reports/")
print("=" * 60)
