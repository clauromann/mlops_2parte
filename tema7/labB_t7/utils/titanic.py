import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def preprocess_titanic(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the cleaning and feature engineering steps defined for the course."""
    df = df.copy()

    df = df.drop(columns=["Cabin"], errors="ignore")
    df["Embarked"] = df["Embarked"].fillna("S")
    df["Fare"] = df["Fare"].fillna(df["Fare"].mean())
    df["Age"] = df.groupby(["Sex", "Pclass"])["Age"].transform(
        lambda values: values.fillna(values.median())
    )
    df["Age"] = df["Age"].astype(int)

    df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()
    rare_titles = [
        "Lady",
        "Countess",
        "Capt",
        "Col",
        "Don",
        "Dr",
        "Major",
        "Rev",
        "Sir",
        "Jonkheer",
        "Dona",
    ]
    df["Title"] = df["Title"].replace(rare_titles, "Rare")
    df["Title"] = df["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})

    df["Family_size"] = df["SibSp"] + df["Parch"] + 1
    df = df.drop(
        columns=["Name", "Parch", "SibSp", "Ticket", "PassengerId"],
        errors="ignore",
    )
    df["Family_size_cat"] = df["Family_size"].apply(categorize_family_size)
    df = df.drop(columns=["Family_size"])
    return df


def categorize_family_size(size: int) -> str:
    if size == 1:
        return "Alone"
    if size <= 4:
        return "Small"
    return "Large"


def build_knn_pipeline(config: dict) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), config["numeric_features"]),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                config["categorical_features"],
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", KNeighborsClassifier(n_neighbors=config["n_neighbors"])),
        ]
    )
