import pandas as pd


def drop_cabin(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=["Cabin"], errors="ignore")


def impute_embarked(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["Embarked"] = result["Embarked"].fillna("S")
    return result


def impute_fare(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["Fare"] = result["Fare"].fillna(result["Fare"].mean())
    return result


def impute_age_by_sex_and_class(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["Age"] = result.groupby(["Sex", "Pclass"])["Age"].transform(
        lambda values: values.fillna(values.median())
    )
    result["Age"] = result["Age"].astype(int)
    return result


def drop_redundant_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(
        columns=["Name", "Parch", "SibSp", "Ticket", "PassengerId", "Family_size"],
        errors="ignore",
    )
