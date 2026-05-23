import pandas as pd


RARE_TITLES = [
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

TITLE_NORMALIZATION = {
    "Mlle": "Miss",
    "Ms": "Miss",
    "Mme": "Mrs",
}


def extract_title(name: str) -> str:
    title = pd.Series([name]).str.extract(r",\s*([^\.]+)\.", expand=False).iloc[0]
    return str(title).strip()


def normalize_title(title: str) -> str:
    normalized = TITLE_NORMALIZATION.get(title, title)
    if normalized in RARE_TITLES:
        return "Rare"
    return normalized


def add_title_feature(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["Title"] = result["Name"].apply(extract_title).apply(normalize_title)
    return result


def categorize_family_size(size: int) -> str:
    if size == 1:
        return "Alone"
    if size <= 4:
        return "Small"
    return "Large"


def add_family_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["Family_size"] = result["SibSp"] + result["Parch"] + 1
    result["Family_size_cat"] = result["Family_size"].apply(categorize_family_size)
    return result
