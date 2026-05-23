from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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
