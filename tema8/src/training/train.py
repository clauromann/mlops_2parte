import json
import os
import tempfile
from pathlib import Path

import joblib
import pandas as pd
import wandb
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from training.model import build_knn_pipeline


def load_config() -> dict:
    config_path = Path(__file__).resolve().parent / "config.json"
    return json.loads(config_path.read_text(encoding="utf-8"))


def load_processed_dataset(run: wandb.sdk.wandb_run.Run, config: dict) -> pd.DataFrame:
    artifact_ref = (
        f"{config['processed_artifact_name']}:{config['processed_artifact_version']}"
    )
    artifact = run.use_artifact(artifact_ref)
    artifact_dir = artifact.download(root=tempfile.gettempdir())
    return pd.read_csv(Path(artifact_dir) / config["processed_filename"])


def split_dataset(
    df: pd.DataFrame, config: dict
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = df[config["numeric_features"] + config["categorical_features"]]
    y = df[config["target"]]
    return train_test_split(
        X,
        y,
        test_size=config["test_size"],
        random_state=config["random_state"],
        stratify=y,
    )


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }


def register_model(model, run: wandb.sdk.wandb_run.Run, config: dict) -> None:
    model_path = Path(tempfile.gettempdir()) / "model.joblib"
    joblib.dump(model, model_path)
    artifact = wandb.Artifact(
        name=config["model_artifact_name"],
        type=config["model_artifact_type"],
    )
    artifact.add_file(str(model_path))
    run.log_artifact(artifact)


def main() -> None:
    full_config = load_config()
    config = {**full_config["wandb"], **full_config["training"]}

    wandb.login(key=os.getenv("WANDB_API_KEY"))
    with wandb.init(
        entity=os.getenv("WANDB_ENTITY", config["entity"]),
        project=config["project"],
        job_type=config["train_job_type"],
        config=config,
        save_code=True,
        dir=tempfile.gettempdir(),
    ) as run:
        df = load_processed_dataset(run, config)
        X_train, X_test, y_train, y_test = split_dataset(df, config)
        model = build_knn_pipeline(config)
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        run.log(metrics)
        register_model(model, run, config)
        print(metrics)


if __name__ == "__main__":
    main()
