from dotenv import load_dotenv
load_dotenv()

import logging
import os
import tempfile
from pathlib import Path
from time import sleep

import joblib
import pandas as pd
import requests
import wandb
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from utils.config import load_config
from utils.titanic import build_knn_pipeline


logger = logging.getLogger(__name__)


def load_processed_dataset(run: wandb.sdk.wandb_run.Run, config: dict) -> pd.DataFrame:
    artifact_ref = f"{config['processed_artifact_name']}:{config['processed_artifact_version']}"
    artifact = run.use_artifact(artifact_ref)
    artifact_dir = artifact.download(root=tempfile.gettempdir())
    df = pd.read_csv(Path(artifact_dir) / config["processed_filename"])
    logger.info("Loaded processed artifact '%s' with %s rows", artifact_ref, len(df))
    return df


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


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, run) -> dict:
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }
    run.log(metrics)
    logger.info("Metrics: %s", metrics)
    return metrics


def register_model(model, run: wandb.sdk.wandb_run.Run, config: dict) -> None:
    model_path = Path(tempfile.gettempdir()) / "model.joblib"
    joblib.dump(model, model_path)

    artifact = wandb.Artifact(
        name=config["model_artifact_name"],
        type=config["model_artifact_type"],
        description="Titanic KNN model trained inside a RunPod pod.",
    )
    artifact.add_file(str(model_path))
    run.log_artifact(artifact)
    logger.info("Registered model artifact '%s'", config["model_artifact_name"])


def train_model_pipeline() -> None:
    config = {**load_config("wandb"), **load_config("training")}
    wandb.login(key=os.getenv("WANDB_API_KEY"))

    with wandb.init(
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
        logger.info("Trained KNN with n_neighbors=%s", config["n_neighbors"])
        evaluate_model(model, X_test, y_test, run)
        register_model(model, run, config)


def stop_pod() -> None:
    pod_id = os.getenv("RUNPOD_POD_ID")
    api_key = os.getenv("RUNPOD_MANAGE_KEY")
    if not pod_id or not api_key:
        logger.warning("RUNPOD_POD_ID or RUNPOD_MANAGE_KEY not set, skipping pod stop")
        return

    url = f"https://rest.runpod.io/v1/pods/{pod_id}/stop"
    response = requests.post(url, headers={"Authorization": f"Bearer {api_key}"})
    response.raise_for_status()
    logger.info("Pod %s stopped successfully", pod_id)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_model_pipeline()
    logger.info("Sleeping 60 seconds before stopping pod to allow log inspection")
    for remaining in range(60, 0, -10):
        logger.info("%s seconds until pod stops", remaining)
        sleep(10)
    stop_pod()
