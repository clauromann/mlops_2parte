from dotenv import load_dotenv
load_dotenv()

import os
import tempfile
from pathlib import Path

import joblib
import pandas as pd
import wandb
from prefect import flow, task
from prefect.logging import get_run_logger
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from utils.config import load_config
from utils.titanic import build_knn_pipeline, preprocess_titanic


@task(name="Download raw Titanic dataset from W&B")
def load_raw_dataset(run: wandb.sdk.wandb_run.Run, config: dict) -> pd.DataFrame:
    logger = get_run_logger()
    artifact_ref = f"{config['raw_artifact_name']}:{config['raw_artifact_version']}"
    artifact = run.use_artifact(artifact_ref)
    artifact_dir = artifact.download(root=tempfile.gettempdir())
    df = pd.read_csv(Path(artifact_dir) / config["dataset_filename"])
    logger.info(f"Downloaded raw artifact '{artifact_ref}' with {len(df)} rows")
    return df


@task(name="Clean and engineer Titanic features")
def process_dataset(df: pd.DataFrame) -> pd.DataFrame:
    logger = get_run_logger()
    processed = preprocess_titanic(df)
    logger.info(f"Processed dataset: {processed.shape[0]} rows, {processed.shape[1]} columns")
    return processed


@task(name="Persist processed dataset to W&B")
def persist_processed_dataset(
    df: pd.DataFrame,
    run: wandb.sdk.wandb_run.Run,
    config: dict,
) -> None:
    logger = get_run_logger()
    dataset_path = Path(tempfile.gettempdir()) / config["processed_filename"]
    df.to_csv(dataset_path, index=False)

    artifact = wandb.Artifact(
        name=config["processed_artifact_name"],
        type=config["processed_artifact_type"],
        description="Titanic dataset after the predefined cleaning and feature engineering steps.",
    )
    artifact.add_file(str(dataset_path))
    run.log_artifact(artifact)
    logger.info(f"Persisted processed artifact '{config['processed_artifact_name']}'")


@task(name="Load processed dataset from W&B")
def load_processed_dataset(run: wandb.sdk.wandb_run.Run, config: dict) -> pd.DataFrame:
    logger = get_run_logger()
    artifact_ref = (
        f"{config['processed_artifact_name']}:{config['processed_artifact_version']}"
    )
    artifact = run.use_artifact(artifact_ref)
    artifact_dir = artifact.download(root=tempfile.gettempdir())
    df = pd.read_csv(Path(artifact_dir) / config["processed_filename"])
    logger.info(f"Loaded processed artifact '{artifact_ref}' with {len(df)} rows")
    return df


@task(name="Split dataset into train/test")
def split_dataset(
    df: pd.DataFrame, config: dict
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    logger = get_run_logger()
    X = df[config["numeric_features"] + config["categorical_features"]]
    y = df[config["target"]]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["test_size"],
        random_state=config["random_state"],
        stratify=y,
    )
    logger.info(f"Split: {len(X_train)} train rows, {len(X_test)} test rows")
    return X_train, X_test, y_train, y_test


@task(name="Train KNN model")
def train_model(
    X_train: pd.DataFrame, y_train: pd.Series, config: dict
) -> Pipeline:
    logger = get_run_logger()
    model = build_knn_pipeline(config)
    model.fit(X_train, y_train)
    logger.info(f"Trained KNN with n_neighbors={config['n_neighbors']}")
    return model


@task(name="Evaluate model")
def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    run: wandb.sdk.wandb_run.Run,
) -> dict:
    logger = get_run_logger()
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }
    run.log(metrics)
    logger.info(f"Metrics: {metrics}")
    return metrics


@task(name="Register serialized model in W&B")
def register_model(
    model: Pipeline,
    run: wandb.sdk.wandb_run.Run,
    config: dict,
) -> None:
    logger = get_run_logger()
    model_path = Path(tempfile.gettempdir()) / "titanic_knn_decoupled.joblib"
    joblib.dump(model, model_path)

    artifact = wandb.Artifact(
        name=config["model_artifact_name"],
        type=config["model_artifact_type"],
        description="Serialized Titanic KNN model from the decoupled Prefect flow.",
    )
    artifact.add_file(str(model_path))
    run.log_artifact(artifact)
    logger.info(f"Registered model artifact '{config['model_artifact_name']}'")


@flow(name="Titanic Decoupled - Process data")
def process_data_pipeline() -> None:
    config = load_config("f02_decoupled_config")
    wandb.login(key=os.getenv("WANDB_API_KEY"))

    with wandb.init(
        project=config["project"],
        job_type=config["process_job_type"],
        config=config,
        save_code=True,
        dir=tempfile.gettempdir(),
    ) as run:
        wandb.save(os.path.relpath(__file__))
        df = load_raw_dataset(run, config)
        processed = process_dataset(df)
        persist_processed_dataset(processed, run, config)


@flow(name="Titanic Decoupled - Train model")
def train_model_pipeline() -> None:
    config = load_config("f02_decoupled_config")
    wandb.login(key=os.getenv("WANDB_API_KEY"))

    with wandb.init(
        project=config["project"],
        job_type=config["train_job_type"],
        config=config,
        save_code=True,
        dir=tempfile.gettempdir(),
    ) as run:
        wandb.save(os.path.relpath(__file__))
        processed = load_processed_dataset(run, config)
        X_train, X_test, y_train, y_test = split_dataset(processed, config)
        model = train_model(X_train, y_train, config)
        evaluate_model(model, X_test, y_test, run)
        register_model(model, run, config)
