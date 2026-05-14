from dotenv import load_dotenv
load_dotenv()

import os
import tempfile
from pathlib import Path

import pandas as pd
import wandb
from prefect import flow, task
from prefect.logging import get_run_logger

from utils.config import load_config
from utils.titanic import preprocess_titanic


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
        description="Titanic dataset after predefined cleaning and feature engineering.",
    )
    artifact.add_file(str(dataset_path))
    run.log_artifact(artifact)
    logger.info(f"Persisted processed artifact '{config['processed_artifact_name']}'")


@flow(name="Titanic RunPod - Process data")
def process_data_pipeline() -> None:
    config = load_config("wandb")
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
