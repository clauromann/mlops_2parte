import os
import shutil
import tempfile
from pathlib import Path

import wandb

from utils.config import load_config


def download_model(config: dict | None = None) -> None:
    if config is None:
        config = load_config("wandb")

    wandb.login(key=os.getenv("WANDB_API_KEY"))
    api = wandb.Api()
    artifact_ref = (
        f"{config['entity']}/{config['project']}/"
        f"{config['model_artifact_name']}:latest"
    )
    artifact = api.artifact(artifact_ref)
    artifact_dir = artifact.download(root=tempfile.gettempdir())

    output_dir = Path(__file__).resolve().parent / "model"
    output_dir.mkdir(exist_ok=True)
    shutil.copy2(Path(artifact_dir) / "model.joblib", output_dir / "model.joblib")
    print(f"Model saved to {output_dir / 'model.joblib'}")


if __name__ == "__main__":
    download_model()
