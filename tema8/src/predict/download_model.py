import json
import os
import shutil
import tempfile
from pathlib import Path

import wandb


def load_config() -> dict:
    config_path = Path(__file__).resolve().parent / "config.json"
    return json.loads(config_path.read_text(encoding="utf-8"))["wandb"]


def download_model() -> None:
    config = load_config()
    entity = os.getenv("WANDB_ENTITY", config["entity"])
    artifact_ref = (
        f"{entity}/{config['project']}/"
        f"{config['model_artifact_name']}:{config['model_artifact_version']}"
    )

    wandb.login(key=os.getenv("WANDB_API_KEY"))
    api = wandb.Api()
    artifact = api.artifact(artifact_ref)
    artifact_dir = artifact.download(root=tempfile.gettempdir())

    output_dir = Path(__file__).resolve().parent / "model"
    output_dir.mkdir(exist_ok=True)
    shutil.copy2(Path(artifact_dir) / "model.joblib", output_dir / "model.joblib")
    print(f"Model saved to {output_dir / 'model.joblib'}")


if __name__ == "__main__":
    download_model()
