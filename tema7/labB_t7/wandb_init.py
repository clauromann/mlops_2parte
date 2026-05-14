from dotenv import load_dotenv
load_dotenv()

import os
import tempfile

import wandb

from utils.config import PROJECT_DIR, load_config


def main() -> None:
    config = load_config("wandb_init")
    wandb.login(key=os.getenv("WANDB_API_KEY"))

    with wandb.init(
        project=config["project"],
        job_type="upload-raw-dataset",
        config=config,
        dir=tempfile.gettempdir(),
    ) as run:
        artifact = wandb.Artifact(
            name=config["artifact_name"],
            type=config["artifact_type"],
            description="Titanic raw dataset and cleaning instructions.",
        )
        artifact.add_file(str(PROJECT_DIR / "utils" / "data" / config["dataset_filename"]))
        artifact.add_file(
            str(PROJECT_DIR / "utils" / "data" / config["cleaning_steps_filename"])
        )
        run.log_artifact(artifact)


if __name__ == "__main__":
    main()
