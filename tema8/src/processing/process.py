import json
import os
import tempfile
from pathlib import Path

import pandas as pd
import wandb

from processing.pipeline import preprocess_titanic


PROJECT_DIR = Path(__file__).resolve().parents[2]


def load_config() -> dict:
    config_path = Path(__file__).resolve().parent / "config.json"
    return json.loads(config_path.read_text(encoding="utf-8"))


def log_artifact(run, name: str, artifact_type: str, files: list[Path]) -> None:
    artifact = wandb.Artifact(name=name, type=artifact_type)
    for file_path in files:
        artifact.add_file(str(file_path))
    run.log_artifact(artifact)


def main() -> None:
    config = load_config()
    wandb_cfg = config["wandb"]
    data_cfg = config["data"]

    raw_path = PROJECT_DIR / data_cfg["raw_path"]
    cleaning_steps_path = PROJECT_DIR / data_cfg["cleaning_steps_path"]
    output_dir = PROJECT_DIR / data_cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / wandb_cfg["processed_filename"]

    raw_df = pd.read_csv(raw_path)
    processed_df = preprocess_titanic(raw_df)
    processed_df.to_csv(output_path, index=False)

    wandb.login(key=os.getenv("WANDB_API_KEY"))
    with wandb.init(
        entity=os.getenv("WANDB_ENTITY", wandb_cfg["entity"]),
        project=wandb_cfg["project"],
        job_type=wandb_cfg["process_job_type"],
        config=config,
        save_code=True,
        dir=tempfile.gettempdir(),
    ) as run:
        run.log(
            {
                "raw_rows": len(raw_df),
                "processed_rows": len(processed_df),
                "processed_columns": processed_df.shape[1],
            }
        )
        log_artifact(
            run,
            wandb_cfg["raw_artifact_name"],
            wandb_cfg["raw_artifact_type"],
            [raw_path, cleaning_steps_path],
        )
        log_artifact(
            run,
            wandb_cfg["processed_artifact_name"],
            wandb_cfg["processed_artifact_type"],
            [output_path],
        )


if __name__ == "__main__":
    main()
