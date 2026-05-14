import json
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]


def load_config(name: str) -> dict:
    config_path = PROJECT_DIR / "config" / f"{name}.json"
    return json.loads(config_path.read_text(encoding="utf-8"))
