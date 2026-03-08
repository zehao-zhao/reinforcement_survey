from pathlib import Path
import yaml


def load_config(path: str):
    with Path(path).open() as f:
        return yaml.safe_load(f)
