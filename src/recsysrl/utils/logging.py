import json
from pathlib import Path
from datetime import datetime


def init_run_dir(base: str = "runs", name: str = "run") -> Path:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    p = Path(base) / f"{name}_{ts}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def append_jsonl(path: Path, obj: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(obj) + "\n")
