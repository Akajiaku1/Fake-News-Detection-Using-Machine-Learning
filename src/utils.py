import json
import os
from typing import Dict
from pathlib import Path

def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)

def save_json(d: Dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(d, f, indent=2)
