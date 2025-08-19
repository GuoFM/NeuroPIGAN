from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str) -> Dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # Normalize paths
    data.setdefault("logging", {})
    data.setdefault("data", {})
    data.setdefault("model", {})
    data.setdefault("gan", {})
    data.setdefault("physics", {})

    output_dir = data.get("logging", {}).get("output_dir", "outputs")
    log_dir = data.get("logging", {}).get("log_dir", "logs")
    data["output_dir"] = output_dir
    data["log_dir"] = log_dir

    return data


