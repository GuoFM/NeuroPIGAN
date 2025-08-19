from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

from torch.utils.tensorboard import SummaryWriter


class TBLogger:
    def __init__(self, log_dir: str) -> None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir)

    def log_scalars(self, tag: str, scalars: Dict[str, float], step: int) -> None:
        for key, value in scalars.items():
            self.writer.add_scalar(f"{tag}/{key}", float(value), step)

    def close(self) -> None:
        self.writer.close()


