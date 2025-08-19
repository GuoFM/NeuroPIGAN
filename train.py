import argparse
import os
import sys
from pathlib import Path

import torch

from neuro_pigan.utils.config import load_config
from neuro_pigan.training.trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Physics-Informed GAN for Neuropixels-like signals")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config file")
    parser.add_argument("--device", type=str, default=None, help="Override device, e.g., cuda or cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.device is not None:
        cfg["device"] = args.device

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    # Ensure output directories exist
    Path(cfg.get("output_dir", "outputs")).mkdir(parents=True, exist_ok=True)
    Path(cfg.get("log_dir", "logs")).mkdir(parents=True, exist_ok=True)

    trainer = Trainer(cfg=cfg, device=device)
    trainer.fit()


if __name__ == "__main__":
    main()


