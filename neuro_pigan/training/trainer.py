from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, Iterator, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

from neuro_pigan.data.dataset import NeuropixelsDataset
from neuro_pigan.models.generator import CoordinateGenerator
from neuro_pigan.models.discriminator import Conv1dCritic
from neuro_pigan.losses.wgan_gp import gradient_penalty
from neuro_pigan.training.physics_constraints import compute_physics_penalties
from neuro_pigan.utils.logger import TBLogger


def _cycle(loader: DataLoader) -> Iterator[torch.Tensor]:
    while True:
        for batch in loader:
            yield batch


class Trainer:
    def __init__(self, cfg: Dict, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device

        # Directories
        self.output_dir = Path(cfg.get("output_dir", "outputs"))
        self.log_dir = Path(cfg.get("log_dir", "logs"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Data
        data_cfg = cfg.get("data", {})
        self.seq_len = int(data_cfg.get("seq_len", 2048))
        self.sample_rate_hz = int(data_cfg.get("sample_rate_hz", 30000))
        self.batch_size = int(data_cfg.get("batch_size", 32))
        use_synth = bool(data_cfg.get("use_synthetic_if_empty", True))

        dataset = NeuropixelsDataset(
            root=data_cfg.get("root", "data"),
            seq_len=self.seq_len,
            sample_rate_hz=self.sample_rate_hz,
            use_synthetic_if_empty=use_synth,
        )
        self.loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=int(data_cfg.get("num_workers", 0)))
        self.data_iter = _cycle(self.loader)

        # Models
        model_cfg = cfg.get("model", {})
        self.latent_dim = int(model_cfg.get("latent_dim", 64))
        hidden_dim = int(model_cfg.get("hidden_dim", 256))
        num_layers = int(model_cfg.get("num_layers", 5))
        activation = model_cfg.get("activation", "sine")

        self.G = CoordinateGenerator(self.latent_dim, hidden_dim, num_layers, activation).to(self.device)
        self.D = Conv1dCritic(base_channels=32).to(self.device)

        # Optimizers
        gan_cfg = cfg.get("gan", {})
        lr_g = float(gan_cfg.get("lr_g", 2e-4))
        lr_d = float(gan_cfg.get("lr_d", 2e-4))
        betas = tuple(gan_cfg.get("betas", [0.5, 0.9]))  # type: ignore
        self.opt_g = optim.Adam(self.G.parameters(), lr=lr_g, betas=betas)
        self.opt_d = optim.Adam(self.D.parameters(), lr=lr_d, betas=betas)

        # Training schedule
        self.epochs = int(gan_cfg.get("epochs", 50))
        self.d_iters = int(gan_cfg.get("d_iters", 5))
        self.g_iters = int(gan_cfg.get("g_iters", 1))
        self.lambda_gp = float(gan_cfg.get("lambda_gp", 10.0))

        # Physics
        phys_cfg = cfg.get("physics", {})
        self.weight_band = float(phys_cfg.get("weight_bandlimit", 1.0))
        self.passband_hz = tuple(phys_cfg.get("passband_hz", [300, 6000]))  # type: ignore
        self.weight_smooth = float(phys_cfg.get("weight_smoothness", 0.5))
        self.weight_amp = float(phys_cfg.get("weight_amplitude", 0.5))
        self.amp_range = tuple(phys_cfg.get("amplitude_range", [-300e-6, 300e-6]))  # type: ignore
        self.weight_sparse = float(phys_cfg.get("weight_sparsity", 0.25))

        # Logging schedule
        log_cfg = cfg.get("logging", {})
        self.log_every = int(log_cfg.get("log_every_steps", 50))
        self.sample_every = int(log_cfg.get("sample_every_steps", 200))
        self.ckpt_every_epochs = int(log_cfg.get("ckpt_every_epochs", 5))

        self.logger = TBLogger(str(self.log_dir))
        self.global_step = 0

    # ---------------------------- Helpers ----------------------------
    def _time_coords(self, batch_size: int, seq_len: int) -> torch.Tensor:
        t = torch.linspace(-1.0, 1.0, steps=seq_len, device=self.device)
        t = t.view(1, seq_len, 1).expand(batch_size, seq_len, 1)
        return t

    def _sample_z(self, batch_size: int) -> torch.Tensor:
        return torch.randn(batch_size, self.latent_dim, device=self.device)

    def _physics_loss(self, fake: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        penalties = compute_physics_penalties(
            signal=fake,
            sample_rate_hz=self.sample_rate_hz,
            passband_hz=self.passband_hz,  # type: ignore
            amplitude_range=self.amp_range,  # type: ignore
        )
        loss = (
            self.weight_band * penalties["bandlimit"]
            + self.weight_smooth * penalties["smoothness"]
            + self.weight_amp * penalties["amplitude"]
            + self.weight_sparse * penalties["sparsity"]
        )
        scalars = {k: float(v.detach().item()) for k, v in penalties.items()}
        scalars["physics_total"] = float(loss.detach().item())
        return loss, scalars

    # ---------------------------- Train -----------------------------
    def fit(self) -> None:
        steps_per_epoch = max(len(self.loader), 1)
        for epoch in range(1, self.epochs + 1):
            for _ in range(steps_per_epoch):
                # Critic updates
                for _ in range(self.d_iters):
                    real = next(self.data_iter).to(self.device)  # [B, 1, T]
                    if real.dim() == 2:
                        real = real.unsqueeze(1)
                    batch = real.size(0)
                    t = self._time_coords(batch, self.seq_len)
                    z = self._sample_z(batch)
                    with torch.no_grad():
                        fake = self.G(t, z)

                    self.opt_d.zero_grad(set_to_none=True)
                    d_real = self.D(real)
                    d_fake = self.D(fake)
                    gp = gradient_penalty(self.D, real, fake, self.device)
                    d_loss = (d_fake.mean() - d_real.mean()) + self.lambda_gp * gp
                    d_loss.backward()
                    self.opt_d.step()

                # Generator updates
                for _ in range(self.g_iters):
                    real = next(self.data_iter).to(self.device)  # for logging scale alignment
                    if real.dim() == 2:
                        real = real.unsqueeze(1)
                    batch = real.size(0)
                    t = self._time_coords(batch, self.seq_len)
                    z = self._sample_z(batch)

                    self.opt_g.zero_grad(set_to_none=True)
                    fake = self.G(t, z)
                    adv = -self.D(fake).mean()
                    phys_loss, phys_scalars = self._physics_loss(fake)
                    g_loss = adv + phys_loss
                    g_loss.backward()
                    self.opt_g.step()

                # Logging
                if self.global_step % self.log_every == 0:
                    scalars = {
                        "adv_g": float(adv.detach().item()),
                        "g_total": float(g_loss.detach().item()),
                        **phys_scalars,
                    }
                    self.logger.log_scalars("train", scalars, step=self.global_step)

                # Sampling
                if self.global_step % self.sample_every == 0:
                    self._save_samples(self.global_step)

                self.global_step += 1

            # Checkpointing
            if epoch % self.ckpt_every_epochs == 0:
                self._save_checkpoint(epoch)

        self.logger.close()

    # ---------------------------- I/O -------------------------------
    def _save_samples(self, step: int) -> None:
        with torch.no_grad():
            B = 4
            t = self._time_coords(B, self.seq_len)
            z = self._sample_z(B)
            fake = self.G(t, z).detach().cpu().numpy()  # [B, 1, T]
        out_path = self.output_dir / f"samples_step_{step:07d}.npy"
        np.save(str(out_path), fake)

    def _save_checkpoint(self, epoch: int) -> None:
        ckpt = {
            "G": self.G.state_dict(),
            "D": self.D.state_dict(),
            "opt_g": self.opt_g.state_dict(),
            "opt_d": self.opt_d.state_dict(),
            "cfg": self.cfg,
            "global_step": self.global_step,
            "epoch": epoch,
        }
        path = self.output_dir / f"ckpt_epoch_{epoch:04d}.pt"
        torch.save(ckpt, str(path))


