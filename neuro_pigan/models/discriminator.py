from __future__ import annotations

import torch
import torch.nn as nn


class Conv1dCritic(nn.Module):
    """1D convolutional critic for WGAN-GP.

    Expects input of shape [B, 1, T] and outputs a scalar per example.
    """

    def __init__(self, base_channels: int = 32) -> None:
        super().__init__()
        C = base_channels
        self.net = nn.Sequential(
            nn.Conv1d(1, C, kernel_size=9, stride=2, padding=4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(C, C * 2, kernel_size=9, stride=2, padding=4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(C * 2, C * 4, kernel_size=9, stride=2, padding=4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(C * 4, C * 4, kernel_size=9, stride=2, padding=4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(C * 4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, 1, T]
        h = self.net(x)
        out = self.head(h)
        return out.squeeze(-1)


