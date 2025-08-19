from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class Sine(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return torch.sin(x)


def get_activation(name: str) -> nn.Module:
    name = (name or "relu").lower()
    if name == "relu":
        return nn.ReLU(inplace=False)
    if name == "gelu":
        return nn.GELU()
    if name == "sine":
        return Sine()
    raise ValueError(f"Unsupported activation: {name}")


class CoordinateGenerator(nn.Module):
    """Coordinate-based MLP generator.

    Inputs a time coordinate and a latent vector. For a batch with sequence length T,
    concatenates latent z to each timestep coordinate and predicts amplitudes.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        num_layers: int,
        activation: str = "sine",
    ) -> None:
        super().__init__()
        input_dim = latent_dim + 1  # time coordinate + latent vector

        layers: List[nn.Module] = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(get_activation(activation))
        for _ in range(max(num_layers - 2, 0)):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(get_activation(activation))
        layers.append(nn.Linear(hidden_dim, 1))  # scalar amplitude
        self.mlp = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        t: [B, T, 1] in [-1, 1]
        z: [B, Z]
        returns: [B, 1, T]
        """
        assert t.dim() == 3 and t.size(-1) == 1, "t must be [B, T, 1]"
        assert z.dim() == 2, "z must be [B, Z]"
        B, T, _ = t.shape
        z_expanded = z[:, None, :].expand(B, T, z.size(1))
        inp = torch.cat([t, z_expanded], dim=-1)  # [B, T, Z+1]
        out = self.mlp(inp.reshape(B * T, -1))  # [B*T, 1]
        return out.view(B, T, 1).permute(0, 2, 1).contiguous()


