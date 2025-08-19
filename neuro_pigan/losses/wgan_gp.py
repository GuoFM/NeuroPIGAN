from __future__ import annotations

from typing import Tuple

import torch


def gradient_penalty(critic, real: torch.Tensor, fake: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Compute WGAN-GP gradient penalty."""
    batch_size = real.size(0)
    eps = torch.rand(batch_size, 1, 1, device=device)
    x_hat = eps * real + (1.0 - eps) * fake
    x_hat.requires_grad_(True)
    d_hat = critic(x_hat)
    grad = torch.autograd.grad(
        outputs=d_hat.sum(),
        inputs=x_hat,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grad = grad.view(batch_size, -1)
    gp = ((grad.norm(p=2, dim=1) - 1.0) ** 2).mean()
    return gp


