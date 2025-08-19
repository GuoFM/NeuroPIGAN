from __future__ import annotations

from typing import Tuple

import torch


def compute_out_of_band_energy_fraction(
    signal: torch.Tensor,
    sample_rate_hz: int,
    passband_hz: Tuple[float, float],
) -> torch.Tensor:
    """Compute fraction of energy outside [low, high] Hz.

    signal: [B, 1, T]
    returns: scalar tensor
    """
    assert signal.dim() == 3 and signal.size(1) == 1, "signal must be [B, 1, T]"
    B, _, T = signal.shape
    spectrum = torch.fft.rfft(signal, dim=-1)
    power = (spectrum.real ** 2 + spectrum.imag ** 2)  # [B, 1, F]
    freqs = torch.fft.rfftfreq(T, d=1.0 / float(sample_rate_hz)).to(signal.device)

    low, high = float(passband_hz[0]), float(passband_hz[1])
    in_band_mask = (freqs >= low) & (freqs <= high)
    out_band_mask = ~in_band_mask

    total_energy = power.sum(dim=-1).clamp_min(1e-12)
    out_energy = power[..., out_band_mask].sum(dim=-1)
    fraction = (out_energy / total_energy).mean()
    return fraction


def second_derivative_l2(signal: torch.Tensor) -> torch.Tensor:
    """Mean squared second finite difference over time.

    signal: [B, 1, T]
    """
    x = signal
    d2 = x[..., 2:] - 2.0 * x[..., 1:-1] + x[..., :-2]
    return (d2.pow(2).mean())


