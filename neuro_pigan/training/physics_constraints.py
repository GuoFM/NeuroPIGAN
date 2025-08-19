from __future__ import annotations

from typing import Dict, Tuple

import torch

from neuro_pigan.utils.signal import (
    compute_out_of_band_energy_fraction,
    second_derivative_l2,
)


def compute_physics_penalties(
    signal: torch.Tensor,
    sample_rate_hz: int,
    passband_hz: Tuple[float, float],
    amplitude_range: Tuple[float, float],
) -> Dict[str, torch.Tensor]:
    """Compute physics-inspired penalties for Neuropixels-like extracellular signals.

    signal: [B, 1, T] in Volts
    returns dict of individual penalties
    """
    penalties: Dict[str, torch.Tensor] = {}

    # Spectral band-limiting: keep energy within passband
    penalties["bandlimit"] = compute_out_of_band_energy_fraction(signal, sample_rate_hz, passband_hz)

    # Smoothness in time (discourage extremely jagged waveforms; spikes remain via amplitude)
    penalties["smoothness"] = second_derivative_l2(signal)

    # Amplitude range: discourage values beyond given physical range
    low, high = float(amplitude_range[0]), float(amplitude_range[1])
    over = torch.relu(signal - high)
    under = torch.relu(low - signal)
    penalties["amplitude"] = (over.abs() + under.abs()).mean()

    # Sparsity prior: encourage mostly near-zero baseline with sparse excursions
    penalties["sparsity"] = signal.abs().mean()

    return penalties


