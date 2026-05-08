"""Audio normalization utilities."""

import torch


def rms_normalize(waveform: torch.Tensor, target_rms: float = 0.1) -> torch.Tensor:
    """Normalize waveform to a target RMS level.

    Args:
        waveform: [channels, time]
    """
    rms = waveform.pow(2).mean(dim=-1, keepdim=True).sqrt()
    rms = torch.clamp(rms, min=1e-8)
    return waveform * (target_rms / rms)


def peak_normalize(waveform: torch.Tensor) -> torch.Tensor:
    """Normalize waveform so that peak amplitude is 1.0.

    Args:
        waveform: [channels, time]
    """
    peak = waveform.abs().max(dim=-1, keepdim=True).values
    peak = torch.clamp(peak, min=1e-8)
    return waveform / peak


def clamp_waveform(waveform: torch.Tensor) -> torch.Tensor:
    """Clamp waveform to [-1, 1]."""
    return torch.clamp(waveform, min=-1.0, max=1.0)
