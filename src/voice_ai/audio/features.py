"""Audio feature extraction utilities."""

from dataclasses import dataclass

import torch
import torchaudio.transforms as T


@dataclass
class AudioFeatureConfig:
    sample_rate: int = 22050
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mels: int = 80
    f_min: float = 0.0
    f_max: float | None = 8000.0
    mel_norm: str | None = "slaney"  # or None
    mel_scale: str = "htk"  # or "slaney"


def mel_spectrogram(
    waveform: torch.Tensor,
    config: AudioFeatureConfig,
) -> torch.Tensor:
    """Compute mel spectrogram.

    Args:
        waveform: [channels, time] or [batch, channels, time].
            If stereo, mean is taken across channels.

    Returns:
        mel: [n_mels, frames] for single channel input,
             or [batch, n_mels, frames] for batched input.
    """
    if waveform.dim() == 3:
        # [B, C, T] -> [B, T]
        waveform = waveform.mean(dim=1)
    elif waveform.dim() == 2:
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        # [1, T] -> [T]
        waveform = waveform.squeeze(0)

    mel_transform = T.MelSpectrogram(
        sample_rate=config.sample_rate,
        n_fft=config.n_fft,
        win_length=config.win_length,
        hop_length=config.hop_length,
        f_min=config.f_min,
        f_max=config.f_max,
        n_mels=config.n_mels,
        norm=config.mel_norm,
        mel_scale=config.mel_scale,
    )
    mel = mel_transform(waveform)
    return mel


def amplitude_to_db(mel: torch.Tensor, amin: float = 1e-5, top_db: float = 80.0) -> torch.Tensor:
    """Convert amplitude mel to decibel scale."""
    mel = torch.clamp(mel, min=amin)
    db = 20.0 * torch.log10(mel)
    return torch.clamp(db, min=db.max() - top_db)


def db_to_amplitude(db: torch.Tensor) -> torch.Tensor:
    """Convert decibel mel back to amplitude."""
    return torch.pow(10.0, db / 20.0)


def normalize_mel(mel: torch.Tensor) -> tuple[torch.Tensor, float, float]:
    """Normalize mel to zero mean and unit variance.

    Returns:
        normalized, mean, std
    """
    mean = mel.mean()
    std = mel.std()
    std_value = std.item()
    if std_value == 0.0:
        std_value = 1.0
        std = mel.new_tensor(std_value)
    return (mel - mean) / std, mean.item(), std_value


def denormalize_mel(mel: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    """Denormalize mel."""
    return mel * std + mean
