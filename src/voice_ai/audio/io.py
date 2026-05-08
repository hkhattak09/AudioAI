"""Audio I/O utilities."""

from pathlib import Path

import torch
import torchaudio


def load_audio(
    path: str | Path,
    sample_rate: int | None = None,
    mono: bool = True,
) -> tuple[torch.Tensor, int]:
    """Load audio from path.

    Returns:
        waveform: Tensor of shape [channels, time]
        sr: Sample rate of the returned waveform
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    waveform, sr = torchaudio.load(str(path))
    # waveform shape from torchaudio is [channels, time]

    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sample_rate is not None and sr != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
        waveform = resampler(waveform)
        sr = sample_rate

    return waveform, sr


def save_audio(
    path: str | Path,
    waveform: torch.Tensor,
    sample_rate: int,
) -> None:
    """Save audio to path.

    Args:
        waveform: Tensor of shape [channels, time]
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(path), waveform, sample_rate)
