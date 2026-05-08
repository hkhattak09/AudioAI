"""HiFiGAN inference utilities."""

import torch

from voice_ai.vocoder.hifigan import Generator


def mel_to_waveform(generator: Generator, mel: torch.Tensor) -> torch.Tensor:
    """Convert mel spectrogram to waveform.

    Args:
        generator: HiFiGAN Generator
        mel: [B, n_mels, T]

    Returns:
        waveform: [B, 1, T']
    """
    generator.eval()
    with torch.no_grad():
        wav = generator(mel)
    return wav
