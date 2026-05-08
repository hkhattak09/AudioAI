import pytest
import torch

from voice_ai.audio.features import AudioFeatureConfig, mel_spectrogram, normalize_mel, denormalize_mel


def test_mel_spectrogram_shape():
    config = AudioFeatureConfig(sample_rate=16000, n_fft=512, hop_length=128, win_length=512, n_mels=16)
    waveform = torch.sin(2 * 3.14159 * 440 * torch.arange(0, 16000) / 16000).unsqueeze(0)
    mel = mel_spectrogram(waveform, config)
    assert mel.shape[0] == config.n_mels


def test_normalize_denormalize_roundtrip():
    config = AudioFeatureConfig(n_mels=16)
    waveform = torch.randn(1, 16000)
    mel = mel_spectrogram(waveform, config)
    norm, mean, std = normalize_mel(mel)
    back = denormalize_mel(norm, mean, std)
    assert torch.allclose(back, mel, atol=1e-5)


def test_normalize_mel_constant_tensor():
    mel = torch.zeros(2, 3)
    norm, mean, std = normalize_mel(mel)
    assert torch.allclose(norm, torch.zeros_like(mel))
    assert mean == 0.0
    assert std == 1.0
