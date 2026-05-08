"""Tacotron2 inference utilities."""

import torch

from voice_ai.tts.tacotron2 import Tacotron2, Tacotron2Config
from voice_ai.tts.tokenizer import CharTokenizer


def synthesize_mel(
    model: Tacotron2,
    tokenizer: CharTokenizer,
    text: str,
    config: Tacotron2Config,
    device: torch.device,
) -> torch.Tensor:
    """Synthesize mel spectrogram from text.

    Args:
        model: Tacotron2 model
        tokenizer: CharTokenizer
        text: input text string
        config: Tacotron2Config
        device: torch device

    Returns:
        mel: [1, n_mels, T_mel]
    """
    model.eval()
    ids = tokenizer.encode(text).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model.inference(ids)
    return outputs["mel_after_postnet"]
