"""Smoke test script."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from voice_ai.pipeline.voice_agent import VoiceAgent
from voice_ai.training.device import get_device
from voice_ai.utils.config import load_yaml


def main():
    device = get_device()
    print(f"Device: {device}")

    config = load_yaml("configs/pipeline_local.yaml")
    agent = VoiceAgent.from_config(config)

    # Fake audio input: 1 second of sine wave at 22050 Hz
    t = torch.arange(0, 22050) / 22050
    waveform = torch.sin(2 * 3.14159 * 440 * t).unsqueeze(0)
    turn = agent.process_audio(waveform, sample_rate=22050)

    print(f"Transcript: {turn.transcript}")
    print(f"LLM response: {turn.llm_response}")
    if turn.output_audio is not None:
        print(f"Output waveform shape: {turn.output_audio.shape}")
    else:
        print("Output waveform shape: None")


if __name__ == "__main__":
    main()
