# Voice AI

Local scaffold for a PyTorch-based voice AI pipeline.

## Pipeline

```text
audio input -> VAD/chunking -> STT -> LLM API -> TTS text-to-mel -> vocoder mel-to-waveform -> audio output
```

## Local Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pytest -q tests -m "not slow and not gpu and not download" --maxfail=1
python scripts/smoke_test.py
```

## What runs locally

- Smoke tests and shape checks
- Fake pipeline end-to-end
- Import and compilation checks

## What only runs on Colab

- Real model training
- Fine-tuning Wav2Vec2
- Training Tacotron2 and HiFiGAN
- Full voice pipeline with real checkpoints

## Device strategy

- Local 8GB M2: smoke tests only
- Colab T4: default training
- Colab L4/A100: larger configs

## Current limitations

- No trained checkpoints included
- Notebooks are handoff skeletons
- Real training comes later
