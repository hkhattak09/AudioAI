"""Pipeline data schemas."""

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class AudioChunk:
    waveform: torch.Tensor
    sample_rate: int
    start_seconds: float = 0.0
    end_seconds: float = 0.0


@dataclass
class STTResult:
    text: str
    confidence: float | None = None
    words: list[dict[str, Any]] | None = None


@dataclass
class LLMResult:
    text: str
    finish_reason: str | None = None
    usage: dict[str, int] | None = None


@dataclass
class TTSResult:
    mel: torch.Tensor | None = None
    waveform: torch.Tensor | None = None
    sample_rate: int = 22050


@dataclass
class VoiceTurn:
    transcript: str = ""
    llm_response: str = ""
    output_audio: torch.Tensor | None = None
    output_sample_rate: int = 22050
    metadata: dict[str, Any] = field(default_factory=dict)
