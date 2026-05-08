"""STT base interface."""

from abc import ABC, abstractmethod

from voice_ai.pipeline.schemas import STTResult


class SpeechToTextModel(ABC):
    @abstractmethod
    def transcribe(self, waveform, sample_rate: int) -> STTResult:
        """Transcribe audio waveform to text.

        Args:
            waveform: torch.Tensor of shape [channels, time] or [time]
            sample_rate: audio sample rate in Hz

        Returns:
            STTResult
        """
        ...
