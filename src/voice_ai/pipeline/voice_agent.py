"""End-to-end voice pipeline."""

import torch

from voice_ai.audio.vad import detect_speech_segments
from voice_ai.pipeline.schemas import STTResult, VoiceTurn
from voice_ai.stt.base import SpeechToTextModel
from voice_ai.llm.base import LLMClient
from voice_ai.llm.local_stub import LocalStubLLM


class FakeSTT(SpeechToTextModel):
    """Fake STT that returns fixed text."""

    def transcribe(self, waveform: torch.Tensor, sample_rate: int) -> STTResult:
        return STTResult(text="hello from fake stt")


class FakeTTS:
    """Fake TTS that returns a random mel."""

    def __init__(self, n_mels: int = 80, n_frames: int = 50):
        self.n_mels = n_mels
        self.n_frames = n_frames

    def synthesize(self, text: str) -> torch.Tensor:
        return torch.randn(1, self.n_mels, self.n_frames)


class FakeVocoder:
    """Fake vocoder that returns zeros."""

    def __init__(self, output_samples: int = 1000):
        self.output_samples = output_samples

    def generate(self, mel: torch.Tensor) -> torch.Tensor:
        B = mel.shape[0]
        return torch.zeros(B, 1, self.output_samples)


class VoiceAgent:
    """End-to-end voice agent pipeline."""

    def __init__(
        self,
        stt: SpeechToTextModel | None = None,
        llm: LLMClient | None = None,
        tts: FakeTTS | None = None,
        vocoder: FakeVocoder | None = None,
        vad_enabled: bool = True,
    ):
        self.stt = stt or FakeSTT()
        self.llm = llm or None
        self.tts = tts or FakeTTS()
        self.vocoder = vocoder or FakeVocoder()
        self.vad_enabled = vad_enabled

    @classmethod
    def from_config(cls, config: dict) -> "VoiceAgent":
        """Build a VoiceAgent from a config dict.

        Currently supports fake local components only.
        """
        llm_config = config.get("llm", {})
        llm_type = llm_config.get("type", "local_stub")
        if llm_type == "local_stub":
            llm = LocalStubLLM()
        else:
            llm = None

        tts_config = config.get("tts", {})
        vocoder_config = config.get("vocoder", {})

        tts = FakeTTS(
            n_mels=tts_config.get("n_mels", 80),
            n_frames=tts_config.get("n_frames", 50),
        )
        vocoder = FakeVocoder(
            output_samples=vocoder_config.get("output_samples", 1000),
        )
        return cls(stt=FakeSTT(), llm=llm, tts=tts, vocoder=vocoder)

    def process_audio(self, waveform: torch.Tensor, sample_rate: int) -> VoiceTurn:
        """Process audio through the full pipeline.

        Args:
            waveform: [channels, time]
            sample_rate: sample rate in Hz

        Returns:
            VoiceTurn
        """
        # VAD
        if self.vad_enabled:
            segments = detect_speech_segments(waveform, sample_rate)
            if not segments:
                return VoiceTurn(transcript="", llm_response="", output_audio=None)

        # STT
        stt_result = self.stt.transcribe(waveform, sample_rate)
        transcript = stt_result.text

        # LLM
        if self.llm is not None:
            llm_result = self.llm.generate(transcript)
            response = llm_result.text
        else:
            response = "Hello. This is a local stub response."

        # TTS
        mel = self.tts.synthesize(response)

        # Vocoder
        wav = self.vocoder.generate(mel)

        return VoiceTurn(
            transcript=transcript,
            llm_response=response,
            output_audio=wav,
            output_sample_rate=sample_rate,
        )
