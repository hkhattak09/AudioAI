"""Wav2Vec2 CTC wrapper."""

import torch

from voice_ai.pipeline.schemas import STTResult
from voice_ai.stt.base import SpeechToTextModel
from voice_ai.stt.ctc_decode import ctc_greedy_decode


class Wav2Vec2CTCModel(SpeechToTextModel):
    """Lazy-loading wrapper around Hugging Face Wav2Vec2 CTC."""

    def __init__(self, model_name: str = "facebook/wav2vec2-base-960h", device: str = "cpu"):
        self.model_name = model_name
        self.device = torch.device(device)
        self._processor = None
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        from transformers import AutoModelForCTC, AutoProcessor
        self._processor = AutoProcessor.from_pretrained(self.model_name)
        self._model = AutoModelForCTC.from_pretrained(self.model_name).to(self.device)
        self._model.eval()

    def transcribe(self, waveform: torch.Tensor, sample_rate: int) -> STTResult:
        """Transcribe audio to text.

        Args:
            waveform: [channels, time] or [time]
            sample_rate: sample rate in Hz
        """
        self._load()
        waveform = waveform.detach().cpu().float()
        if waveform.dim() == 2:
            waveform = waveform.mean(dim=0)

        if sample_rate != 16000:
            import torchaudio
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)

        inputs = self._processor(waveform.numpy(), sampling_rate=16000, return_tensors="pt")
        input_values = inputs.input_values.to(self.device)

        with torch.no_grad():
            logits = self._model(input_values).logits

        predicted_ids = ctc_greedy_decode(logits[0])
        transcription = self._processor.decode(predicted_ids)
        return STTResult(text=transcription.strip())
