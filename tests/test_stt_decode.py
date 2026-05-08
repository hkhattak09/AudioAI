import torch
from unittest.mock import MagicMock

from voice_ai.stt.ctc_decode import ctc_greedy_decode
from voice_ai.stt.wav2vec2_ctc import Wav2Vec2CTCModel


def test_ctc_greedy_decode_2d():
    # [T=5, vocab=4], blank=0
    logits = torch.tensor([
        [0.0, 1.0, 0.0, 0.0],  # -> 1
        [0.0, 1.0, 0.0, 0.0],  # -> 1 (repeat, collapsed)
        [0.0, 0.0, 1.0, 0.0],  # -> 2
        [1.0, 0.0, 0.0, 0.0],  # -> 0 (blank)
        [0.0, 0.0, 0.0, 1.0],  # -> 3
    ])
    result = ctc_greedy_decode(logits, blank_id=0)
    assert result == [1, 2, 3]


def test_ctc_greedy_decode_3d():
    logits = torch.zeros(2, 5, 3)
    # batch 0: 1,1,2,0,1 -> [1,2,1]
    logits[0, 0, 1] = 1.0
    logits[0, 1, 1] = 1.0
    logits[0, 2, 2] = 1.0
    logits[0, 3, 0] = 1.0
    logits[0, 4, 1] = 1.0
    # batch 1: 2,2,0,0,2 -> [2, 2]
    logits[1, 0, 2] = 1.0
    logits[1, 1, 2] = 1.0
    logits[1, 2, 0] = 1.0
    logits[1, 3, 0] = 1.0
    logits[1, 4, 2] = 1.0
    results = ctc_greedy_decode(logits, blank_id=0)
    assert results == [[1, 2, 1], [2, 2]]


def test_wav2vec2_transcribe_with_grad_tensor():
    """Wav2Vec2 wrapper should handle tensors with requires_grad=True."""
    model = Wav2Vec2CTCModel()
    fake_processor = MagicMock()
    fake_processor.return_value = MagicMock(input_values=torch.zeros(1, 10))
    fake_model = MagicMock()
    fake_logits = torch.zeros(1, 5, 3)
    fake_logits[0, 0, 1] = 1.0
    fake_logits[0, 1, 2] = 1.0
    fake_model.return_value = MagicMock(logits=fake_logits)
    fake_processor.decode = MagicMock(return_value="hello")

    model._processor = fake_processor
    model._model = fake_model

    waveform = torch.randn(1, 16000, requires_grad=True)
    result = model.transcribe(waveform, sample_rate=16000)
    assert result.text == "hello"
