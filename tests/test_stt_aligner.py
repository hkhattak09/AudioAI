import pytest
import torch

from voice_ai.stt.aligner import (
    AlignmentError,
    align_words,
    forced_align_ctc,
    get_trellis,
)


def _emission(best_ids: list[int], vocab_size: int = 4) -> torch.Tensor:
    logits = torch.full((len(best_ids), vocab_size), -8.0)
    for frame, token_id in enumerate(best_ids):
        logits[frame, token_id] = 8.0
    return torch.log_softmax(logits, dim=-1)


def test_forced_align_ctc_produces_word_spans():
    # blank=0, A=1, |=2, B=3
    emission = _emission([0, 1, 0, 2, 3, 0])
    tokens = [1, 2, 3]
    id_to_token = {0: "<pad>", 1: "A", 2: "|", 3: "B"}

    result = forced_align_ctc(
        emission=emission,
        tokens=tokens,
        id_to_token=id_to_token,
        blank_id=0,
        duration_sec=6.0,
    )

    assert [span.word for span in result.word_spans] == ["A", "B"]
    assert result.num_frames == 6
    assert result.seconds_per_frame == 1.0
    assert result.average_word_confidence > 0.99
    assert result.word_spans[0].start_sec < result.word_spans[1].start_sec
    assert result.word_spans[0].end_sec <= result.word_spans[1].start_sec
    assert result.to_dict()["word_spans"][0]["word"] == "A"


def test_get_trellis_shape():
    emission = _emission([0, 1, 2, 3])
    trellis = get_trellis(emission, tokens=[1, 2, 3], blank_id=0)
    assert trellis.shape == (5, 4)
    assert torch.isfinite(trellis[-1, -1])


def test_forced_align_ctc_rejects_too_many_tokens():
    emission = _emission([1, 2])
    with pytest.raises(AlignmentError, match="too many tokens"):
        forced_align_ctc(
            emission=emission,
            tokens=[1, 2],
            id_to_token={0: "<pad>", 1: "A", 2: "B"},
            blank_id=0,
            duration_sec=2.0,
        )


def test_align_words_is_not_naive_fallback_anymore():
    with pytest.raises(NotImplementedError, match="scaffold-only"):
        align_words("hello world", 2.0)
