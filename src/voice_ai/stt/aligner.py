"""CTC forced-alignment utilities.

The functions in this module align a known transcript to frame-level CTC
emissions. They are intended for data preparation pipelines that need real
word timestamps from model probabilities, not heuristic duration splitting.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import torch


class AlignmentError(RuntimeError):
    """Raised when CTC forced alignment cannot produce a valid path."""


@dataclass(frozen=True)
class AlignmentPoint:
    """One frame on the backtracked CTC path."""

    token_index: int
    time_index: int
    score: float


@dataclass(frozen=True)
class TokenSpan:
    """Aligned span for one transcript token position."""

    token_index: int
    token_id: int
    token: str
    start_frame: int
    end_frame: int
    start_sec: float
    end_sec: float
    score: float

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec

    def to_dict(self) -> dict:
        data = asdict(self)
        data["duration_sec"] = self.duration_sec
        return data


@dataclass(frozen=True)
class WordSpan:
    """Aligned word span derived from token spans."""

    word: str
    start_frame: int
    end_frame: int
    start_sec: float
    end_sec: float
    duration_sec: float
    score: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class AlignmentResult:
    """Successful CTC forced-alignment result."""

    token_spans: list[TokenSpan]
    word_spans: list[WordSpan]
    average_word_confidence: float
    num_frames: int
    duration_sec: float
    seconds_per_frame: float
    blank_id: int

    def to_dict(self) -> dict:
        return {
            "token_spans": [span.to_dict() for span in self.token_spans],
            "word_spans": [span.to_dict() for span in self.word_spans],
            "average_word_confidence": self.average_word_confidence,
            "num_frames": self.num_frames,
            "duration_sec": self.duration_sec,
            "seconds_per_frame": self.seconds_per_frame,
            "blank_id": self.blank_id,
        }


def align_words(text: str, duration: float) -> list[dict]:
    """Deprecated scaffold helper.

    This name used to evenly split audio duration across words. That behavior
    is intentionally removed because it is not forced alignment. Use
    :func:`forced_align_ctc` with real CTC emissions instead.
    """
    raise NotImplementedError(
        "align_words was a scaffold-only duration splitter. "
        "Use forced_align_ctc(emission, tokens, id_to_token, blank_id, duration_sec)."
    )


def get_trellis(
    emission: torch.Tensor,
    tokens: Sequence[int],
    blank_id: int,
) -> torch.Tensor:
    """Build the CTC trellis for transcript tokens.

    Args:
        emission: Log probabilities with shape ``[num_frames, vocab_size]``.
        tokens: Transcript token ids, without CTC blanks.
        blank_id: Token id used as the CTC blank.

    Returns:
        Trellis tensor with shape ``[num_frames + 1, len(tokens) + 1]``.
    """
    _validate_emission_and_tokens(emission, tokens, blank_id)
    num_frames = emission.size(0)
    num_tokens = len(tokens)

    trellis = emission.new_empty((num_frames + 1, num_tokens + 1))
    trellis[0, 0] = 0.0
    trellis[0, 1:] = -float("inf")
    trellis[1:, 0] = torch.cumsum(emission[:, blank_id], dim=0)

    token_tensor = torch.tensor(tokens, dtype=torch.long, device=emission.device)
    for t in range(num_frames):
        stay = trellis[t, 1:] + emission[t, blank_id]
        change = trellis[t, :-1] + emission[t, token_tensor]
        trellis[t + 1, 1:] = torch.maximum(stay, change)

    return trellis


def backtrack(
    trellis: torch.Tensor,
    emission: torch.Tensor,
    tokens: Sequence[int],
    blank_id: int,
) -> list[AlignmentPoint]:
    """Backtrack the most likely CTC alignment path from a trellis."""
    if len(tokens) == 0:
        return []
    if trellis.dim() != 2:
        raise ValueError(f"Expected trellis to be 2D, got {tuple(trellis.shape)}")
    if trellis.size(1) != len(tokens) + 1:
        raise ValueError("Trellis token dimension does not match tokens.")

    j = len(tokens)
    best_final = trellis[:, j]
    if not torch.isfinite(best_final).any():
        raise AlignmentError("No finite CTC path reached the final transcript token.")

    t_start = int(torch.argmax(best_final).item())
    if t_start == 0:
        raise AlignmentError("Best CTC path ended before any emission frame was consumed.")

    path: list[AlignmentPoint] = []
    for t in range(t_start, 0, -1):
        if j == 0:
            break

        token_index = j - 1
        stayed = trellis[t - 1, j] + emission[t - 1, blank_id]
        changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[token_index]]

        if changed > stayed:
            score = emission[t - 1, tokens[token_index]].exp().item()
            path.append(
                AlignmentPoint(
                    token_index=token_index,
                    time_index=t - 1,
                    score=score,
                )
            )
            j -= 1
        else:
            score = emission[t - 1, blank_id].exp().item()
            path.append(
                AlignmentPoint(
                    token_index=token_index,
                    time_index=t - 1,
                    score=score,
                )
            )

    if j != 0:
        raise AlignmentError("Backtracking failed to reach the first transcript token.")

    return list(reversed(path))


def merge_repeats(
    path: Sequence[AlignmentPoint],
    tokens: Sequence[int],
    id_to_token: Mapping[int, str],
    seconds_per_frame: float,
) -> list[TokenSpan]:
    """Merge a backtracked CTC path into one span per transcript token position."""
    if not path:
        return []

    spans: list[TokenSpan] = []
    group: list[AlignmentPoint] = [path[0]]

    def flush(points: list[AlignmentPoint]) -> None:
        token_index = points[0].token_index
        token_id = int(tokens[token_index])
        token = id_to_token.get(token_id)
        if token is None:
            raise AlignmentError(f"Token id {token_id} is missing from id_to_token.")
        start_frame = int(points[0].time_index)
        end_frame = int(points[-1].time_index) + 1
        score = float(sum(point.score for point in points) / len(points))
        spans.append(
            TokenSpan(
                token_index=int(token_index),
                token_id=token_id,
                token=str(token),
                start_frame=start_frame,
                end_frame=end_frame,
                start_sec=start_frame * seconds_per_frame,
                end_sec=end_frame * seconds_per_frame,
                score=score,
            )
        )

    for point in path[1:]:
        if point.token_index == group[-1].token_index:
            group.append(point)
        else:
            flush(group)
            group = [point]
    flush(group)
    return spans


def merge_words(
    token_spans: Sequence[TokenSpan],
    word_delimiter: str = "|",
) -> list[WordSpan]:
    """Merge token spans into word spans using the CTC word delimiter token."""
    words: list[WordSpan] = []
    current_tokens: list[TokenSpan] = []

    def flush() -> None:
        if not current_tokens:
            return
        word = "".join(span.token for span in current_tokens)
        if not word:
            return
        start = current_tokens[0]
        end = current_tokens[-1]
        score = float(sum(span.score for span in current_tokens) / len(current_tokens))
        words.append(
            WordSpan(
                word=word,
                start_frame=start.start_frame,
                end_frame=end.end_frame,
                start_sec=start.start_sec,
                end_sec=end.end_sec,
                duration_sec=end.end_sec - start.start_sec,
                score=score,
            )
        )

    for span in token_spans:
        if span.token == word_delimiter:
            flush()
            current_tokens = []
        else:
            current_tokens.append(span)
    flush()
    return words


def forced_align_ctc(
    emission: torch.Tensor,
    tokens: Sequence[int],
    id_to_token: Mapping[int, str],
    blank_id: int,
    duration_sec: float,
    word_delimiter: str = "|",
) -> AlignmentResult:
    """Align transcript tokens to CTC emissions and return word timestamps.

    Args:
        emission: Log probabilities with shape ``[num_frames, vocab_size]``.
        tokens: Transcript token ids, without CTC blanks.
        id_to_token: Mapping from token id to tokenizer string.
        blank_id: Token id used as CTC blank.
        duration_sec: Duration of the source audio in seconds.
        word_delimiter: Token string separating words. Wav2Vec2 CTC uses ``|``.

    Returns:
        AlignmentResult with token and word spans.
    """
    if duration_sec <= 0:
        raise ValueError("duration_sec must be positive.")

    _validate_emission_and_tokens(emission, tokens, blank_id)
    if len(tokens) >= emission.size(0):
        raise AlignmentError("Transcript has too many tokens for the emission frames.")

    emission = emission.detach().float().cpu()
    seconds_per_frame = duration_sec / float(emission.size(0))
    trellis = get_trellis(emission, tokens, blank_id)
    path = backtrack(trellis, emission, tokens, blank_id)
    token_spans = merge_repeats(path, tokens, id_to_token, seconds_per_frame)
    word_spans = merge_words(token_spans, word_delimiter=word_delimiter)
    if not word_spans:
        raise AlignmentError("Alignment produced no word spans.")

    average_word_confidence = float(
        sum(span.score for span in word_spans) / len(word_spans)
    )
    return AlignmentResult(
        token_spans=token_spans,
        word_spans=word_spans,
        average_word_confidence=average_word_confidence,
        num_frames=int(emission.size(0)),
        duration_sec=float(duration_sec),
        seconds_per_frame=float(seconds_per_frame),
        blank_id=int(blank_id),
    )


def _validate_emission_and_tokens(
    emission: torch.Tensor,
    tokens: Sequence[int],
    blank_id: int,
) -> None:
    if emission.dim() != 2:
        raise ValueError(f"Expected emission shape [time, vocab], got {tuple(emission.shape)}")
    if emission.size(0) == 0 or emission.size(1) == 0:
        raise ValueError("Emission must have non-empty time and vocab dimensions.")
    if not tokens:
        raise AlignmentError("Cannot align an empty token sequence.")
    if blank_id < 0 or blank_id >= emission.size(1):
        raise ValueError(f"blank_id {blank_id} is outside vocab size {emission.size(1)}.")
    for token in tokens:
        if token < 0 or token >= emission.size(1):
            raise ValueError(f"Token id {token} is outside vocab size {emission.size(1)}.")
