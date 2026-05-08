"""Simple energy-based VAD fallback."""

import torch


def detect_speech_segments(
    waveform: torch.Tensor,
    sample_rate: int,
    frame_ms: int = 30,
    threshold: float | None = None,
) -> list[tuple[float, float]]:
    """Detect speech segments using frame energy.

    Args:
        waveform: [channels, time] or [time].
        sample_rate: sample rate in Hz.
        frame_ms: frame length in milliseconds.
        threshold: energy threshold. If None, use median of frame energies.

    Returns:
        List of (start_seconds, end_seconds) for detected speech segments.
    """
    if waveform.dim() == 2:
        waveform = waveform.mean(dim=0)

    frame_len = int(sample_rate * frame_ms / 1000)
    if frame_len == 0:
        frame_len = 1

    num_frames = waveform.shape[0] // frame_len
    if num_frames == 0:
        return []

    frames = waveform[: num_frames * frame_len].view(num_frames, frame_len)
    energies = frames.pow(2).mean(dim=1)

    if threshold is None:
        threshold = energies.median().item()

    speech = energies > threshold

    segments: list[tuple[float, float]] = []
    in_segment = False
    start_frame = 0
    for i in range(num_frames):
        if speech[i] and not in_segment:
            in_segment = True
            start_frame = i
        elif not speech[i] and in_segment:
            in_segment = False
            segments.append(
                (start_frame * frame_ms / 1000.0, i * frame_ms / 1000.0)
            )

    if in_segment:
        segments.append(
            (start_frame * frame_ms / 1000.0, num_frames * frame_ms / 1000.0)
        )

    return segments
