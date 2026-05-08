"""Audio chunking utilities."""

import torch


def chunk_audio(
    waveform: torch.Tensor,
    sample_rate: int,
    chunk_seconds: float,
    overlap_seconds: float = 0.0,
) -> list[tuple[torch.Tensor, float, float]]:
    """Split waveform into overlapping chunks.

    Args:
        waveform: [channels, time]
        sample_rate: sample rate in Hz
        chunk_seconds: duration of each chunk in seconds
        overlap_seconds: overlap between consecutive chunks in seconds

    Returns:
        List of (chunk_tensor, start_seconds, end_seconds).
        chunk_tensor shape is [channels, chunk_samples].
    """
    if waveform.dim() != 2:
        raise ValueError(f"Expected waveform of shape [channels, time], got {waveform.shape}")

    chunk_samples = int(chunk_seconds * sample_rate)
    overlap_samples = int(overlap_seconds * sample_rate)
    hop_samples = chunk_samples - overlap_samples

    if hop_samples <= 0:
        raise ValueError("chunk_seconds must be greater than overlap_seconds")

    chunks: list[tuple[torch.Tensor, float, float]] = []
    total_samples = waveform.shape[-1]
    start = 0
    while start < total_samples:
        end = min(start + chunk_samples, total_samples)
        chunk = waveform[:, start:end]
        # Pad if necessary
        if chunk.shape[-1] < chunk_samples:
            pad = chunk_samples - chunk.shape[-1]
            chunk = torch.nn.functional.pad(chunk, (0, pad))
        start_sec = start / sample_rate
        end_sec = end / sample_rate
        chunks.append((chunk, start_sec, end_sec))
        start += hop_samples

    return chunks
