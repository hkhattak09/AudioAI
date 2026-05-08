"""Audio codec placeholder.

This module provides an interface for future EnCodec integration.
References:
- https://github.com/facebookresearch/encodec
- https://github.com/priyammaz/EncodecTrainer
"""

from abc import ABC, abstractmethod

import torch


class AudioCodec(ABC):
    """Abstract interface for neural audio codecs."""

    @abstractmethod
    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        """Encode waveform to discrete tokens.

        Args:
            waveform: [B, channels, T]

        Returns:
            tokens: [B, N, T']
        """
        ...

    @abstractmethod
    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """Decode tokens back to waveform.

        Args:
            tokens: [B, N, T']

        Returns:
            waveform: [B, channels, T]
        """
        ...


class FakeCodec(AudioCodec):
    """Fake codec for testing."""

    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        # Just return random tokens shaped roughly like a downsampled version
        B, C, T = waveform.shape
        T_prime = max(1, T // 320)
        return torch.randint(0, 1024, (B, 1, T_prime), device=waveform.device)

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        B, N, T_prime = tokens.shape
        T = T_prime * 320
        return torch.zeros(B, 1, T, device=tokens.device)
