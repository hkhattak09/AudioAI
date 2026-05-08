"""LLM base interface."""

from abc import ABC, abstractmethod

from voice_ai.pipeline.schemas import LLMResult


class LLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> LLMResult:
        """Generate text from a prompt."""
        ...
