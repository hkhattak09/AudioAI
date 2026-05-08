"""Colab AI adapter."""

from voice_ai.llm.base import LLMClient
from voice_ai.pipeline.schemas import LLMResult


class ColabAIAdapter(LLMClient):
    """Adapter for google.colab.ai.generate_text.

    Do not instantiate this outside Google Colab.
    """

    def generate(self, prompt: str, **kwargs) -> LLMResult:
        try:
            from google.colab import ai
        except ImportError as exc:
            raise RuntimeError("Colab AI is only available inside Google Colab.") from exc

        response = ai.generate_text(prompt, **kwargs)
        return LLMResult(
            text=str(response),
            finish_reason="stop",
        )
