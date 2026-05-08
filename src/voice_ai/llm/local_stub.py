"""Local deterministic LLM stub."""

from voice_ai.llm.base import LLMClient
from voice_ai.pipeline.schemas import LLMResult


class LocalStubLLM(LLMClient):
    """Deterministic local stub for LLM."""

    def generate(self, prompt: str, **kwargs) -> LLMResult:
        return LLMResult(
            text="Hello. This is a local stub response.",
            finish_reason="stop",
        )
