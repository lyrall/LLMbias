from __future__ import annotations

from abc import ABC, abstractmethod

from llmbias.schemas import ModelResponse


class BlackBoxLLM(ABC):
    """Abstract adapter for closed or open LLM APIs."""

    _CONTINUATION_SYSTEM_PROMPT = (
        "You are completing a sentence or short passage in English. "
        "Continue the given text naturally and concisely. "
        "Keep the response to 1-2 sentences, avoid long explanations, and do not use bullet points."
    )

    @abstractmethod
    def generate(self, prompt: str) -> ModelResponse:
        raise NotImplementedError

    def build_messages(self, prompt: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self._CONTINUATION_SYSTEM_PROMPT},
            {"role": "user", "content": f"Continue the following text:\n\n{prompt}"},
        ]
