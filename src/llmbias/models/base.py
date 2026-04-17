from __future__ import annotations

from abc import ABC, abstractmethod

from llmbias.schemas import ModelResponse


class BlackBoxLLM(ABC):
    """Abstract adapter for closed or open LLM APIs."""

    @abstractmethod
    def generate(self, prompt: str) -> ModelResponse:
        raise NotImplementedError

