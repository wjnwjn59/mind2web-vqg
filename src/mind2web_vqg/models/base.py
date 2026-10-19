"""Shared interface for every question generator model."""

from __future__ import annotations

from abc import ABC, abstractmethod


class QuestionGenerator(ABC):
    """Every model (API or local) implements this interface.

    This is why ``evaluate.py`` and ``train.py`` don't need to know what the model is.
    """

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a response (question) from the prompt."""
        raise NotImplementedError

    def generate_batch(self, prompts: list[str]) -> list[str]:
        """Defaults to sequential calls; subclasses can override for batching/async."""
        return [self.generate(p) for p in prompts]
