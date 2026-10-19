"""Shared interface for every question generator model."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class QuestionGenerator(ABC):
    """Every model (API or local) implements this interface.

    This is why ``evaluate.py`` and ``train.py`` don't need to know what the model is.
    """

    @abstractmethod
    def generate(self, prompt: str, image: str | Path | None = None) -> str:
        """Generate a response (question) from the prompt.

        Args:
            prompt: text instruction.
            image: optional path to a screenshot. Text-only backends may ignore it
                or raise; vision backends must use it.
        """
        raise NotImplementedError

    def generate_batch(
        self, prompts: list[str], images: list[str | Path | None] | None = None
    ) -> list[str]:
        """Defaults to sequential calls; subclasses can override for batching/async."""
        if images is None:
            images = [None] * len(prompts)
        return [self.generate(p, image=i) for p, i in zip(prompts, images, strict=True)]
