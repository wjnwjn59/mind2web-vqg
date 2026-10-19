"""Baseline that calls an LLM API (OpenAI or a compatible endpoint)."""

from __future__ import annotations

import os

from mind2web_vqg.models.base import QuestionGenerator


class APIModel(QuestionGenerator):
    """Thin wrapper around the OpenAI Chat Completions API.

    Requires ``OPENAI_API_KEY`` (see ``.env.example``). Install: ``uv sync --extra api``.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        system_prompt: str | None = None,
    ) -> None:
        from openai import OpenAI  # lazy import so the `api` extra is optional

        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("OPENAI_BASE_URL") or None,
        )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt

    def generate(self, prompt: str) -> str:
        messages: list[dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()
