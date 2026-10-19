"""Baseline that calls an LLM API (OpenAI or a compatible endpoint).

Supports images through the OpenAI vision format, so the same class works for
``gpt-4o-mini`` and for a local vLLM/SGLang server that exposes
``http://localhost:8000/v1``.
"""

from __future__ import annotations

import base64
import mimetypes
import os
from pathlib import Path

from mind2web_vqg.models.base import QuestionGenerator


def _data_url(path: str | Path) -> str:
    """Encode a local image as a ``data:image/...;base64,...`` URL."""
    path = Path(path)
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


class APIModel(QuestionGenerator):
    """Thin wrapper around the OpenAI Chat Completions API.

    Requires ``OPENAI_API_KEY`` (see ``.env.example``). Install: ``uv sync --extra api``.

    For a local server, leave ``OPENAI_API_KEY`` empty and set
    ``OPENAI_BASE_URL=http://localhost:8000/v1``.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        system_prompt: str | None = None,
        extra_body: dict | None = None,
    ) -> None:
        from openai import OpenAI  # lazy import so the `api` extra is optional

        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY") or "EMPTY",
            base_url=os.environ.get("OPENAI_BASE_URL") or None,
        )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.extra_body = extra_body

    def generate(self, prompt: str, image: str | Path | None = None) -> str:
        messages: list[dict] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        if image is None:
            content: str | list[dict] = prompt
        else:
            content = [
                {"type": "image_url", "image_url": {"url": _data_url(image)}},
                {"type": "text", "text": prompt},
            ]
        messages.append({"role": "user", "content": content})

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            extra_body=self.extra_body,
        )
        return (resp.choices[0].message.content or "").strip()
