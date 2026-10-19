"""Prompt templates for the question generation (VQG) task."""

from __future__ import annotations

import json

from mind2web_vqg.paths import PROMPT_DIR

#: Default template — EDIT to match your actual task.
DEFAULT_TEMPLATE = """\
You are given a web page and a user task. Generate a clarifying question.

Task: {task}
Page (cleaned HTML):
{html}

Question:"""


def load_prompt(name: str = "question_gen.json") -> dict:
    """Read a prompt JSON file from ``src/mind2web_vqg/prompts``."""
    with open(PROMPT_DIR / name, encoding="utf-8") as f:
        return json.load(f)


__all__ = ["DEFAULT_TEMPLATE", "load_prompt"]
