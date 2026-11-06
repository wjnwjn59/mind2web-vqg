"""TreeVQG: tree-conditioned visual question generation across 3 UI granularities.

This package implements the MVP of the method described in ``docs/document.pdf``:

* :mod:`mind2web_vqg.treevqg.schema` — the structural node/tree used everywhere;
* :mod:`mind2web_vqg.treevqg.parse` — cleaned/raw HTML -> :class:`UiTree`;
* :mod:`mind2web_vqg.treevqg.levels` — Element / Container / Workflow anchors;
* :mod:`mind2web_vqg.treevqg.templates` — self-supervised reference questions;
* :mod:`mind2web_vqg.treevqg.build` — join screenshots + HTML -> TreeVQG JSONL;
* :mod:`mind2web_vqg.treevqg.evaluate` — run an API/local model on that JSONL.

The MVP deliberately stops before the paper's learned encoders: it produces a *runnable
dataset* and a *text outline of the tree* that the existing API/VLM backends can already
use. The learned Structural Transformer Encoder, dual contrastive loss and center loss
slot in behind the same schema — see ``docs/treevqg/plan.md``.

Usage::

    uv run m2w-build-treevqg --split test_task --limit 5
    uv run m2w-eval-treevqg --split test_task --model api/gpt-4o-mini --max-samples 6
"""

from __future__ import annotations

__all__ = ["LEVELS", "LEVEL_NAMES", "LEVEL_DESCRIPTIONS"]

#: Short level ids used in every row: ``L1`` (element), ``L2`` (container), ``L3`` (workflow).
LEVELS: tuple[str, ...] = ("L1", "L2", "L3")

#: Human name of each level.
LEVEL_NAMES: dict[str, str] = {
    "L1": "element",
    "L2": "container",
    "L3": "workflow",
}

#: One-line description of what a question at each level should resolve.
LEVEL_DESCRIPTIONS: dict[str, str] = {
    "L1": "an isolated interactive element (button, link, input, ...)",
    "L2": "a grouping container and the elements it relates (form, card, toolbar)",
    "L3": "the whole screen state plus the action history (progress, safety)",
}
