"""The VQG task definition: screenshot -> "ask a question whose answer is on the screen".

The task is **self-supervised Visual Question Generation**: given a web-page screenshot
(and the user's task), the model asks a question about the page whose answer can be read
off the screenshot. The answer is already visible, so the reference question is derived
by rule from the action-level metadata — no human or LLM labelling.

The implementation lives in :mod:`mind2web_vqg.data.vqg`; this module is the documented
entry point used by ``evaluate.py`` / ``train.py`` and simply re-exports it.

Input: the ``examples.jsonl`` written by ``m2w-mm-export`` (one action == one screenshot).
Build it with::

    uv run m2w-mm-export --split test_task --limit 5
    uv run m2w-build-vqg  --split test_task --limit 5

Category reference (see ``data/vqg.py`` for the full docstring):

    target_label    what text is shown on the element needed for the task
    element_type    what kind of UI element it is (button, link, text field, ...)
    target_region   where on the page it is located
    field_value     the value to type/select  (opt-in; answered from the task, not the image)
"""

from __future__ import annotations

from mind2web_vqg.data.vqg import build_examples, load_exported

__all__ = ["build_examples", "load_exported"]
