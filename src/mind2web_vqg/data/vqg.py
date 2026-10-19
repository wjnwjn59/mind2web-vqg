"""Self-supervised Visual Question Generation (VQG) for GUI-agent screenshots.

The task
--------
The model is shown a web-page screenshot and must **ask a question about it**. In a GUI
agent setting the most useful questions are the ones a user could actually answer by
looking at the screen — *what is this element called, what kind of element is it, where
is it*. Those questions are also *self-supervised*: the answer is already on the screen,
so we get a ground-truth label for free from the action-level metadata that ships with
Multimodal-Mind2Web. No human labels, no LLM judging.

For every exported example (one action == one screenshot) we derive, by rule, the
reference question(s) and their answers:

    category        question is about ...          answer comes from
    --------------  -----------------------------  --------------------------------
    target_label    the element's visible text      ``target_text`` or  (the screen)
                                                    ``target_action``
    element_type    what kind of UI element it is   tag / input type  (the screen)
    target_region   where on the page it sits       bounding box      (the screen)
    field_value     the value to type or select     task + operation  (the task)

``target_label``, ``element_type`` and ``target_region`` are *screenshot-grounded* and are
the default. ``field_value`` is the only category whose answer lives in the task text
rather than the image, so it is opt-in (``--categories``).

Example schema (one JSON object per line)::

    {"id": "<source id>::<category>",
     "input": <prompt asking the model to generate a question>,
     "target": <reference question, the generation target>,
     "answer": <ground-truth answer>,
     "image": <path to the screenshot>,
     "meta": {"category": ..., "answer_source": "screenshot"|"task",
              "answer_type": "text"|"region", "annotation_id": ..., "split": ...,
              "task": ..., "website": ..., "domain": ..., "target_action": ...}}

Usage::

    uv run m2w-mm-export --split test_task --limit 5      # screenshots + metadata
    uv run m2w-build-vqg  --split test_task --limit 5     # -> data/processed/vqg/test_task

    uv run m2w-build-vqg --split test_task --categories target_label target_region
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from mind2web_vqg.cli import ensure_utf8
from mind2web_vqg.data.multimodal import SPLITS, export_samples
from mind2web_vqg.paths import PROCESSED_DIR

#: Categories whose answer can be read off the screenshot (the default task).
SCREENSHOT_GROUNDED: tuple[str, ...] = ("target_label", "element_type", "target_region")

#: All categories, including the one answered from the task text.
ALL_CATEGORIES: tuple[str, ...] = (*SCREENSHOT_GROUNDED, "field_value")

#: HTML tag -> the word a person would use for that element.
ROLE_BY_TAG: dict[str, str] = {
    "a": "link",
    "button": "button",
    "textarea": "text area",
    "select": "dropdown",
    "option": "option",
    "img": "image",
    "input": "input field",
    "label": "label",
    "form": "form",
    "nav": "navigation menu",
    "table": "table",
    "th": "table header",
    "td": "table cell",
    "tr": "table row",
    "li": "list item",
    "ul": "list",
    "ol": "list",
    "h1": "heading",
    "h2": "heading",
    "h3": "heading",
    "h4": "heading",
    "h5": "heading",
    "h6": "heading",
    "header": "header",
    "footer": "footer",
    "main": "main content area",
}

#: ``<input type=...>`` -> the word a person would use for that field.
ROLE_BY_INPUT_TYPE: dict[str, str] = {
    "text": "text field",
    "search": "search field",
    "email": "email field",
    "password": "password field",
    "number": "number field",
    "tel": "phone field",
    "url": "URL field",
    "date": "date field",
    "range": "slider",
    "checkbox": "checkbox",
    "radio": "radio button",
    "file": "file upload",
    "submit": "button",
    "button": "button",
    "reset": "button",
}

#: What each category asks about, used to steer the generator.
CATEGORY_HINTS: dict[str, str] = {
    "target_label": "the text or label shown on the element the task needs",
    "element_type": "what kind of UI element it is (button, link, text field, ...)",
    "target_region": "where on the page that element is located",
    "field_value": "the value that should be typed or selected",
}

#: Instruction prepended to the user's task to form the generation prompt.
INSTRUCTION = (
    "You are shown a screenshot of a web page. Ask exactly one question about the page "
    "whose answer can be read from the screenshot, focusing on {hint}. "
    "Output only the question."
)


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested, no I/O)
# --------------------------------------------------------------------------- #
def element_role(example: dict[str, Any]) -> str | None:
    """Human word for the target element's kind, or ``None`` if we can't tell.

    ``<input>`` is refined by its ``type`` attribute (a ``submit`` input is a button,
    a ``checkbox`` is a checkbox, ...). Unknown tags return ``None`` so the caller can
    skip questions whose answer would not be recognizable on the screen.
    """
    tag = str(example.get("target_tag") or "").strip().lower()
    if not tag:
        return None
    if tag == "input":
        input_type = str(example.get("target_type") or "text").strip().lower()
        return ROLE_BY_INPUT_TYPE.get(input_type, "input field")
    return ROLE_BY_TAG.get(tag)


def relative_region(
    bbox: list[float] | None,
    *,
    source_size: list[int] | None = None,
    image_size: list[int] | None = None,
    thirds: int = 3,
) -> str | None:
    """Coarse page region of a bounding box, e.g. ``"top-right"`` (or ``None``).

    ``bbox`` is ``[x, y, w, h]`` in the raw screenshot's pixel space; ``source_size`` is
    that raw size and ``image_size`` its (possibly downscaled) version. We rescale, then
    bucket the box's centre into a ``thirds x thirds`` grid. Boxes whose centre falls
    outside the page are dropped (a mismatch between DOM and screenshot coordinates).
    """
    if not bbox or len(bbox) != 4:
        return None
    size = image_size or source_size
    if not size or size[0] <= 0 or size[1] <= 0:
        return None
    scale = 1.0
    if source_size and image_size and source_size[0]:
        scale = image_size[0] / source_size[0]

    x, y, w, h = (float(v) for v in bbox)
    cx = (x + w / 2) * scale / size[0]
    cy = (y + h / 2) * scale / size[1]
    if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
        return None

    rows = ("top", "middle", "bottom")
    cols = ("left", "center", "right")
    row = rows[min(int(cy * thirds), thirds - 1)]
    col = cols[min(int(cx * thirds), thirds - 1)]
    return f"{row}-{col}"


def _task(example: dict[str, Any]) -> str:
    return str(example.get("task") or "").strip()


#: Leading ``[tag]`` of a ``target_action_reprs`` string, e.g. ``"[link]  "``.
_TAG_PREFIX = re.compile(r"^\s*\[[^\]]*\]\s*")


def action_repr_label(target_action: Any) -> str:
    """Text of the target element in ``target_action_reprs``.

    The dataset renders the element as ``"[link]  Budget Truck -> CLICK"``, so the label is
    the part between the ``[tag]`` prefix and the ``->`` operation. Used when the element's
    HTML attributes carry no ``text`` — which is the common case, because text content is
    not an attribute. Returns ``""`` when the string is missing or not in that shape.
    """
    text = str(target_action or "").strip()
    if not _TAG_PREFIX.match(text):
        return ""
    return _TAG_PREFIX.sub("", text).rsplit("->", 1)[0].strip()


def _label(example: dict[str, Any]) -> str:
    """The element's visible text: the stored attribute text, else the action repr."""
    label = str(example.get("target_text") or "").strip()
    return label or action_repr_label(example.get("target_action"))


def build_input(example: dict[str, Any], category: str, *, with_task: bool = True) -> str:
    """The generation prompt: instruction (+ user task) for one category."""
    lines = [INSTRUCTION.format(hint=CATEGORY_HINTS[category])]
    task = _task(example)
    if with_task and task:
        lines.append(f"The user's task is: {task}")
    return "\n".join(lines)


def generate_questions(
    example: dict[str, Any], categories: list[str] | tuple[str, ...] | None = None
) -> list[dict[str, str]]:
    """Reference questions + answers derivable from one exported example.

    Each item is ``{category, question, answer, answer_source, answer_type}``. A category
    is silently skipped when its answer is missing (no label, unmapped tag, no bbox, ...),
    which keeps every emitted question answerable.
    """
    cats = tuple(categories) if categories else SCREENSHOT_GROUNDED
    unknown = sorted(set(cats) - set(ALL_CATEGORIES))
    if unknown:
        raise ValueError(f"unknown categor(y|ies): {unknown}. Choose from {list(ALL_CATEGORIES)}")

    role = element_role(example)
    label = _label(example)
    task = _task(example)
    value = str(example.get("value") or "").strip()
    region = relative_region(
        example.get("target_bbox"),
        source_size=example.get("source_size"),
        image_size=example.get("image_size"),
    )

    # ``where`` names the step the element is needed for; ``noun`` what to call the element.
    where = f'the task "{task}"' if task else "this step"
    noun = role or "element"

    out: list[dict[str, str]] = []
    for category in cats:
        if category == "target_label" and label:
            # Only this category truly needs the text: the text *is* the answer.
            out.append(
                {
                    "category": category,
                    "question": f"What text is shown on the {noun} the user needs for {where}?",
                    "answer": label,
                    "answer_source": "screenshot",
                    "answer_type": "text",
                }
            )
        elif category == "element_type" and role:
            # The answer is the element kind, so a missing label only changes the wording.
            question = (
                f'What kind of UI element is "{label}"?'
                if label
                else f"What kind of UI element does the user need for {where}?"
            )
            out.append(
                {
                    "category": category,
                    "question": question,
                    "answer": role,
                    "answer_source": "screenshot",
                    "answer_type": "text",
                }
            )
        elif category == "target_region" and region:
            # Same: the answer is the location, not the text.
            question = (
                f'In which part of the page is "{label}" located?'
                if label
                else f"In which part of the page is it needed for {where}?"
            )
            out.append(
                {
                    "category": category,
                    "question": question,
                    "answer": region,
                    "answer_source": "screenshot",
                    "answer_type": "region",
                }
            )
        elif category == "field_value" and value:
            field = f' the "{label}" field' if label else " the field"
            tail = f' to complete the task "{task}"' if task else ""
            out.append(
                {
                    "category": category,
                    "question": f"What value should be entered in{field}{tail}?",
                    "answer": value,
                    "answer_source": "task",
                    "answer_type": "text",
                }
            )
    return out


def build_examples(
    examples: list[dict[str, Any]],
    *,
    categories: list[str] | tuple[str, ...] | None = None,
    with_task: bool = True,
) -> list[dict[str, Any]]:
    """Expand exported screenshot examples into ``{id, input, target, answer, ...}``.

    One source example yields up to one VQG example per applicable category, all sharing
    the same screenshot.
    """
    out: list[dict[str, Any]] = []
    for example in examples:
        for qa in generate_questions(example, categories):
            out.append(
                {
                    "id": f"{example.get('id')}::{qa['category']}",
                    "input": build_input(example, qa["category"], with_task=with_task),
                    "target": qa["question"],
                    "answer": qa["answer"],
                    "image": example.get("image"),
                    "meta": {
                        "category": qa["category"],
                        "answer_source": qa["answer_source"],
                        "answer_type": qa["answer_type"],
                        "annotation_id": example.get("annotation_id"),
                        "split": example.get("split"),
                        "task": example.get("task"),
                        "website": example.get("website"),
                        "domain": example.get("domain"),
                        "target_action": example.get("target_action"),
                        "source_id": example.get("id"),
                    },
                }
            )
    return out


def category_stats(examples: list[dict[str, Any]]) -> dict[str, int]:
    """Number of generated examples per category (for the console summary)."""
    counts = Counter(ex["meta"]["category"] for ex in examples)
    return {category: counts.get(category, 0) for category in ALL_CATEGORIES}


# --------------------------------------------------------------------------- #
# I/O: read exported screenshots, write the VQG set
# --------------------------------------------------------------------------- #
def load_exported(
    split: str,
    *,
    path: str | Path | None = None,
    limit: int | None = None,
    export_if_missing: bool = True,
) -> list[dict[str, Any]]:
    """Read ``examples.jsonl`` written by ``m2w-mm-export`` (exporting it if absent)."""
    jsonl = Path(path) if path else PROCESSED_DIR / "multimodal" / split / "examples.jsonl"
    if not jsonl.exists():
        if not export_if_missing:
            raise FileNotFoundError(
                f"{jsonl} not found. Run `uv run m2w-mm-export --split {split}` first."
            )
        export_samples(split, limit=limit if limit is not None else 5)
    with open(jsonl, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return rows[:limit] if limit else rows


def write_examples(examples: list[dict[str, Any]], out_dir: str | Path) -> Path:
    """Write the VQG examples as ``<out_dir>/examples.jsonl`` and return the path."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    jsonl = out / "examples.jsonl"
    with open(jsonl, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    return jsonl


def main(argv: list[str] | None = None) -> int:
    ensure_utf8()
    parser = argparse.ArgumentParser(
        description="Build the self-supervised screenshot VQG set from exported examples."
    )
    parser.add_argument("--split", default="test_task", choices=SPLITS)
    parser.add_argument("--limit", type=int, default=None, help="max source examples")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        choices=list(ALL_CATEGORIES),
        help=f"subset of {list(ALL_CATEGORIES)} (default: {list(SCREENSHOT_GROUNDED)})",
    )
    parser.add_argument(
        "--examples", type=Path, default=None, help="input examples.jsonl (default: processed)"
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="output dir (default: processed/vqg/<split>)"
    )
    parser.add_argument(
        "--no-task", action="store_true", help="omit the user task from the generation prompt"
    )
    args = parser.parse_args(argv)

    exported = load_exported(args.split, path=args.examples, limit=args.limit)
    examples = build_examples(
        exported, categories=args.categories, with_task=not args.no_task
    )

    out_dir = args.out or PROCESSED_DIR / "vqg" / args.split
    jsonl = write_examples(examples, out_dir)

    stats = category_stats(examples)
    print(f"[vqg] {len(exported)} screenshot(s) -> {len(examples)} example(s)")
    for category, count in stats.items():
        print(f"  {category:14s} {count:5d}")
    print(f"[vqg] -> {jsonl}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
