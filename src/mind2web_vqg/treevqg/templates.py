"""Reference questions for the MVP: deterministic, template-based, self-supervised.

The proposal synthesises questions with GPT-4o / Qwen2-VL and filters them with
Sentence-BERT. That is the right final recipe, but it is slow, costs money and makes the
first end-to-end run depend on a model that can fail. The MVP therefore starts from
**rules**:

* all three levels use only information already in the row (tree, task, trajectory);
* every question has a ground-truth answer, so the dataset can be scored immediately;
* the generator is a pure function — swapping it for the LMM + contrastive filter later
  changes one module, not the schema or the trainer.

The wording is deliberately close to the paper's examples, and the ``hint`` field records
which signal the question targets (text / role / region / relationship / progress).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mind2web_vqg.treevqg import LEVEL_NAMES

#: A question plus the on-screen (or trajectory) answer it must resolve.
Question = dict[str, str]


@dataclass
class LevelContext:
    """Everything outside the tree that a template may use."""

    task: str = ""
    action_history: list[str] = field(default_factory=list)
    target_action: str = ""
    op: str = ""
    value: str = ""
    image_size: list[int] | None = None


#: Instruction shown to a generator, one per structural level.
LEVEL_INSTRUCTIONS: dict[str, str] = {
    "L1": (
        "You are shown a screenshot of a web page and a text outline of its UI tree. "
        "Ask exactly one question about the individual element marked with * — the element "
        "the user needs for the task. The answer must be readable from the screenshot. "
        "Output only the question."
    ),
    "L2": (
        "You are shown a screenshot of a web page and a text outline of its UI tree. "
        "Ask exactly one question about the container marked with * and how the elements "
        "inside it relate to each other for the task. The answer must be readable from the "
        "screenshot. Output only the question."
    ),
    "L3": (
        "You are shown a screenshot of a web page, a text outline of its UI tree, and the "
        "steps already completed. Ask exactly one question about the overall workflow state "
        "marked with * (progress or safety). The answer must follow from the screen or the "
        "completed steps. Output only the question."
    ),
}


def region_label(bbox: list[float] | None, image_size: list[int] | None) -> str | None:
    """Coarse 3x3 page region of a ``[x, y, w, h]`` box, e.g. ``"top-right"``."""
    if not bbox or len(bbox) != 4 or not image_size or len(image_size) != 2:
        return None
    width, height = image_size
    if width <= 0 or height <= 0:
        return None
    x, y, w, h = (float(value) for value in bbox)
    cx, cy = (x + w / 2) / width, (y + h / 2) / height
    if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
        return None
    rows = ("top", "middle", "bottom")
    cols = ("left", "center", "right")
    row = rows[min(int(cy * 3), 2)]
    col = cols[min(int(cx * 3), 2)]
    return f"{row}-{col}"


def _first_labelled(tree: Any, node_id: int) -> int | None:
    for candidate in tree.interactive(node_id):
        if candidate != node_id and tree.label(candidate):
            return candidate
    return None


def _first_heading(tree: Any) -> str:
    for node in tree.nodes:
        if node.tag in {"h1", "h2", "h3"} and node.label:
            return node.label
    return ""


def question_for_level(
    tree: Any,
    anchor: int,
    level: str,
    context: LevelContext,
    *,
    element_anchor: int | None = None,
) -> Question | None:
    """Build the reference question/answer for one anchor, or ``None`` when unanswerable."""
    node = tree.node(anchor)
    role = node.role or node.tag
    where = f'the task "{context.task}"' if context.task else "this step"

    if level == "L1":
        label = tree.label(anchor)
        if label:
            return {
                "question": f"What text is shown on the {role} the user needs for {where}?",
                "answer": label,
                "answer_type": "text",
                "answer_source": "screenshot",
                "hint": "text",
            }
        if node.clickable:
            return {
                "question": f"What kind of UI element should the user interact with for {where}?",
                "answer": role,
                "answer_type": "text",
                "answer_source": "screenshot",
                "hint": "role",
            }
        region = region_label(node.bbox, context.image_size)
        if region:
            return {
                "question": f"Where on the page is the element needed for {where}?",
                "answer": region,
                "answer_type": "region",
                "answer_source": "screenshot",
                "hint": "region",
            }
        return None

    if level == "L2":
        container_label = tree.label(anchor)
        container_role = node.role or node.tag
        child = (
            element_anchor
            if element_anchor not in (None, anchor)
            else _first_labelled(tree, anchor)
        )
        child_label = tree.label(child) if child is not None else ""
        child_role = tree.node(child).role if child is not None else "element"

        if container_label and child_label:
            return {
                "question": (
                    f'In the "{container_label}" {container_role}, which {child_role} '
                    f"should the user use for {where}?"
                ),
                "answer": child_label,
                "answer_type": "text",
                "answer_source": "screenshot",
                "hint": "relationship",
            }
        if container_label:
            return {
                "question": (
                    f'Which kind of control does the "{container_label}" {container_role} '
                    f"contain for {where}?"
                ),
                "answer": child_role,
                "answer_type": "text",
                "answer_source": "screenshot",
                "hint": "relationship",
            }
        if child_label:
            return {
                "question": (
                    f'Which {container_role} on the page contains the "{child_label}" '
                    f"{child_role} needed for {where}?"
                ),
                "answer": container_role,
                "answer_type": "text",
                "answer_source": "screenshot",
                "hint": "relationship",
            }
        return None

    if level == "L3":
        steps = [str(step) for step in context.action_history if str(step).strip()]
        if steps:
            return {
                "question": (
                    f"How many steps of {where} have already been completed on this screen?"
                ),
                "answer": str(len(steps)),
                "answer_type": "count",
                "answer_source": "trajectory",
                "hint": "progress",
            }
        heading = _first_heading(tree)
        if heading:
            return {
                "question": (
                    f'Which page or section is the user currently working on for {where}?'
                ),
                "answer": heading,
                "answer_type": "text",
                "answer_source": "screenshot",
                "hint": "state",
            }
        return None

    raise ValueError(f"unknown level: {level!r}; choose from {list(LEVEL_NAMES)}")


def build_input(
    tree: Any,
    anchor: int,
    level: str,
    context: LevelContext,
    *,
    max_lines: int = 60,
    max_label: int = 60,
) -> str:
    """The text condition: level instruction + task + completed steps + tree outline.

    The outline is why the MVP works with today's API/VLM backends: it gives a text-only
    model the same parent/child evidence the structural encoder will eventually learn.
    """
    if level not in LEVEL_INSTRUCTIONS:
        raise ValueError(f"unknown level: {level!r}; choose from {list(LEVEL_NAMES)}")
    lines = [LEVEL_INSTRUCTIONS[level], ""]
    lines.append(f"User's task: {context.task or '(not given)'}")
    steps = [str(step) for step in context.action_history if str(step).strip()]
    if steps:
        lines.append("Steps already completed:")
        lines.extend(f"  {index}. {step}" for index, step in enumerate(steps, start=1))
    else:
        lines.append("Steps already completed: none")
    lines.append("")
    lines.append(f"UI tree ({LEVEL_NAMES.get(level, level)} anchor marked with *):")
    lines.append(tree.outline(anchor=anchor, max_lines=max_lines, max_label=max_label))
    return "\n".join(lines).strip()
