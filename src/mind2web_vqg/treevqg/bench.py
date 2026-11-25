"""TreeVQG-Bench v1 — six question types built on the UI tree.

The first TreeVQG pass asked one question per *level* (element / container / workflow).
This module keeps the same tree and the same exporter, but replaces the level
taxonomy with a **type** taxonomy that can be scored and compared:

``target_label``
    What text is on the control the task needs? (control: the old L1 question —
    the answer is often copyable from the task text, which is exactly what the
    benchmark should expose.)
``target_region``
    Where on the page is that control? (3x3 coarse region.)
``container_membership``
    Which section/card/form on the page contains that control? The answer is a
    *structural* referent: a flat widget caption cannot produce it.
``sibling_disambiguation``
    The page has N controls with the same label; which one does the task mean?
    The answer names the container (or region) that distinguishes them.
``absence``
    Does section ``X`` contain a control the user could use? Balanced yes/no —
    the negative rows come from a sibling section that *lacks* the control.
``consequence``
    If the user clicks the control, what happens? The gold label is derived from
    the **next state of the recorded trajectory** (``actions[t+1]``), so it is
    observed rather than guessed by an LLM.

Every row carries two different views:

``condition``
    What a *question generator* sees: task + tree outline with the anchor marked ``*``.
``input``
    What an *answerer* sees: task + the question, **no tree and no image path**.
    ``--image`` is supplied separately by the evaluator, so a text-only baseline
    and an image baseline differ only by the screenshot.

Leakage is reported for the answerer's view (``answer_in_input``); the matching
``answer_in_condition`` is expected to be high — the generator legitimately sees
the evidence it is asking about.

Usage::

    uv run m2w-build-bench --split test_task --per-type 20
    uv run m2w-visualize-bench --examples data/processed/bench/v1/test_task/examples.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mind2web_vqg.cli import ensure_utf8
from mind2web_vqg.data.multimodal import SPLITS, normalize_limit
from mind2web_vqg.data.vqg import load_exported
from mind2web_vqg.paths import PROCESSED_DIR
from mind2web_vqg.treevqg import audit
from mind2web_vqg.treevqg import io as tree_io
from mind2web_vqg.treevqg.build import HTML_FIELDS, join_split
from mind2web_vqg.treevqg.clarify import (
    Container,
    dedupe_nested,
    is_weak_container,
    is_weak_label,
    nearest_container,
    norm,
    plural,
    same_label_controls,
    shorten,
)
from mind2web_vqg.treevqg.levels import target_node
from mind2web_vqg.treevqg.parse import (
    apply_candidate_metadata,
    parse_html,
    target_from_candidates,
    tree_stats,
)
from mind2web_vqg.treevqg.roles import is_container
from mind2web_vqg.treevqg.schema import UiTree
from mind2web_vqg.treevqg.templates import LevelContext, region_label

#: Type order is also the order used in reports.
TYPES: tuple[str, ...] = (
    "container_membership",
    "sibling_disambiguation",
    "absence",
    "consequence",
    "target_label",
    "target_region",
)

#: Types whose answer is a structural referent, not a string on the screen.
REQUIRES_TREE = frozenset(
    {"container_membership", "sibling_disambiguation", "absence", "consequence"}
)

TYPE_DESCRIPTIONS: dict[str, str] = {
    "container_membership": "Which section/card/form contains the needed control?",
    "sibling_disambiguation": "N controls share a label; which one does the task mean?",
    "absence": "Does section X contain a usable control? (balanced yes/no)",
    "consequence": "What happens after the click? (gold from the next trajectory state)",
    "target_label": "What text is on the needed control? (legacy L1 control)",
    "target_region": "Where on the page is the needed control? (legacy L1 control)",
}

_GENERATOR_INSTRUCTION = (
    "You are building ONE question about the web page shown by the UI tree below. "
    "The node marked with * is the element the question must be about. "
    "The answer must be something a person can say after looking at the screenshot. "
    "Output only the question."
)


# --------------------------------------------------------------------------- #
# Question value object
# --------------------------------------------------------------------------- #
@dataclass
class BenchQuestion:
    """One generated question plus the nodes a visualiser should draw."""

    kind: str
    question: str
    answer: str
    answer_type: str = "text"
    answer_source: str = "screenshot"
    primary: int | None = None
    container: int | None = None
    #: ``[(node_id, kind)]`` with kind in anchor / container / candidate / reference.
    referents: list[tuple[int, str]] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def with_primary(self, node: int, kind: str = "anchor") -> BenchQuestion:
        self.primary = node if self.primary is None else self.primary
        self.referents.append((node, kind))
        return self


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _role(tree: UiTree, node_id: int) -> str:
    node = tree.node(node_id)
    return node.role or node.tag or "element"


def _label(tree: UiTree, node_id: int) -> str:
    return tree.label(node_id, max_len=80).strip()


def _task_phrase(context: LevelContext, limit: int = 120) -> str:
    task = shorten(context.task or "this step", limit)
    return f'"{task}"'


def _controls_inside(tree: UiTree, container: int) -> list[int]:
    """Labelled interactive nodes inside a container, nested duplicates removed."""
    found = [
        node_id
        for node_id in tree.interactive(container)
        if node_id != container and _label(tree, node_id)
    ]
    return dedupe_nested(tree, found)


def _descriptor(tree: UiTree, node_id: int, image_size: list[int] | None) -> tuple[str, str]:
    """``(text, mode)`` describing where a control is; mode is ``section`` or ``region``."""
    label = _label(tree, node_id)
    section = _section(tree, node_id, avoid_label=label)
    if section is not None:
        return section.label, "section"
    region = region_label(tree.node(node_id).bbox, image_size) or ""
    return region, "region" if region else ""


_HEADING_TAGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6", "legend", "caption"})

_PLACEHOLDER_PREFIXES = (
    "enter ",
    "type ",
    "search ",
    "select a ",
    "select an ",
    "choose ",
    "pick ",
    "mm/dd",
    "dd/mm",
    "yyyy",
)


def _usable_title(text: str) -> str:
    """Keep short, human-nameable section titles; drop placeholders and page metadata."""
    cleaned = " ".join(str(text or "").split())
    if len(cleaned) < 3 or len(cleaned) > 64:
        return ""
    if is_weak_container(cleaned) or _looks_like_metadata(cleaned):
        return ""
    if cleaned.casefold().startswith(_PLACEHOLDER_PREFIXES):
        return ""
    if re.match(r"^(object|node|div|span|item)[:_-]?\d+$", cleaned, re.IGNORECASE):
        return ""
    return cleaned


def _container_label(tree: UiTree, node_id: int) -> str:
    """A *nameable* section title: own label, ARIA label, or a heading inside.

    ``tree.label`` falls back to the first labelled descendant, which on real pages is
    often a price ("0.7 miles away") or an input placeholder rather than the card title.
    Using that as the gold answer would make ``container_membership`` unanswerable, so a
    container without a usable title is simply not used.
    """
    node = tree.node(node_id)
    for candidate in (node.label, node.attrs.get("aria-label"), node.attrs.get("title")):
        title = _usable_title(str(candidate or ""))
        if title:
            return title
    for descendant in tree.subtree(node_id):
        if descendant == node_id:
            continue
        child = tree.node(descendant)
        role = str(child.attrs.get("role") or "").strip().lower()
        if (child.tag.lower() in _HEADING_TAGS or role == "heading") and child.label.strip():
            title = _usable_title(child.label)
            if title:
                return title
    return ""


def _section(tree: UiTree, node_id: int, *, avoid_label: str = "") -> Container | None:
    """Nearest ancestor that is a container *and* has a usable title.

    Pass 1 accepts only a real title (own/ARIA name or a heading inside). Pass 2 falls
    back to the first labelled container, but rejects labels that are page metadata
    (a price, a distance, a rating) because those cannot name a section.
    """
    avoided = norm(avoid_label)
    for ancestor in reversed(tree.path(node_id)[:-1]):
        node = tree.nodes[ancestor]
        if not is_container(node.tag, node.attrs):
            continue
        label = _container_label(tree, ancestor)
        if not label or norm(label) == avoided:
            continue
        return Container(node=ancestor, label=label, role=node.role or node.tag)

    fallback = nearest_container(tree, node_id, avoid_label=avoid_label)
    if fallback is not None:
        title = _usable_title(fallback.label)
        if title:
            return Container(node=fallback.node, label=title, role=fallback.role)
    return None


_METADATA_PREFIX = re.compile(r"^\s*[\$\u20ac\u00a3]?\s*\d")
_METADATA_HINT = (" miles", " mi away", " km", " per night", " /night", "stars", "star rating")


def _looks_like_metadata(label: str) -> bool:
    key = norm(label)
    if _METADATA_PREFIX.match(key):
        return True
    return any(hint in f" {key}" for hint in _METADATA_HINT)


def _section_title(tree: UiTree, node_id: int) -> str:
    """Best effort *name* for a section node (heading first, then a clean label)."""
    title = _container_label(tree, node_id)
    if title:
        return title
    return _usable_title(_label(tree, node_id))


def _node_bbox(tree: UiTree, node_id: int) -> tuple[list[float] | None, bool]:
    """Own bbox, else the union of descendant bboxes. Second value = inferred."""
    own = tree.node(node_id).bbox
    if own:
        return list(own), False
    boxes = [
        node.bbox
        for node in (tree.node(descendant) for descendant in tree.subtree(node_id))
        if node.bbox
    ]
    if not boxes:
        return None, False
    left = min(box[0] for box in boxes)
    top = min(box[1] for box in boxes)
    right = max(box[0] + box[2] for box in boxes)
    bottom = max(box[1] + box[3] for box in boxes)
    return [left, top, right - left, bottom - top], True


# --------------------------------------------------------------------------- #
# Generators — all share the same signature so they can be tabulated
# --------------------------------------------------------------------------- #
def q_target_label(
    tree: UiTree,
    target: int,
    context: LevelContext,
    *,
    image_size: list[int] | None = None,
    current_html: str = "",
    next_html: str = "",
    op: str = "",
) -> BenchQuestion | None:
    label = _label(tree, target)
    if is_weak_label(label):
        return None
    role = _role(tree, target)
    return BenchQuestion(
        kind="target_label",
        question=(
            f"What text is shown on the {role} that the user needs for "
            f"{_task_phrase(context)}?"
        ),
        answer=label,
        answer_type="text",
        answer_source="screenshot",
        primary=target,
        referents=[(target, "anchor")],
    )


def q_target_region(
    tree: UiTree,
    target: int,
    context: LevelContext,
    *,
    image_size: list[int] | None = None,
    current_html: str = "",
    next_html: str = "",
    op: str = "",
) -> BenchQuestion | None:
    region = region_label(tree.node(target).bbox, image_size)
    if not region:
        return None
    role = _role(tree, target)
    return BenchQuestion(
        kind="target_region",
        question=f"Where on the page is the {role} the user needs for {_task_phrase(context)}?",
        answer=region,
        answer_type="region",
        answer_source="screenshot",
        primary=target,
        referents=[(target, "anchor")],
    )


def q_container_membership(
    tree: UiTree,
    target: int,
    context: LevelContext,
    *,
    image_size: list[int] | None = None,
    current_html: str = "",
    next_html: str = "",
    op: str = "",
) -> BenchQuestion | None:
    label = _label(tree, target)
    if is_weak_label(label):
        return None
    container = _section(tree, target, avoid_label=label)
    if container is None or not _usable_title(container.label):
        return None
    role = _role(tree, target)
    return BenchQuestion(
        kind="container_membership",
        question=(
            f'Which section of the page contains the "{label}" {role} '
            f"that the user needs for {_task_phrase(context)}?"
        ),
        answer=container.label,
        answer_type="container",
        answer_source="screenshot",
        primary=target,
        container=container.node,
        referents=[(target, "anchor"), (container.node, "container")],
        extra={"container_role": container.role},
    )


def q_sibling_disambiguation(
    tree: UiTree,
    target: int,
    context: LevelContext,
    *,
    image_size: list[int] | None = None,
    current_html: str = "",
    next_html: str = "",
    op: str = "",
) -> BenchQuestion | None:
    label = _label(tree, target)
    if is_weak_label(label):
        return None
    same = [
        node_id
        for node_id in same_label_controls(tree, label)
        if _node_bbox(tree, node_id)[0]
    ]
    if not (2 <= len(same) <= 12) or target not in same:
        return None

    described: dict[int, tuple[str, str]] = {}
    for node_id in same:
        described[node_id] = _descriptor(tree, node_id, image_size)
    gold_text, gold_mode = described.get(target, ("", ""))
    if not gold_text:
        return None
    # If two candidates cannot be told apart by container/region, the question has no
    # unique answer — drop it instead of emitting an ambiguous gold.
    if any(
        norm(other_text) == norm(gold_text) and other_mode == gold_mode
        for node_id, (other_text, other_mode) in described.items()
        if node_id != target
    ):
        return None

    role = plural(_role(tree, target))
    options = sorted({text for text, _ in described.values() if text})
    gold_container = _section(tree, target, avoid_label=label)
    phrase = (
        f'the one in "{gold_text}"'
        if gold_mode == "section"
        else f"the one in the {gold_text} area"
    )
    return BenchQuestion(
        kind="sibling_disambiguation",
        question=(
            f'There are {len(same)} {role} labelled "{label}" on this page. '
            f"Which one should the user choose for {_task_phrase(context)}?"
        ),
        answer=phrase,
        answer_type="referent",
        answer_source="screenshot",
        primary=target,
        container=gold_container.node if gold_container else None,
        referents=[(node_id, "candidate") for node_id in same] + [(target, "anchor")],
        extra={"n_same": len(same), "options": options, "descriptor_mode": gold_mode},
    )


#: Section labels that make a "does this section contain X?" question meaningful.
def q_absence(
    tree: UiTree,
    target: int,
    context: LevelContext,
    *,
    image_size: list[int] | None = None,
    current_html: str = "",
    next_html: str = "",
    op: str = "",
) -> BenchQuestion | None:
    label = _label(tree, target)
    if is_weak_label(label):
        return None
    gold_container = _section(tree, target, avoid_label=label)
    if gold_container is None or not _usable_title(gold_container.label):
        return None
    parent = tree.node(gold_container.node).parent
    if parent is None:
        return None

    negatives: list[int] = []
    for sibling in tree.node(parent).children:
        if sibling == gold_container.node:
            continue
        node = tree.node(sibling)
        if not is_container(node.tag, node.attrs):
            continue
        if not _section_title(tree, sibling) or not _controls_inside(tree, sibling):
            continue
        inside = {norm(_label(tree, node_id)) for node_id in tree.subtree(sibling)}
        if norm(label) not in inside:
            negatives.append(sibling)
    if not negatives:
        return None

    # Deterministic 50/50 polarity, so ``absence`` is not always answered "No".
    polarity_yes = (target + gold_container.node + len(negatives)) % 2 == 0
    if polarity_yes:
        section = gold_container.node
        answer = "Yes"
    else:
        section = negatives[target % len(negatives)]
        answer = "No"

    section_label = _section_title(tree, section)
    if not _usable_title(section_label):
        return None
    role = _role(tree, target)
    return BenchQuestion(
        kind="absence",
        question=(
            f'Does the "{section_label}" section contain a "{label}" {role} '
            f"that the user could use for {_task_phrase(context)}?"
        ),
        answer=answer,
        answer_type="boolean",
        answer_source="screenshot",
        primary=target,
        container=section,
        referents=[(target, "reference"), (section, "container")],
        extra={"polarity": "yes" if polarity_yes else "no", "section": section_label},
    )


_CONFIRM_WORDS = (
    "thank you for your order",
    "thank you for your purchase",
    "order confirmed",
    "order confirmation",
    "payment successful",
    "purchase complete",
    "booking confirmed",
    "reservation confirmed",
    "confirmation number",
    "receipt",
)

_DELETE_WORDS = ("delete", "remove", "trash", "clear", "discard", "unsubscribe")

_SUBMIT_WORDS = ("submit", "save", "send", "confirm", "apply", "done")
_NAVIGATE_WORDS = ("continue", "next", "proceed", "go", "start")


def _next_state(instance: dict[str, Any], action: dict[str, Any], field: str) -> str:
    """HTML of the page *after* ``action`` (the next recorded step), or ``""``."""
    actions = instance.get("actions") or []
    uid = action.get("action_uid")
    index = next(
        (i for i, candidate in enumerate(actions) if candidate.get("action_uid") == uid),
        None,
    )
    if index is None or index + 1 >= len(actions):
        return ""
    return str(actions[index + 1].get(field) or "")


def q_consequence(
    tree: UiTree,
    target: int,
    context: LevelContext,
    *,
    image_size: list[int] | None = None,
    current_html: str = "",
    next_html: str = "",
    op: str = "",
) -> BenchQuestion | None:
    """Gold consequence read off the *next* state of the recorded trajectory."""
    label = _label(tree, target)
    if is_weak_label(label) or not next_html:
        return None
    lowered = next_html.casefold()
    current = current_html.casefold()
    key = norm(label)
    operation = (op or context.op or "").upper()
    clicked = operation in {"", "CLICK"}
    if not clicked:
        # A click question about a text field would be misleading; ``TYPE``/``SELECT``
        # consequences need their own wording and are left for a later pass.
        return None

    consequence = ""
    evidence = ""
    confirmation = next(
        (
            word
            for word in _CONFIRM_WORDS
            # The phrase must be *new*: a "receipt" link in the footer is not a purchase.
            if word in lowered and word not in current
        ),
        "",
    )
    if confirmation and clicked:
        consequence, evidence = "transaction", f'"{confirmation}" appears on the next page'
    elif clicked and any(word in key for word in _DELETE_WORDS) and key not in lowered:
        consequence, evidence = "deletion", "label disappeared from the next page"
    elif clicked and key in _SUBMIT_WORDS:
        consequence, evidence = "submission", "submit-like button, no confirmation text"
    elif clicked and key in _NAVIGATE_WORDS:
        consequence, evidence = "navigation", "generic button, no confirmation text"

    if not consequence:
        return None

    phrases = {
        "transaction": "It completes a purchase or booking and shows a confirmation.",
        "deletion": "It permanently removes the item from the page.",
        "submission": "It submits the form with the current values.",
        "navigation": "It only moves to another page; no data is changed.",
    }
    role = _role(tree, target)
    return BenchQuestion(
        kind="consequence",
        question=f'If the user clicks the "{label}" {role}, what will happen?',
        answer=phrases[consequence],
        answer_type="consequence",
        answer_source="trajectory:t+1",
        primary=target,
        referents=[(target, "anchor")],
        extra={
            "consequence_class": consequence,
            "evidence": evidence,
            "op": operation,
        },
    )


GENERATORS = {
    "container_membership": q_container_membership,
    "sibling_disambiguation": q_sibling_disambiguation,
    "absence": q_absence,
    "consequence": q_consequence,
    "target_label": q_target_label,
    "target_region": q_target_region,
}


# --------------------------------------------------------------------------- #
# Row assembly
# --------------------------------------------------------------------------- #
def _referents(
    tree: UiTree, question: BenchQuestion, image_size: list[int] | None
) -> list[dict[str, Any]]:
    seen: set[tuple[int, str]] = set()
    out: list[dict[str, Any]] = []
    for node_id, kind in question.referents:
        if (node_id, kind) in seen or node_id is None or node_id >= len(tree):
            continue
        seen.add((node_id, kind))
        node = tree.node(node_id)
        bbox, inferred = _node_bbox(tree, node_id)
        out.append(
            {
                "node": node_id,
                "kind": kind,
                "tag": node.tag,
                "role": node.role or node.tag,
                "label": _label(tree, node_id),
                "bbox": bbox,
                "bbox_inferred": inferred,
                "depth": node.depth,
            }
        )
    return out


def _condition(
    tree: UiTree, question: BenchQuestion, context: LevelContext, *, max_lines: int
) -> str:
    lines = [
        _GENERATOR_INSTRUCTION,
        "",
        f"User's task: {context.task or '(not given)'}",
    ]
    if question.kind == "consequence":
        lines.append(f"Planned action: {context.target_action or '(not given)'}")
    lines.append("")
    lines.append("UI tree (the node marked with * is what the question is about):")
    lines.append(tree.outline(anchor=question.primary, max_lines=max_lines, max_label=80))
    return "\n".join(lines).strip()


def _make_row(
    *,
    source: dict[str, Any],
    instance: dict[str, Any],
    split: str,
    question: BenchQuestion,
    tree: UiTree,
    context: LevelContext,
    image_size: list[int] | None,
    stats: dict[str, int],
    max_lines: int,
) -> dict[str, Any]:
    source_id = str(source.get("id") or source.get("image_name") or "row")
    input_text = f"Task: {context.task or '(not given)'}\n\nQuestion: {question.question}"
    row: dict[str, Any] = {
        "id": f"{source_id}::{question.kind}",
        "type": question.kind,
        "input": input_text,
        "condition": _condition(tree, question, context, max_lines=max_lines),
        "question": question.question,
        "target": question.question,
        "answer": question.answer,
        "image": source.get("image"),
        "source_size": list(source.get("source_size") or image_size or []) or None,
        "image_size": list(source.get("image_size") or image_size or []) or None,
        "referents": _referents(tree, question, image_size),
        "meta": {
            "category": question.kind,
            "type": question.kind,
            "description": TYPE_DESCRIPTIONS.get(question.kind, ""),
            "requires_tree": question.kind in REQUIRES_TREE,
            "answer_type": question.answer_type,
            "answer_source": question.answer_source,
            "anchor_id": question.primary,
            "anchor_tag": (
                tree.node(question.primary).tag if question.primary is not None else None
            ),
            "anchor_role": (
                _role(tree, question.primary) if question.primary is not None else ""
            ),
            "anchor_label": (
                _label(tree, question.primary) if question.primary is not None else ""
            ),
            "anchor_depth": (
                tree.node(question.primary).depth if question.primary is not None else None
            ),
            "container_id": question.container,
            "n_candidates": question.extra.get("n_same", 0),
            "split": source.get("split") or split,
            "task": context.task,
            "website": source.get("website") or instance.get("website"),
            "domain": source.get("domain") or instance.get("domain"),
            "subdomain": source.get("subdomain") or instance.get("subdomain"),
            "target_action": context.target_action,
            "op": context.op,
            "annotation_id": source.get("annotation_id") or instance.get("annotation_id"),
            "action_uid": source.get("action_uid"),
            "tree_nodes": stats["nodes"],
            "tree_max_depth": stats["max_depth"],
            **question.extra,
        },
    }
    row["meta"]["answer_in_input"] = audit.answer_in_input(row)
    row["meta"]["answer_in_question"] = audit.answer_in_question(row)
    answer_key = audit.normalize(row["answer"])
    condition_key = f" {audit.normalize(row['condition'])} "
    row["meta"]["answer_in_condition"] = bool(
        len(answer_key) >= 2 and f" {answer_key} " in condition_key
    )
    return row


# --------------------------------------------------------------------------- #
# Stratified sampling
# --------------------------------------------------------------------------- #
def stratified_sample(
    items: list[dict[str, Any]],
    *,
    per_type: int,
    max_per_website: int,
    seed: int = 13,
) -> list[dict[str, Any]]:
    """Round-robin over websites, then relax the per-website cap if still short."""
    rng = random.Random(seed)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        buckets[str(item["meta"].get("website") or "unknown")].append(item)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    order = sorted(buckets, key=lambda name: (-len(buckets[name]), name))
    chosen: list[dict[str, Any]] = []
    used: Counter[str] = Counter()
    keep_going = True
    while len(chosen) < per_type and keep_going:
        keep_going = False
        for website in order:
            if len(chosen) >= per_type:
                break
            if used[website] >= max_per_website or not buckets[website]:
                continue
            chosen.append(buckets[website].pop())
            used[website] += 1
            keep_going = True

    while len(chosen) < per_type:
        available = [name for name in order if buckets[name]]
        if not available:
            break
        for website in available:
            if len(chosen) >= per_type:
                break
            chosen.append(buckets[website].pop())
    return chosen


# --------------------------------------------------------------------------- #
# Split builder
# --------------------------------------------------------------------------- #
def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 3) if values else 0.0


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_type[row["type"]].append(row)

    summary: dict[str, Any] = {}
    for kind in TYPES:
        group = by_type.get(kind, [])
        summary[kind] = {
            "n": len(group),
            "websites": dict(
                Counter(str(row["meta"].get("website")) for row in group).most_common()
            ),
            "answer_sources": dict(
                Counter(str(row["meta"].get("answer_source")) for row in group).most_common()
            ),
            "leak": {
                "answer_in_input": _mean(
                    [1.0 if row["meta"].get("answer_in_input") else 0.0 for row in group]
                ),
                "answer_in_question": _mean(
                    [1.0 if row["meta"].get("answer_in_question") else 0.0 for row in group]
                ),
                "answer_in_condition": _mean(
                    [1.0 if row["meta"].get("answer_in_condition") else 0.0 for row in group]
                ),
            },
            "requires_tree": kind in REQUIRES_TREE,
            "description": TYPE_DESCRIPTIONS.get(kind, ""),
        }
        if kind == "absence":
            summary[kind]["polarity"] = dict(
                Counter(str(row["meta"].get("polarity")) for row in group).most_common()
            )
        if kind == "consequence":
            summary[kind]["classes"] = dict(
                Counter(str(row["meta"].get("consequence_class")) for row in group).most_common()
            )
        if kind == "sibling_disambiguation":
            counts = [int(row["meta"].get("n_same") or 0) for row in group]
            summary[kind]["mean_same_label"] = _mean([float(value) for value in counts])
    return summary


def build_bench(
    split: str = "test_task",
    *,
    examples_path: str | Path | None = None,
    limit: int | None = None,
    types: Iterable[str] = TYPES,
    per_type: int = 20,
    max_per_website: int = 2,
    out_dir: str | Path | None = None,
    html_field: str = "cleaned",
    max_lines: int = 60,
    include_missing_images: bool = False,
    seed: int = 13,
) -> dict[str, Any]:
    requested = tuple(types)
    unknown = sorted(set(requested) - set(TYPES))
    if unknown:
        raise ValueError(f"unknown type(s): {unknown}. Choose from {list(TYPES)}")

    limit = normalize_limit(limit)
    exported = load_exported(split, path=examples_path, limit=limit, export_if_missing=False)
    if not exported:
        raise FileNotFoundError(
            f"no exported examples for {split!r}. Run "
            f"`uv run m2w-mm-export --split {split} --limit {limit or 0}` first."
        )

    field = HTML_FIELDS.get(html_field, html_field)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    skipped: Counter[str] = Counter()

    for source, instance, action in join_split(split, exported, html_field=html_field):
        image = source.get("image")
        if not include_missing_images and image and not Path(str(image)).is_file():
            skipped["missing_image"] += 1
            continue
        tree = parse_html(str(action.get(field) or ""))
        if len(tree) <= 1:
            skipped["no_html"] += 1
            continue

        apply_candidate_metadata(tree, action)
        target = target_from_candidates(tree, action)
        if target is None:
            target = target_node(
                tree,
                tag=source.get("target_tag"),
                text=source.get("target_text"),
                action_repr=source.get("target_action") or instance.get("target_action_reprs"),
            )
        if target is None:
            skipped["no_target"] += 1
            continue

        image_size = list(source.get("source_size") or source.get("image_size") or []) or None
        context = LevelContext(
            task=str(source.get("task") or instance.get("confirmed_task") or ""),
            action_history=[str(step) for step in (source.get("action_history") or [])],
            target_action=str(source.get("target_action") or ""),
            op=str(source.get("op") or ""),
            value=str(source.get("value") or ""),
            image_size=image_size,
        )
        stats = tree_stats(tree)
        current_html = str(action.get(field) or "")
        next_html = _next_state(instance, action, field)

        for kind in requested:
            question = GENERATORS[kind](
                tree,
                target,
                context,
                image_size=image_size,
                current_html=current_html,
                next_html=next_html,
                op=str(source.get("op") or ""),
            )
            if question is None:
                skipped[f"no_{kind}"] += 1
                continue
            buckets[kind].append(
                _make_row(
                    source=source,
                    instance=instance,
                    split=split,
                    question=question,
                    tree=tree,
                    context=context,
                    image_size=image_size,
                    stats=stats,
                    max_lines=max_lines,
                )
            )

    rows: list[dict[str, Any]] = []
    per_type_stats: dict[str, Any] = {}
    for kind in requested:
        candidates = buckets.get(kind, [])
        chosen = stratified_sample(
            candidates, per_type=per_type, max_per_website=max_per_website, seed=seed
        )
        per_type_stats[kind] = {
            "available": len(candidates),
            "selected": len(chosen),
            "websites_available": len(
                {str(item["meta"].get("website")) for item in candidates}
            ),
        }
        rows.extend(chosen)

    destination = Path(out_dir) if out_dir else PROCESSED_DIR / "bench" / "v1" / split
    examples = tree_io.write_jsonl(rows, destination / "examples.jsonl")
    summary = summarize(rows)
    stats = {
        "split": split,
        "html_field": HTML_FIELDS.get(html_field, html_field),
        "types": list(requested),
        "requested_per_type": per_type,
        "max_per_website": max_per_website,
        "built": len(rows),
        "by_type": summary,
        "selection": per_type_stats,
        "skipped": {name: count for name, count in skipped.most_common() if count},
        "output": str(examples),
    }
    tree_io.write_json(stats, destination / "stats.json")
    return {"examples": examples, "stats": stats, "stats_path": destination / "stats.json"}


def print_summary(result: dict[str, Any]) -> None:
    stats = result["stats"]
    print(f"[bench] {stats['built']} row(s) -> {stats['output']}")
    print(f"{'type':26s} {'n':>4s} {'websites':>9s} {'leak(input)':>12s} {'tree':>5s}")
    for kind, info in stats["by_type"].items():
        print(
            f"{kind:26s} {info['n']:4d} {len(info['websites']):9d} "
            f"{info['leak']['answer_in_input']:12.2f} "
            f"{'yes' if info['requires_tree'] else 'no':>5s}"
        )
    if stats["skipped"]:
        print("[bench] skipped:", json.dumps(stats["skipped"], ensure_ascii=False))


def main(argv: list[str] | None = None) -> int:
    ensure_utf8()
    parser = argparse.ArgumentParser(
        description="Build TreeVQG-Bench v1 (six question types, stratified sample)."
    )
    parser.add_argument("--split", default="test_task", choices=SPLITS)
    parser.add_argument("--examples", default=None, help="exported multimodal examples.jsonl")
    parser.add_argument("--limit", type=int, default=None, help="0 = whole split")
    parser.add_argument("--types", nargs="+", default=list(TYPES), choices=list(TYPES))
    parser.add_argument("--per-type", type=int, default=20)
    parser.add_argument("--max-per-website", type=int, default=2)
    parser.add_argument("--html", default="cleaned", choices=sorted(HTML_FIELDS))
    parser.add_argument("--max-lines", type=int, default=60)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--include-missing-images", action="store_true")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args(argv)

    result = build_bench(
        args.split,
        examples_path=args.examples,
        limit=args.limit,
        types=args.types,
        per_type=args.per_type,
        max_per_website=args.max_per_website,
        out_dir=args.out_dir,
        html_field=args.html,
        max_lines=args.max_lines,
        include_missing_images=args.include_missing_images,
        seed=args.seed,
    )
    print_summary(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
