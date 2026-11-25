"""H2 — tree-conditioned **clarification** questions (TreeVQG-Clarify).

The first TreeVQG pass asked about screen content ("what text is on this button?").
A GUI agent already has the screenshot *and* the DOM, so that answer is not new
information. This module builds the opposite kind of example:

* the agent sees the page and the **completed** steps;
* it must decide whether the user's intent is ambiguous **on this screen**;
* if it is, it asks a question whose answer only the user knows;
* the UI tree supplies the *referents* that make the question unambiguous
  (the two "Add" buttons are ``in "Recommended"`` vs ``in "Recently viewed"``).

Three positive tracks, matching the paper's three granularities:

``L1`` element
    Same label, different role (button vs link) or different page region.
``L2`` container
    Same label, different labelled container — the strongest signal, and the one a
    screenshot alone often cannot resolve.
``L3`` workflow
    A planned irreversible action needs confirmation.

Everything else is a **negative**: the gold output is ``NO_QUESTION``. Negative
examples are what teach *selective* asking (cf. Ask-F1 in HiL-Bench) and stop an
agent from interrogating the user at every step.

Leakage rule for this track: the prompt never lists the candidates and never marks
the gold control. The model has to find the ambiguity in the tree itself. Candidate
sets and ``answer_index`` are metadata for scoring only.

Usage::

    uv run m2w-build-clarify --split test_task --limit 0
    # -> data/processed/clarify/test_task/{examples.jsonl,stats.json}

    uv run m2w-build-clarify --split test_task --limit 0 --no-negatives --no-safety
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import unicodedata
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mind2web_vqg.cli import ensure_utf8
from mind2web_vqg.data.multimodal import SPLITS, normalize_limit
from mind2web_vqg.data.vqg import load_exported
from mind2web_vqg.paths import PROCESSED_DIR
from mind2web_vqg.treevqg import io as tree_io
from mind2web_vqg.treevqg.build import HTML_FIELDS, join_split
from mind2web_vqg.treevqg.parse import (
    apply_candidate_metadata,
    parse_html,
    target_from_candidates,
    tree_stats,
)
from mind2web_vqg.treevqg.roles import is_container
from mind2web_vqg.treevqg.schema import UiTree
from mind2web_vqg.treevqg.templates import LevelContext, region_label

#: Gold output for an unambiguous step. A model can emit it instead of a question.
NO_QUESTION = "NO_QUESTION"

#: Words that make an action worth confirming. Deliberately high recall: the safety
#: track is small, and a false positive costs one extra confirmation, not a wrong action.
_IRREVERSIBLE = re.compile(
    r"\b(delete|remove|cancel|discard|deactivate|unsubscribe|pay|purchase|buy|checkout|"
    r"place order|submit|send|confirm|deactivate|close account|book now)\b",
    re.IGNORECASE,
)

_L1_INSTRUCTION = (
    "You are a GUI assistant. Decide whether the user's intent is ambiguous at the "
    "ELEMENT level on this screen: several controls share the same name, or the same "
    "control type appears in different regions. If so, ask ONE question that lets the "
    f"user say which element they mean. If not, output exactly {NO_QUESTION}."
)
_L2_INSTRUCTION = (
    "You are a GUI assistant. Decide whether the user's intent is ambiguous at the "
    "CONTAINER level on this screen: the same control appears in several sections, "
    "cards, or forms. If so, ask ONE question that lets the user name the container "
    f"they mean. If not, output exactly {NO_QUESTION}."
)
_L3_INSTRUCTION = (
    "You are a GUI assistant. The planned action is listed below. If it is irreversible "
    "or its value is uncertain, ask ONE short confirmation question. If it does not need "
    f"confirmation, output exactly {NO_QUESTION}."
)
_NONE_INSTRUCTION = (
    "You are a GUI assistant. Decide whether the user's intent is ambiguous on this "
    "screen at the element, container, or workflow level. If a clarification is needed, "
    f"ask ONE question. If not, output exactly {NO_QUESTION}."
)
_LEVEL_INSTRUCTIONS = {
    "L1": _L1_INSTRUCTION,
    "L2": _L2_INSTRUCTION,
    "L3": _L3_INSTRUCTION,
    "none": _NONE_INSTRUCTION,
}


def clean_text(text: Any) -> str:
    """Drop control/private-use glyphs (web icon fonts leak them into labels)."""
    kept = "".join(char for char in str(text or "") if unicodedata.category(char)[0] != "C")
    return " ".join(kept.split())


def norm(text: Any) -> str:
    """Casefold and collapse whitespace (for label equality)."""
    return clean_text(text).casefold()


def shorten(text: str, limit: int = 60) -> str:
    text = clean_text(text)
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


#: Labels too weak to refer to a control unambiguously (``"2"``, ``"×"``, ``"i"``).
_WEAK_LABELS = frozenset({"", "i", "x", "×", "+", "-", "..."})


def is_weak_label(label: str) -> bool:
    key = norm(label)
    return len(key) < 2 or key in _WEAK_LABELS or key.isdigit()


_ROLE_GROUPS = {
    "dropdown": "choice",
    "list box": "choice",
    "combobox": "choice",
    "select": "choice",
    "option": "choice",
    "menu item": "choice",
    "menuitem": "choice",
    "text field": "text",
    "input field": "text",
    "search box": "text",
    "search field": "text",
    "textbox": "text",
    "searchbox": "text",
    "checkbox": "toggle",
    "check box": "toggle",
    "radio": "toggle",
    "radio button": "toggle",
    "switch": "toggle",
}

_ACTION_ROLES = frozenset(
    {"button", "link", "tab", "menu item", "menuitem", "option", "tree item", "treeitem"}
)
_INPUT_ROLES = frozenset(
    {
        "text field",
        "input field",
        "search box",
        "search field",
        "textbox",
        "searchbox",
        "dropdown",
        "list box",
        "combobox",
        "select",
        "checkbox",
        "check box",
        "radio",
        "radio button",
        "switch",
        "slider",
        "spin button",
        "spinbutton",
    }
)


def role_group(role: str) -> str:
    """Collapse rendering variants of the same control (dropdown vs list box)."""
    key = norm(role)
    return _ROLE_GROUPS.get(key, key)


def role_family(role: str) -> str:
    """Broad interaction family; a button and a text field are not alternatives."""
    key = norm(role)
    if key in _ACTION_ROLES:
        return "action"
    if key in _INPUT_ROLES:
        return "input"
    return "other"


def plural(word: str) -> str:
    word = word or "control"
    if word.endswith("s"):
        return word
    if word.endswith("y"):
        return word[:-1] + "ies"
    return word + "s"


_URL = re.compile(r"^(https?://|www\.)", re.IGNORECASE)


def is_weak_container(label: str) -> bool:
    """Reject referents a user cannot say out loud: URLs, paths, ``"1"``, punctuation.

    Four-digit numeric labels stay: they are often storage-unit or ZIP-code cards
    (``"90412"``), which are exactly the kind of section a user *can* name.
    """
    key = norm(label)
    if len(key) < 2 or key in _WEAK_LABELS:
        return True
    if _URL.match(key):
        return True
    if "/" in key and " " not in key:
        return True
    if key.isdigit() and len(key) < 4:
        return True
    return not any(char.isalnum() for char in key)


def dedupe_nested(tree: UiTree, node_ids: Iterable[int]) -> list[int]:
    """Drop nodes whose ancestor is also in the set (a card and its own button)."""
    ids = sorted(set(node_ids))
    present = set(ids)
    return [node_id for node_id in ids if not (set(tree.path(node_id)[:-1]) & present)]


@dataclass(frozen=True)
class Container:
    node: int
    label: str
    role: str


def nearest_container(tree: UiTree, node_id: int, *, avoid_label: str = "") -> Container | None:
    """Nearest ancestor that groups elements and has a *distinct* human label.

    ``tree.label`` falls back to a descendant label, so a wrapper whose only text is the
    button itself would otherwise be called "Add" — the degenerate container label that
    made the first L2 pass ask ``In the "Add" card, which Add button…?``. Skipping
    labels equal to the control label keeps the container informative.
    """
    avoided = norm(avoid_label)
    for ancestor in reversed(tree.path(node_id)[:-1]):
        node = tree.nodes[ancestor]
        if not is_container(node.tag, node.attrs):
            continue
        label = tree.label(ancestor)
        if not label or norm(label) == avoided or is_weak_container(label):
            continue
        return Container(node=ancestor, label=label, role=node.role or node.tag)
    return None


@dataclass
class Candidate:
    node: int
    label: str
    role: str
    container: str = ""
    container_node: int | None = None
    region: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "node": self.node,
            "label": self.label,
            "role": self.role,
            "container": self.container,
            "container_node": self.container_node,
            "region": self.region,
        }


def same_label_controls(tree: UiTree, label: str) -> list[int]:
    """Clickable nodes whose visible label equals ``label`` (nested duplicates removed)."""
    key = norm(label)
    if not key:
        return []
    found = [
        node.id
        for node in tree.nodes[1:]
        if node.clickable
        # A <label> is the name of a control, not an alternative control.
        and node.tag.lower() != "label"
        and norm(tree.label(node.id)) == key
    ]
    return dedupe_nested(tree, found)


def _descendants(tree: UiTree, node_id: int) -> list[int]:
    out: list[int] = []
    stack = list(tree.nodes[node_id].children)
    while stack:
        current = stack.pop()
        out.append(current)
        stack.extend(tree.nodes[current].children)
    return out


def resolve_target(tree: UiTree, target: int | None) -> tuple[int | None, str]:
    """Map an ambiguous hit to the clickable control a user can actually choose.

    Mind2Web sometimes records a click on a wrapper/text node whose nearest interactive
    ancestor (or only labelled descendant) is the real control. Resolving that node
    recovers examples that would otherwise be dropped as ``target_not_clickable``.
    """
    if target is None:
        return None, ""
    label = tree.label(target)
    if tree.nodes[target].clickable and label:
        return target, label

    for ancestor in reversed(tree.path(target)[:-1]):
        if tree.nodes[ancestor].clickable and tree.label(ancestor):
            return ancestor, tree.label(ancestor)

    matches = [
        node_id
        for node_id in _descendants(tree, target)
        if tree.nodes[node_id].clickable
        and tree.label(node_id)
        and (not label or norm(tree.label(node_id)) == norm(label))
    ]
    if len(matches) == 1:
        return matches[0], tree.label(matches[0])
    return target, label


def map_to_candidate(tree: UiTree, target: int, candidates: list[Candidate]) -> int | None:
    """After nested dedupe, move the target to the surviving ancestor/descendant."""
    ids = {candidate.node for candidate in candidates}
    if target in ids:
        return target
    target_path = set(tree.path(target))
    for candidate in candidates:
        if set(tree.path(candidate.node)) & target_path:
            return candidate.node
    return None


def candidate_set(
    tree: UiTree, node_ids: Iterable[int], image_size: list[int] | None
) -> list[Candidate]:
    out: list[Candidate] = []
    for node_id in node_ids:
        node = tree.nodes[node_id]
        label = tree.label(node_id)
        container = nearest_container(tree, node_id, avoid_label=label)
        out.append(
            Candidate(
                node=node_id,
                label=label,
                role=node.role or node.tag,
                container=container.label if container else "",
                container_node=container.node if container else None,
                region=region_label(node.bbox, image_size) or "",
            )
        )
    return out


def _join_options(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} or {items[1]}"
    return ", ".join(items[:-1]) + f", or {items[-1]}"


def _quoted(items: list[str], limit: int = 120) -> list[str]:
    # Do not truncate an option: the user must be able to repeat the label that
    # identifies the right candidate. ``tree.label`` is already capped at 120 chars.
    return [f'"{shorten(item, limit)}"' for item in items]


def _container_question(candidates: list[Candidate]) -> str:
    label = shorten(candidates[0].label)
    roles = {candidate.role for candidate in candidates}
    role_word = plural(candidates[0].role) if len(roles) == 1 else "controls"
    places = _quoted([candidate.container for candidate in candidates])
    return (
        f'There are {len(candidates)} "{label}" {role_word} on this page: '
        f"{_join_options(places)}. Which one do you mean?"
    )


def _role_question(candidates: list[Candidate]) -> str:
    label = shorten(candidates[0].label)
    roles = [candidate.role for candidate in candidates]
    return (
        f'There are several "{label}" controls on this page: '
        f"{_join_options(roles)}. Which one do you mean?"
    )


def _region_question(candidates: list[Candidate]) -> str:
    label = shorten(candidates[0].label)
    places = [f"the one in the {candidate.region}" for candidate in candidates]
    return (
        f'There are {len(candidates)} "{label}" {plural(candidates[0].role)} on this page. '
        f"Which one do you mean: {_join_options(places)}?"
    )


_ACTION = re.compile(
    r"^\s*\[[^\]]*\]\s*(?P<label>.*?)\s*->\s*(?P<op>[A-Z]+)(?::\s*(?P<value>.*))?$"
)


def humanize_action(action_repr: str) -> str:
    """``[button]  Delete -> CLICK`` -> ``click "Delete"`` (for a natural confirmation)."""
    match = _ACTION.match(str(action_repr or ""))
    if not match:
        return shorten(action_repr, 100) or "do that"
    label = shorten(match.group("label"), 60)
    op = match.group("op").lower()
    value = shorten(match.group("value") or "", 60)
    if op == "click":
        return f'click "{label}"'
    if op == "type":
        return f'type "{value}" into "{label}"'
    if op == "select":
        return f'select "{value}" in "{label}"'
    if op == "hover":
        return f'hover over "{label}"'
    return f'{op} "{label}"'


def _safety_question(action_repr: str) -> str:
    return (
        f"I am about to {humanize_action(action_repr)}. "
        "This may be irreversible — should I proceed?"
    )


def ambiguity_kind(candidates: list[Candidate]) -> str:
    """Best available disambiguation, strongest first: container > role > region."""
    if len(candidates) < 2:
        return "unique"
    containers = [candidate.container for candidate in candidates]
    if all(containers) and len({norm(value) for value in containers}) == len(candidates):
        return "container"
    roles = [candidate.role for candidate in candidates]
    groups = [role_group(role) for role in roles]
    families = {role_family(role) for role in roles}
    # Only alternatives inside one interaction family ("button vs link"), never a
    # label/input pair or a button next to the search box it submits.
    if (
        all(roles)
        and len(families) == 1
        and next(iter(families)) in {"action", "input"}
        and len(set(groups)) == len(candidates)
    ):
        return "role"
    regions = [candidate.region for candidate in candidates]
    if all(regions) and len(set(regions)) == len(candidates):
        return "region"
    return "indistinguishable"


def lowest_common_ancestor(tree: UiTree, node_ids: Iterable[int]) -> int | None:
    ids = list(dict.fromkeys(node_ids))
    if not ids:
        return None
    paths = [tree.path(node_id) for node_id in ids]
    common = set(paths[0])
    for path in paths[1:]:
        common &= set(path)
    for node_id in reversed(paths[0]):
        if node_id in common:
            return node_id
    return tree.root


def outline_subtree(
    tree: UiTree,
    root: int,
    *,
    keep: Iterable[int] | None = None,
    max_lines: int = 300,
    max_label: int = 120,
) -> str:
    """Indented outline of one subtree, with the subtree root at depth 0.

    ``keep`` nodes (the candidates and their ancestors) are never cut, even when the
    line budget is exhausted, so every option a question must name is visible.
    """
    lines: list[str] = []
    kept = set(keep or ())

    def walk(node_id: int, depth: int) -> None:
        if depth > 200 or (len(lines) >= max_lines and node_id not in kept):
            return
        node = tree.nodes[node_id]
        label = tree.label(node_id, max_len=max_label)
        shown = f' "{label}"' if label else ""
        lines.append(f"{'  ' * depth}- <{node.tag}> {node.role}{shown}")
        for child in node.children:
            walk(child, depth + 1)

    walk(root, 0)
    return "\n".join(lines)


def build_input(
    *,
    level: str,
    task: str,
    history: list[str],
    tree: UiTree,
    target_action: str = "",
    candidate_nodes: Iterable[int] | None = None,
    max_lines: int = 120,
) -> str:
    """Prompt the generator with the screen + history only — never the candidate list."""
    lines = [_LEVEL_INSTRUCTIONS[level], ""]
    lines.append(f"User's task: {task or '(not given)'}")
    if target_action:
        # The agent knows what it is about to do; it must not know *which* duplicate
        # it is. ``humanize_action`` drops the tag, so the L1 role answer is not leaked.
        lines.append(f"Planned action: {humanize_action(target_action)}")
    if history:
        lines.append("Steps already completed:")
        lines.extend(f"  {index}. {step}" for index, step in enumerate(history, start=1))
    else:
        lines.append("Steps already completed: none")
    lines.append("")
    nodes = list(candidate_nodes or [])
    root = lowest_common_ancestor(tree, nodes) if len(nodes) > 1 else None
    if root is not None:
        keep = {ancestor for node_id in nodes for ancestor in tree.path(node_id)}
        lines.append(
            "UI subtree around the planned action (no element is marked; use the labels "
            "and the parent/child structure to decide):"
        )
        lines.append(
            outline_subtree(
                tree,
                root,
                keep=keep,
                max_lines=max_lines,
                max_label=120,
            )
        )
    else:
        lines.append(
            "UI tree (the element the assistant is considering is NOT marked; "
            "use the labels and the parent/child structure to decide):"
        )
        lines.append(tree.outline(anchor=None, max_lines=max_lines, max_label=120))
    return "\n".join(lines).strip()


def _row(
    *,
    source: dict[str, Any],
    instance: dict[str, Any],
    split: str,
    level: str,
    track: str,
    ambiguity: str,
    target: str,
    answer: str | None,
    should_ask: bool,
    candidates: list[Candidate],
    answer_index: int | None,
    info_gain_bits: float,
    tree: UiTree,
    target_node: int | None,
    context: LevelContext,
    max_lines: int,
    stats: dict[str, int],
) -> dict[str, Any]:
    source_id = str(source.get("id") or source.get("image_name") or "row")
    return {
        "id": f"{source_id}::{level}",
        "level": level,
        "track": track,
        "input": build_input(
            level=level,
            task=context.task,
            history=context.action_history,
            tree=tree,
            target_action=context.target_action,
            candidate_nodes=(
                [
                    node_id
                    for candidate in candidates
                    for node_id in (candidate.node, candidate.container_node)
                    if node_id is not None
                ]
                if should_ask
                else None
            ),
            max_lines=max_lines,
        ),
        "target": target,
        "answer": answer,
        "image": source.get("image"),
        "tree": {**tree.to_dict(), "anchor": target_node},
        "candidates": [candidate.to_dict() for candidate in candidates],
        "answer_index": answer_index,
        "meta": {
            "category": f"clarify-{level}" if should_ask else "clarify-none",
            "level": level,
            "track": track,
            "ambiguity_type": ambiguity,
            "answer_source": "user" if should_ask else "none",
            "should_ask": should_ask,
            "n_candidates": len(candidates),
            "info_gain_bits": round(info_gain_bits, 3),
            "answer_index": answer_index,
            "anchor_id": target_node,
            "anchor_tag": tree.node(target_node).tag if target_node is not None else None,
            "anchor_role": tree.node(target_node).role if target_node is not None else None,
            "anchor_label": tree.label(target_node) if target_node is not None else "",
            "anchor_depth": tree.node(target_node).depth if target_node is not None else None,
            "tree_nodes": stats["nodes"],
            "tree_max_depth": stats["max_depth"],
            "target_matched": target_node is not None,
            "html_field": "cleaned",
            "annotation_id": source.get("annotation_id") or instance.get("annotation_id"),
            "action_uid": source.get("action_uid"),
            "split": source.get("split") or split,
            "task": context.task,
            "website": source.get("website") or instance.get("website"),
            "domain": source.get("domain") or instance.get("domain"),
            "target_action": context.target_action,
            "source_id": source.get("id"),
            "generator": "template-clarify-v0",
        },
    }


def build_rows_for_action(
    source: dict[str, Any],
    action: dict[str, Any],
    *,
    instance: dict[str, Any] | None = None,
    split: str = "?",
    html_field: str = "cleaned",
    include_safety: bool = True,
    include_negative: bool = True,
    max_lines: int = 120,
    max_candidates: int = 8,
) -> tuple[list[dict[str, Any]], list[str]]:
    """One action -> at most one ambiguity row, one safety row, one negative row."""
    field = HTML_FIELDS.get(html_field, html_field)
    html = action.get(field) or ""
    tree = parse_html(str(html))
    if len(tree) <= 1:
        return [], ["no_html"]

    apply_candidate_metadata(tree, action)
    instance = instance or {}
    exported_history = source.get("action_history")
    history = list(
        exported_history if exported_history is not None else (instance.get("action_reprs") or [])
    )
    context = LevelContext(
        task=str(source.get("task") or instance.get("confirmed_task") or ""),
        action_history=[str(step) for step in history],
        target_action=str(source.get("target_action") or ""),
        op=str(source.get("op") or ""),
        value=str(source.get("value") or ""),
        image_size=list(source.get("source_size") or source.get("image_size") or []) or None,
    )
    stats = tree_stats(tree)
    rows: list[dict[str, Any]] = []
    skipped: list[str] = []

    raw_target = target_from_candidates(tree, action)
    target, target_label = resolve_target(tree, raw_target)
    candidates: list[Candidate] = []
    if target is None:
        skipped.append("no_target")
    elif not target_label:
        skipped.append("no_target_label")
    elif is_weak_label(target_label):
        skipped.append("weak_label")
    else:
        candidates = candidate_set(
            tree,
            same_label_controls(tree, target_label),
            context.image_size,
        )
        mapped = map_to_candidate(tree, target, candidates)
        if mapped is None:
            candidates = []
            skipped.append("target_not_clickable")
        else:
            target = mapped

    kind = ambiguity_kind(candidates)
    if kind in {"container", "role", "region"} and len(candidates) > max_candidates:
        # A 37-option question is technically informative but unusable for a user.
        skipped.append("too_many_candidates")
        kind = "too_many"
    if kind in {"container", "role", "region"}:
        index = next(
            index for index, candidate in enumerate(candidates) if candidate.node == target
        )
        target_candidate = candidates[index]
        if kind == "container":
            level, ambiguous, answer = "L2", "container", target_candidate.container
            question = _container_question(candidates)
        elif kind == "role":
            level, ambiguous, answer = "L1", "role", target_candidate.role
            question = _role_question(candidates)
        else:
            level, ambiguous, answer = "L1", "region", target_candidate.region
            question = _region_question(candidates)
        rows.append(
            _row(
                source=source,
                instance=instance,
                split=split,
                level=level,
                track="ambiguity",
                ambiguity=ambiguous,
                target=question,
                answer=answer,
                should_ask=True,
                candidates=candidates,
                answer_index=index,
                info_gain_bits=math.log2(len(candidates)),
                tree=tree,
                target_node=target,
                context=context,
                max_lines=max_lines,
                stats=stats,
            )
        )
    elif kind == "indistinguishable":
        skipped.append("indistinguishable_candidates")

    target_action = context.target_action
    if include_safety and target_action and _IRREVERSIBLE.search(target_action):
        rows.append(
            _row(
                source=source,
                instance=instance,
                split=split,
                level="L3",
                track="safety",
                ambiguity="confirmation",
                target=_safety_question(target_action),
                answer="yes",
                should_ask=True,
                candidates=[],
                answer_index=None,
                info_gain_bits=0.0,
                tree=tree,
                target_node=target,
                context=context,
                max_lines=max_lines,
                stats=stats,
            )
        )

    # Only a genuinely unambiguous, safe step is a valid NO_QUESTION target. An
    # indistinguishable duplicate is still ambiguous — we just cannot verbalise a
    # disambiguation for it, so it must be skipped rather than taught as "do not ask".
    if include_negative and not rows and kind == "unique" and target is not None:
        rows.append(
            _row(
                source=source,
                instance=instance,
                split=split,
                level="none",
                track="negative",
                ambiguity=kind,
                target=NO_QUESTION,
                answer=None,
                should_ask=False,
                candidates=candidates,
                answer_index=None,
                info_gain_bits=0.0,
                tree=tree,
                target_node=target,
                context=context,
                max_lines=max_lines,
                stats=stats,
            )
        )

    if not rows and not skipped:
        skipped.append("no_rows")
    return rows, skipped


def build_split(
    split: str = "test_task",
    *,
    examples_path: str | Path | None = None,
    limit: int | None = 5,
    html_field: str = "cleaned",
    out_dir: str | Path | None = None,
    include_safety: bool = True,
    include_negative: bool = True,
    max_lines: int = 120,
    max_candidates: int = 8,
    include_missing_images: bool = False,
) -> dict[str, Any]:
    """Build the clarification JSONL + stats for one split."""
    limit = normalize_limit(limit)
    exported = load_exported(split, path=examples_path, limit=limit, export_if_missing=False)
    if not exported:
        raise FileNotFoundError(
            f"no exported examples for {split!r}. Run "
            f"`uv run m2w-mm-export --split {split} --limit {limit or 0}` first."
        )

    rows: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    by_level: Counter[str] = Counter()
    by_track: Counter[str] = Counter()
    by_ambiguity: Counter[str] = Counter()
    candidate_counts: list[int] = []
    gains: list[float] = []
    matched = 0

    for source, instance, action in join_split(split, exported, html_field=html_field):
        matched += 1
        image = source.get("image")
        if not include_missing_images and image and not Path(str(image)).is_file():
            skipped["missing_image"] += 1
            continue
        built, reasons = build_rows_for_action(
            source,
            action,
            instance=instance,
            split=split,
            html_field=html_field,
            include_safety=include_safety,
            include_negative=include_negative,
            max_lines=max_lines,
            max_candidates=max_candidates,
        )
        skipped.update(reasons)
        for row in built:
            by_level[row["level"]] += 1
            by_track[row["meta"]["track"]] += 1
            by_ambiguity[row["meta"]["ambiguity_type"]] += 1
            if row["meta"]["should_ask"]:
                candidate_counts.append(int(row["meta"]["n_candidates"]))
                gains.append(float(row["meta"]["info_gain_bits"]))
        rows.extend(built)

    skipped["unmatched_exported"] += len(exported) - matched
    dest = Path(out_dir) if out_dir else PROCESSED_DIR / "clarify" / split
    examples = tree_io.write_jsonl(rows, dest / "examples.jsonl")

    positives = by_track.get("ambiguity", 0) + by_track.get("safety", 0)
    total = len(rows)
    stats: dict[str, Any] = {
        "split": split,
        "track": "clarify-v0",
        "html_field": HTML_FIELDS.get(html_field, html_field),
        "exported_examples": len(exported),
        "matched_actions": matched,
        "built_examples": total,
        "by_level": dict(by_level),
        "by_track": dict(by_track),
        "by_ambiguity": dict(by_ambiguity),
        "positive_examples": positives,
        "negative_examples": by_track.get("negative", 0),
        "positive_rate": round(positives / matched, 3) if matched else 0.0,
        "mean_candidates": (
            round(sum(candidate_counts) / len(candidate_counts), 2) if candidate_counts else 0.0
        ),
        "mean_info_gain_bits": round(sum(gains) / len(gains), 3) if gains else 0.0,
        "skipped": {name: count for name, count in skipped.most_common() if count},
        "output": str(examples),
    }
    tree_io.write_json(stats, dest / "stats.json")
    return {"examples": examples, "stats": stats, "stats_path": dest / "stats.json"}


def print_summary(result: dict[str, Any]) -> None:
    stats = result["stats"]
    print(
        f"[clarify] {stats['matched_actions']}/{stats['exported_examples']} action(s) matched, "
        f"{stats['built_examples']} row(s)"
    )
    print("  tracks: " + "  ".join(f"{name}={count}" for name, count in stats["by_track"].items()))
    print("  levels: " + "  ".join(f"{name}={count}" for name, count in stats["by_level"].items()))
    print(
        f"  positives={stats['positive_examples']} "
        f"({stats['positive_rate']:.0%})  negatives={stats['negative_examples']}  "
        f"mean_candidates={stats['mean_candidates']}  "
        f"mean_info_gain={stats['mean_info_gain_bits']} bits"
    )
    if stats["skipped"]:
        print(
            "  skipped: " + ", ".join(f"{name}={count}" for name, count in stats["skipped"].items())
        )
    print(f"[clarify] -> {result['examples']}")
    print(f"[clarify] -> {result['stats_path']}")


def main(argv: list[str] | None = None) -> int:
    ensure_utf8()
    parser = argparse.ArgumentParser(
        description="Build TreeVQG-Clarify examples (intent questions, not screen captions)."
    )
    parser.add_argument("--split", default="test_task", choices=SPLITS)
    parser.add_argument(
        "--limit", type=int, default=5, help="max exported rows (default 5; 0 = whole split)"
    )
    parser.add_argument(
        "--examples",
        type=Path,
        default=None,
        help="input multimodal examples.jsonl (default: data/processed/multimodal/<split>)",
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="default: data/processed/clarify/<split>"
    )
    parser.add_argument("--html", default="cleaned", choices=sorted(HTML_FIELDS))
    parser.add_argument("--no-safety", action="store_true", help="do not emit L3 confirmation rows")
    parser.add_argument(
        "--no-negatives",
        action="store_true",
        help="do not emit NO_QUESTION rows for unambiguous/safe steps",
    )
    parser.add_argument(
        "--max-lines", type=int, default=120, help="tree outline lines in the prompt"
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=8,
        help="skip ambiguity rows with more than this many options (default 8)",
    )
    parser.add_argument(
        "--include-missing-images",
        action="store_true",
        help="keep rows whose screenshot is missing (default: drop + count)",
    )
    args = parser.parse_args(argv)

    result = build_split(
        args.split,
        examples_path=args.examples,
        limit=args.limit,
        html_field=args.html,
        out_dir=args.out,
        include_safety=not args.no_safety,
        include_negative=not args.no_negatives,
        max_lines=args.max_lines,
        max_candidates=args.max_candidates,
        include_missing_images=args.include_missing_images,
    )
    print_summary(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
