"""Pick one anchor per granularity: Element (L1), Container (L2), Workflow (L3).

The proposal names depth ranges (leaf 8-12, container 4-8) from an Accessibility Tree.
Mind2Web gives us a pruned DOM, whose depth is both shallower and site-dependent, so the
selector first tries the paper's absolute bands and **rescales them to the tree's own
depth** when they would select nothing. That keeps the definition ("deep leaves" vs
"middle containers") instead of silently producing zero examples on a short page.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from mind2web_vqg.treevqg.roles import is_container
from mind2web_vqg.treevqg.schema import UiTree

#: Leading ``[tag]`` of an action-repr string, e.g. ``"[link]  Budget Truck -> CLICK"``.
_TAG_PREFIX = re.compile(r"^\s*\[[^\]]*\]\s*")

#: Structural tags that are never interesting anchors on their own.
_BORING = frozenset({"document", "html", "body", "head"})


@dataclass(frozen=True)
class DepthBands:
    """Inclusive depth ranges (matching the paper's 8-12 / 4-8 A11y depths)."""

    element: tuple[int, int] = (8, 12)
    container: tuple[int, int] = (4, 8)


#: Depth scale the paper assumes for an Accessibility Tree (leaf 8-12, container 4-8).
_PAPER_MAX_DEPTH = 12


def _max_depth(tree: UiTree) -> int:
    return max((node.depth for node in tree.nodes), default=0)


def resolve_bands(tree: UiTree, bands: DepthBands | None = None) -> DepthBands:
    """Use the paper's absolute bands, or rescale them to this tree's depth."""
    bands = bands or DepthBands()
    max_depth = _max_depth(tree)

    def count(rng: tuple[int, int]) -> int:
        low, high = rng
        return sum(1 for node in tree.nodes[1:] if low <= node.depth <= high)

    element, container = bands.element, bands.container
    if count(element) == 0 and max_depth > 0:
        start = max(1, round(bands.element[0] / _PAPER_MAX_DEPTH * max_depth))
        element = (start, max_depth)
    if count(container) == 0 and max_depth > 0:
        start = max(1, round(bands.container[0] / _PAPER_MAX_DEPTH * max_depth))
        end = max(start, round(bands.container[1] / _PAPER_MAX_DEPTH * max_depth))
        container = (start, end)
    return DepthBands(element=element, container=container)


def action_repr_label(action: object) -> str:
    """Text of the target element in ``"[link]  Budget Truck -> CLICK"`` (or ``""``)."""
    text = str(action or "").strip()
    if not _TAG_PREFIX.match(text):
        return ""
    return _TAG_PREFIX.sub("", text).rsplit("->", 1)[0].strip()


def _normalize(text: object) -> str:
    return " ".join(str(text or "").casefold().split())


def target_node(
    tree: UiTree,
    *,
    tag: str | None = None,
    text: str | None = None,
    action_repr: str | None = None,
) -> int | None:
    """Find the recorded target element inside the parsed tree.

    Mind2Web stores the target's tag + label in the candidate metadata, so a match is
    usually exact. When it is not, we return ``None`` and the caller falls back to a
    deterministic in-band anchor instead of inventing a wrong one.
    """
    want_tag = str(tag or "").strip().lower()
    want = _normalize(text) or _normalize(action_repr_label(action_repr))
    best_id: int | None = None
    best_score = 0
    for node in tree.nodes[1:]:
        score = 0
        if want_tag and node.tag != want_tag:
            continue
        label = _normalize(tree.label(node.id))
        if want and label:
            if label == want:
                score = 4
            elif want in label:
                score = 3
            elif label in want:
                score = 2
        if score == 0 and want_tag:
            score = 1
        if score > best_score or (score == best_score and score and node.id < (best_id or 10**9)):
            best_id, best_score = node.id, score
    return best_id


def _interactive_count(tree: UiTree, node_id: int) -> int:
    return sum(1 for nid in tree.subtree(node_id) if tree.nodes[nid].clickable and nid != node_id)


def element_candidates(tree: UiTree, band: tuple[int, int]) -> list[int]:
    """Actionable nodes whose depth sits in ``band`` (fallbacks keep it non-empty)."""
    low, high = band
    in_band = [
        node.id
        for node in tree.nodes[1:]
        if low <= node.depth <= high and node.tag not in _BORING
    ]
    actionable = [nid for nid in in_band if tree.nodes[nid].clickable]
    if actionable:
        return actionable
    labelled = [nid for nid in in_band if tree.label(nid)]
    if labelled:
        return labelled
    return [node.id for node in tree.nodes[1:] if node.clickable]


def container_candidates(
    tree: UiTree, band: tuple[int, int], *, min_descendants: int = 2
) -> list[int]:
    """Middle-level containers holding at least ``min_descendants`` interactive nodes."""
    low, high = band
    in_band = [
        node.id
        for node in tree.nodes[1:]
        if low <= node.depth <= high
        and node.tag not in _BORING
        and is_container(node.tag, node.attrs)
    ]
    for threshold in (min_descendants, 1):
        found = [nid for nid in in_band if _interactive_count(tree, nid) >= threshold]
        if found:
            return found
    return in_band


def choose_element_anchor(tree: UiTree, *, target: int | None, band: tuple[int, int]) -> int | None:
    """Prefer the recorded target; otherwise the most informative in-band control."""
    if target is not None and target != tree.root:
        return target
    candidates = element_candidates(tree, band)
    if not candidates:
        return None
    return max(candidates, key=lambda nid: (len(tree.label(nid)), -nid))


def choose_container_anchor(
    tree: UiTree, *, element: int | None, band: tuple[int, int]
) -> int | None:
    """The nearest container around the element anchor (or the richest in-band one)."""
    if element is not None:
        for ancestor in reversed(tree.path(element)[:-1]):
            node = tree.nodes[ancestor]
            if node.tag in _BORING or not is_container(node.tag, node.attrs):
                continue
            if _interactive_count(tree, ancestor) >= 1:
                return ancestor
    candidates = container_candidates(tree, band)
    if not candidates:
        return None
    return max(candidates, key=lambda nid: (_interactive_count(tree, nid), -nid))


def select_anchors(
    tree: UiTree,
    *,
    target: int | None = None,
    levels: tuple[str, ...] | list[str] = ("L1", "L2", "L3"),
    bands: DepthBands | None = None,
) -> dict[str, int]:
    """Return ``{"L1": node_id, "L2": node_id, "L3": root}`` for the requested levels.

    A level is omitted when the tree has nothing suitable; the builder records the reason
    in ``stats.json`` rather than emitting an unanswerable question.
    """
    resolved = resolve_bands(tree, bands)
    anchors: dict[str, int] = {}
    element: int | None = None
    if "L1" in levels:
        element = choose_element_anchor(tree, target=target, band=resolved.element)
        if element is not None:
            anchors["L1"] = element
    if "L2" in levels:
        container = choose_container_anchor(tree, element=element, band=resolved.container)
        if container is not None:
            anchors["L2"] = container
    if "L3" in levels and tree.nodes:
        anchors["L3"] = tree.root
    return anchors
