"""The UI tree: one node type, one tree type, and the JSON form stored next to each row.

A screenshot alone hides *which* element belongs to *which* container. The paper's fix is
to make that hierarchy an explicit condition, so every example carries the DOM/A11y tree
it was generated from. This module is the contract between the data pipeline and any
model (text baseline now, structural encoder later):

``nodes[*]`` fields
    ``id``          stable index; **preorder**, parents always come before children
    ``parent``      id of the parent (``null`` for the root)
    ``tag``         HTML tag
    ``role``        human word (``button``, ``text field``, ``form``, ...)
    ``label``       visible text / aria-label / title / placeholder (may be empty)
    ``clickable``   can the user act on it
    ``bbox``        ``[x, y, w, h]`` when the source provided one, else ``null``
    ``depth``       distance from the root (root = 0)
    ``preorder`` / ``postorder``  topological indices (same as ``id`` for preorder)
    ``attrs``       small whitelist of HTML attributes (class, type, href, ...)

Everything here is pure Python and unit-testable without a browser or a dataset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mind2web_vqg.treevqg.roles import KEPT_ATTRS, element_role, is_interactive


@dataclass
class UiNode:
    """One element of the interface."""

    id: int
    tag: str
    role: str = ""
    label: str = ""
    attrs: dict[str, str] = field(default_factory=dict)
    depth: int = 0
    preorder: int = 0
    postorder: int = 0
    parent: int | None = None
    children: list[int] = field(default_factory=list)
    clickable: bool = False
    bbox: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "parent": self.parent,
            "tag": self.tag,
            "role": self.role,
            "label": self.label,
            "clickable": self.clickable,
            "bbox": self.bbox,
            "depth": self.depth,
            "preorder": self.preorder,
            "postorder": self.postorder,
            "attrs": self.attrs,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UiNode:
        return cls(
            id=int(data["id"]),
            tag=str(data.get("tag") or ""),
            role=str(data.get("role") or ""),
            label=str(data.get("label") or ""),
            attrs={str(k): str(v) for k, v in (data.get("attrs") or {}).items()},
            depth=int(data.get("depth") or 0),
            preorder=int(data.get("preorder") or data["id"]),
            postorder=int(data.get("postorder") or data["id"]),
            parent=None if data.get("parent") is None else int(data["parent"]),
            clickable=bool(data.get("clickable")),
            bbox=list(data["bbox"]) if data.get("bbox") else None,
        )


@dataclass
class UiTree:
    """A parsed interface, stored as a flat preorder list (easy to batch/tensorize)."""

    nodes: list[UiNode]
    root: int = 0

    # -- basic navigation ------------------------------------------------- #
    def __len__(self) -> int:
        return len(self.nodes)

    def node(self, node_id: int) -> UiNode:
        return self.nodes[node_id]

    def path(self, node_id: int | None) -> list[int]:
        """Root -> ``node_id`` (inclusive); ``[]`` for ``None``."""
        if node_id is None:
            return []
        out: list[int] = []
        current: int | None = node_id
        while current is not None:
            out.append(current)
            current = self.nodes[current].parent
        return list(reversed(out))

    def subtree(self, node_id: int) -> list[int]:
        """All descendants of ``node_id`` in preorder, including the node itself."""
        out: list[int] = []
        stack = [node_id]
        while stack:
            current = stack.pop()
            out.append(current)
            stack.extend(reversed(self.nodes[current].children))
        return out

    def is_ancestor(self, ancestor: int, node_id: int) -> bool:
        return ancestor in set(self.path(node_id))

    def label(self, node_id: int, *, max_len: int = 120) -> str:
        """Visible name of a node, falling back to a descendant heading/label.

        Containers often have no own text (``<div class="card">``); their title lives in a
        child heading or a labelled control. Falling back makes Level-2 questions readable
        without changing the stored node.
        """
        own = self.nodes[node_id].label.strip()
        if own:
            return _truncate(own, max_len)
        # The structural shell has no meaningful visible name; its descendants do, but
        # inheriting one would make the prompt say `body "Main"` — confusing, not helpful.
        if self.nodes[node_id].tag.lower() in {"document", "html", "body", "head"}:
            return ""
        # Prefer a heading, then the first labelled child (breadth-first, shallow first).
        for child in self.nodes[node_id].children:
            tag = self.nodes[child].tag.lower()
            if tag in {"h1", "h2", "h3", "h4", "h5", "h6"} and self.nodes[child].label:
                return _truncate(self.nodes[child].label, max_len)
        queue = list(self.nodes[node_id].children)
        while queue:
            current = queue.pop(0)
            if self.nodes[current].label:
                return _truncate(self.nodes[current].label, max_len)
            queue.extend(self.nodes[current].children)
        return ""

    def interactive(self, node_id: int) -> list[int]:
        """Interactive nodes inside ``node_id`` (preorder)."""
        return [nid for nid in self.subtree(node_id) if self.nodes[nid].clickable]

    # -- serialization ---------------------------------------------------- #
    def to_dict(self) -> dict[str, Any]:
        return {"root": self.root, "nodes": [node.to_dict() for node in self.nodes]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UiTree:
        tree = cls(
            nodes=[UiNode.from_dict(node) for node in data.get("nodes") or []],
            root=int(data.get("root") or 0),
        )
        tree.rebuild()
        return tree

    # -- structural operations -------------------------------------------- #
    def rebuild(self) -> UiTree:
        """Recompute children/depth/preorder/postorder from ``parent`` links in place.

        Node ids are left untouched. After :meth:`pruned` they are assigned in preorder,
        so ``id == preorder``; after loading an external tree this keeps the indices
        consistent even if the file was hand-edited.
        """
        for node in self.nodes:
            node.children = []
        for node in self.nodes:
            if node.parent is not None:
                if node.parent < 0 or node.parent >= len(self.nodes):
                    node.parent = None
                else:
                    self.nodes[node.parent].children.append(node.id)

        order: list[int] = []
        depths: dict[int, int] = {}
        roots = [self.root] if self.root < len(self.nodes) else []
        roots += [
            node.id
            for node in self.nodes
            if node.parent is None and node.id not in roots
        ]
        for root in roots:
            stack = [(root, 0)]
            while stack:
                node_id, depth = stack.pop()
                if node_id in depths:  # cycle guard: malformed input must not hang
                    continue
                depths[node_id] = depth
                order.append(node_id)
                for child in reversed(self.nodes[node_id].children):
                    stack.append((child, depth + 1))
        # Any node unreachable from a root (bad parent chain) is appended flat.
        for node in self.nodes:
            if node.id not in depths:
                order.append(node.id)
                depths[node.id] = node.depth

        postorder: dict[int, int] = {}
        for index, node_id in enumerate(order):
            self.nodes[node_id].preorder = index
            self.nodes[node_id].depth = depths[node_id]
            postorder[node_id] = index
        # A real postorder index: children before parents.
        counter = 0
        visited: set[int] = set()
        for root in roots:
            stack: list[tuple[int, bool]] = [(root, False)]
            while stack:
                node_id, done = stack.pop()
                if done:
                    if node_id not in visited:
                        postorder[node_id] = counter
                        counter += 1
                        visited.add(node_id)
                    continue
                stack.append((node_id, True))
                for child in reversed(self.nodes[node_id].children):
                    stack.append((child, False))
        for node in self.nodes:
            node.postorder = postorder.get(node.id, node.preorder)
        return self

    def pruned(self, max_nodes: int | None = None, *, anchor: int | None = None) -> UiTree:
        """A smaller copy that always keeps the root, the path to ``anchor``, its subtree.

        The full cleaned HTML can run to thousands of nodes; a prompt or a first-batch
        tensor does not need all of them. The anchor path is never cut, so the level
        condition survives pruning.
        """
        return self.pruned_with_map(max_nodes, anchor=anchor)[0]

    def pruned_with_map(
        self, max_nodes: int | None = None, *, anchor: int | None = None
    ) -> tuple[UiTree, dict[int, int]]:
        """:meth:`pruned` plus the ``old id -> new id`` map (needed to re-point the anchor)."""
        if not max_nodes or len(self.nodes) <= max_nodes:
            return UiTree.from_dict(self.to_dict()), {node.id: node.id for node in self.nodes}

        keep: set[int] = set(self.path(anchor)) if anchor is not None else set()
        if anchor is not None:
            keep.update(self.subtree(anchor))
        for node in self.nodes:
            if len(keep) >= max_nodes:
                break
            keep.add(node.id)

        old_order = [node.id for node in self.nodes]
        kept_order = [node_id for node_id in old_order if node_id in keep]
        remap = {old: new for new, old in enumerate(kept_order)}
        nodes: list[UiNode] = []
        for old_id in kept_order:
            old = self.nodes[old_id]
            parent = old.parent
            while parent is not None and parent not in remap:
                parent = self.nodes[parent].parent
            nodes.append(
                UiNode(
                    id=remap[old_id],
                    tag=old.tag,
                    role=old.role,
                    label=old.label,
                    attrs=dict(old.attrs),
                    depth=old.depth,
                    preorder=remap[old_id],
                    postorder=remap[old_id],
                    parent=remap[parent] if parent is not None else None,
                    clickable=old.clickable,
                    bbox=list(old.bbox) if old.bbox else None,
                )
            )
        tree = UiTree(nodes=nodes, root=0)
        tree.rebuild()
        return tree, remap

    # -- prompt rendering -------------------------------------------------- #
    def outline(
        self,
        *,
        anchor: int | None = None,
        max_lines: int = 60,
        max_label: int = 60,
    ) -> str:
        """A compact indented view of the tree for text-only models.

        The anchor's path is rendered even when the budget is exhausted (it is what the
        question is about); unrelated branches are dropped first. The anchor is marked
        with ``*`` so a prompt can point at the target granularity.
        """
        if not self.nodes:
            return ""
        path = set(self.path(anchor)) if anchor is not None else set()
        lines: list[str] = []

        def walk(node_id: int, depth: int) -> None:
            if depth > 200:  # a pathological single-child chain must not blow the stack
                return
            if len(lines) >= max_lines and node_id not in path:
                return
            node = self.nodes[node_id]
            label = self.label(node_id, max_len=max_label)
            shown = f' "{label}"' if label else ""
            mark = " *" if node_id == anchor else ""
            lines.append(f"{'  ' * depth}- <{node.tag}> {node.role}{shown}{mark}")
            for child in node.children:
                walk(child, depth + 1)

        walk(self.root, 0)
        return "\n".join(lines)


def _truncate(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def build_node(
    node_id: int,
    tag: str,
    attrs: dict[str, Any],
    *,
    parent: int | None,
    depth: int,
    text: str = "",
    bbox: list[float] | None = None,
) -> UiNode:
    """Construct a node from raw HTML attributes (shared by the parser and tests)."""
    clean_attrs = {
        key: str(attrs[key])
        for key in KEPT_ATTRS
        if attrs.get(key) not in (None, "")
    }
    label = ""
    for key in ("aria-label", "title", "alt", "placeholder", "data-placeholder"):
        value = str(attrs.get(key) or "").strip()
        if value:
            label = value
            break
    if not label:
        label = str(text or "").strip()
    if not label and tag.lower() in {"input", "button", "option"}:
        label = str(attrs.get("value") or "").strip()

    return UiNode(
        id=node_id,
        tag=tag.lower(),
        role=element_role(tag, attrs),
        label=" ".join(label.split()),
        attrs=clean_attrs,
        depth=depth,
        preorder=node_id,
        postorder=node_id,
        parent=parent,
        clickable=is_interactive(tag, attrs),
        bbox=bbox,
    )
