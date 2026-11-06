"""Parse Mind2Web's ``cleaned_html`` / ``raw_html`` into a :class:`~.schema.UiTree`.

Mind2Web ships the page as a **pruned DOM dump**, not as an Accessibility Tree, and it
does not always keep pixel boxes. Two consequences for the MVP:

* structure, tag, role and visible text are recovered exactly (that is what the parser
  below does);
* boxes are attached only when the dump carried a ``bounding_box_rect`` / ``bbox``
  attribute (or when the caller joins the recorded candidate metadata). The paper's
  Playwright pass that enriches every node is a later step — see ``docs/treevqg/plan.md``.

The parser is stdlib-only on purpose: the project keeps its dependency list short, and a
malformed page must never abort a whole corpus run.
"""

from __future__ import annotations

import json
from html.parser import HTMLParser
from typing import Any

from mind2web_vqg.treevqg.schema import UiNode, UiTree, build_node

#: Tags with no children.
VOID_TAGS: frozenset[str] = frozenset(
    {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }
)

#: Subtrees whose text is not page content.
SKIP_TAGS: frozenset[str] = frozenset(
    {"script", "style", "noscript", "template", "svg", "head"}
)

#: Tags that close an identical open tag (the HTML "implied end tag" rule).
AUTO_CLOSE: frozenset[str] = frozenset(
    {"li", "p", "option", "td", "th", "tr", "dt", "dd", "thead", "tbody", "tfoot"}
)

#: A ``<p>`` is also closed by a block-level start tag.
P_CLOSERS: frozenset[str] = frozenset(
    {
        "address",
        "article",
        "aside",
        "blockquote",
        "div",
        "dl",
        "fieldset",
        "footer",
        "form",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hr",
        "main",
        "nav",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "ul",
    }
)


def parse_bbox(value: Any) -> list[float] | None:
    """``"x,y,w,h"`` (or a JSON list, or ``"x y w h"``) -> ``[x, y, w, h]``."""
    if value in (None, ""):
        return None
    if isinstance(value, (list, tuple)):
        parts = [float(part) for part in value]
    else:
        text = str(value).strip()
        if text.startswith("["):
            try:
                parts = [float(part) for part in json.loads(text)]
            except (TypeError, ValueError):
                return None
        else:
            normalized = text.replace(",", " ").split()
            try:
                parts = [float(part) for part in normalized]
            except ValueError:
                return None
    return parts if len(parts) == 4 else None


def _bbox_from_attrs(attrs: dict[str, Any]) -> list[float] | None:
    for key in ("bounding_box_rect", "data-bbox", "bbox"):
        bbox = parse_bbox(attrs.get(key))
        if bbox is not None:
            return bbox
    return None


class _TreeBuilder(HTMLParser):
    """Streaming HTML parser that builds the flat preorder node list."""

    def __init__(self, *, max_nodes: int = 5000) -> None:
        super().__init__(convert_charrefs=True)
        self.nodes: list[UiNode] = [
            UiNode(id=0, tag="document", role="page", label="", depth=0, preorder=0)
        ]
        self.stack: list[int] = [0]
        self.max_nodes = max_nodes
        self._skip_depth = 0
        self._overflow = 0
        self._text: dict[int, list[str]] = {}

    # -- HTMLParser hooks -------------------------------------------------- #
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if self._skip_depth:
            if tag in SKIP_TAGS:
                self._skip_depth += 1
            return
        if tag in SKIP_TAGS:
            self._skip_depth = 1
            return

        if tag in AUTO_CLOSE and self.stack and self.nodes[self.stack[-1]].tag == tag:
            self._pop()
        elif tag in P_CLOSERS and self.stack and self.nodes[self.stack[-1]].tag == "p":
            self._pop()

        if len(self.nodes) >= self.max_nodes:
            if tag not in VOID_TAGS:
                self._overflow += 1
            return

        parent = self.stack[-1]
        node = build_node(
            len(self.nodes),
            tag,
            {key: value for key, value in attrs},
            parent=parent,
            depth=self.nodes[parent].depth + 1,
            bbox=_bbox_from_attrs({key: value for key, value in attrs}),
        )
        self.nodes.append(node)
        self.nodes[parent].children.append(node.id)
        if tag not in VOID_TAGS:
            self.stack.append(node.id)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if self._skip_depth:
            if tag in SKIP_TAGS:
                self._skip_depth -= 1
            return
        if self._overflow:
            if tag not in VOID_TAGS:
                self._overflow = max(0, self._overflow - 1)
            return
        for index in range(len(self.stack) - 1, 0, -1):
            if self.nodes[self.stack[index]].tag == tag:
                del self.stack[index:]
                return

    def handle_data(self, data: str) -> None:
        if self._skip_depth or self._overflow or not data.strip():
            return
        self._text.setdefault(self.stack[-1], []).append(data)

    def handle_entityref(self, name: str) -> None:  # pragma: no cover - covered by data
        self.handle_data(f"&{name};")

    def close(self) -> None:
        super().close()
        self.stack = [0]

    # -- finalization ------------------------------------------------------ #
    def _pop(self) -> None:
        if len(self.stack) > 1:
            self.stack.pop()

    def finalize(self) -> UiTree:
        """Attach direct text / labels and compute the topological indices."""
        for node_id, chunks in self._text.items():
            text = " ".join(" ".join(chunks).split())
            if text and not self.nodes[node_id].label:
                self.nodes[node_id].label = text
        tree = UiTree(nodes=self.nodes, root=0)
        tree.rebuild()
        return tree


def parse_html(fragment: str, *, max_nodes: int = 5000) -> UiTree:
    """Parse one HTML dump; a broken fragment yields the tree parsed so far."""
    builder = _TreeBuilder(max_nodes=max_nodes)
    try:
        builder.feed(fragment or "")
    except Exception:  # noqa: BLE001 - a bad page must not abort a corpus run
        pass
    finally:
        builder.close()
    return builder.finalize()


def tree_stats(tree: UiTree) -> dict[str, int]:
    """Small size summary used by ``stats.json``."""
    depths = [node.depth for node in tree.nodes]
    return {
        "nodes": len(tree),
        "clickable": sum(1 for node in tree.nodes if node.clickable),
        "labelled": sum(1 for node in tree.nodes if node.label),
        "max_depth": max(depths) if depths else 0,
    }
