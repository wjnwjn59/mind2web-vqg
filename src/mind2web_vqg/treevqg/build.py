"""Join exported screenshots with the raw HTML dump and emit the TreeVQG JSONL.

One input row = one action of one Mind2Web task (see ``m2w-mm-export``). One output row =
one **anchor** at one level, carrying everything the rest of the pipeline needs:

``{id, level, input, target, answer, image, tree, meta}``

* ``input`` — level instruction + task + completed steps + a text outline of the tree;
* ``target`` / ``answer`` — the reference question and its ground truth;
* ``tree`` — the serialized UI tree plus the anchor id (the structural condition);
* ``meta`` — level, anchor, provenance and tree size (for slicing/reporting).

The join is **streaming**: only the exported rows the caller asked for are kept in memory,
so ``--limit`` also bounds RAM when the raw split is large.

Usage::

    uv run m2w-mm-export --split test_task --limit 5
    uv run m2w-build-treevqg --split test_task --limit 5
    # -> data/processed/treevqg/test_task/{examples.jsonl,stats.json}

    uv run m2w-build-treevqg --split train --limit 0 --levels L1 L2 --max-tree-nodes 600
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from mind2web_vqg.cli import ensure_utf8
from mind2web_vqg.data.load import iter_split
from mind2web_vqg.data.multimodal import SPLITS, normalize_limit
from mind2web_vqg.data.vqg import load_exported
from mind2web_vqg.paths import PROCESSED_DIR
from mind2web_vqg.treevqg import LEVEL_NAMES, LEVELS
from mind2web_vqg.treevqg import io as tree_io
from mind2web_vqg.treevqg.levels import select_anchors, target_node
from mind2web_vqg.treevqg.parse import parse_html, tree_stats
from mind2web_vqg.treevqg.templates import LevelContext, build_input, question_for_level

#: ``--html`` -> the action field it reads.
HTML_FIELDS: dict[str, str] = {"cleaned": "cleaned_html", "raw": "raw_html"}


# --------------------------------------------------------------------------- #
# Join helpers
# --------------------------------------------------------------------------- #
def action_key(annotation_id: Any, action_uid: Any, fallback: Any = "") -> str:
    """Stable join key between an exported row and a raw action.

    ``action_uid`` is present in both sources and is the reliable key. The fallback keeps
    the builder honest on hand-made fixtures instead of silently pairing by row order.
    """
    if action_uid:
        return f"uid:{action_uid}"
    return f"row:{annotation_id}:{fallback}"


def join_split(
    split: str,
    exported: list[dict[str, Any]],
    *,
    html_field: str = "cleaned",
) -> Iterator[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]:
    """Yield ``(exported_row, instance, action)`` for matching rows only."""
    wanted = {
        action_key(row.get("annotation_id"), row.get("action_uid"), row.get("target_action")): row
        for row in exported
    }
    for instance in iter_split(split):
        annotation_id = instance.get("annotation_id")
        for action in instance.get("actions") or []:
            key = action_key(
                annotation_id,
                action.get("action_uid"),
                action.get("target_action_reprs") or "",
            )
            row = wanted.pop(key, None)
            if row is not None:
                yield row, instance, action


# --------------------------------------------------------------------------- #
# One action -> up to one row per level (pure apart from HTML parsing)
# --------------------------------------------------------------------------- #
def build_rows_for_action(
    source: dict[str, Any],
    action: dict[str, Any],
    *,
    instance: dict[str, Any] | None = None,
    split: str = "?",
    html_field: str = "cleaned",
    levels: Iterable[str] = LEVELS,
    max_tree_nodes: int = 800,
    max_lines: int = 60,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Turn one exported row + one raw action into TreeVQG rows.

    Returns ``(rows, skipped)`` where ``skipped`` holds ``"<reason>:<level>"`` strings, so
    the caller can report *why* an example could not be produced.
    """
    requested = tuple(levels)
    unknown = sorted(set(requested) - set(LEVELS))
    if unknown:
        raise ValueError(f"unknown level(s): {unknown}. Choose from {list(LEVELS)}")

    field = HTML_FIELDS.get(html_field, html_field)
    html = action.get(field) or ""
    tree = parse_html(str(html))
    if len(tree) <= 1:  # only the synthetic document root: no usable HTML
        return [], ["no_html"]

    instance = instance or {}
    history = list(source.get("action_history") or instance.get("action_reprs") or [])
    target = target_node(
        tree,
        tag=source.get("target_tag"),
        text=source.get("target_text"),
        action_repr=source.get("target_action") or instance.get("target_action_reprs"),
    )
    context = LevelContext(
        task=str(source.get("task") or instance.get("confirmed_task") or ""),
        action_history=[str(step) for step in history],
        target_action=str(source.get("target_action") or ""),
        op=str(source.get("op") or ""),
        value=str(source.get("value") or ""),
        image_size=list(source.get("image_size") or source.get("source_size") or []) or None,
    )
    anchors = select_anchors(tree, target=target, levels=requested)
    stats = tree_stats(tree)

    rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    for level in requested:
        anchor = anchors.get(level)
        if anchor is None:
            skipped.append(f"no_anchor:{level}")
            continue
        question = question_for_level(
            tree, anchor, level, context, element_anchor=anchors.get("L1")
        )
        if question is None:
            skipped.append(f"no_question:{level}")
            continue

        view, remap = tree.pruned_with_map(max_tree_nodes, anchor=anchor)
        new_anchor = remap.get(anchor, 0)
        source_id = str(source.get("id") or source.get("image_name") or "row")
        node = tree.node(anchor)
        rows.append(
            {
                "id": f"{source_id}::{level}",
                "level": level,
                "input": build_input(
                    tree, anchor, level, context, max_lines=max_lines
                ),
                "target": question["question"],
                "answer": question["answer"],
                "image": source.get("image"),
                "tree": {**view.to_dict(), "anchor": new_anchor},
                "meta": {
                    # ``category`` is what the offline metrics group by.
                    "category": level,
                    "level": level,
                    "level_name": LEVEL_NAMES[level],
                    "hint": question["hint"],
                    "answer_type": question["answer_type"],
                    "answer_source": question["answer_source"],
                    "anchor_id": new_anchor,
                    "anchor_source_id": anchor,
                    "anchor_tag": node.tag,
                    "anchor_role": node.role,
                    "anchor_label": tree.label(anchor),
                    "anchor_depth": node.depth,
                    "tree_nodes": len(view),
                    "full_tree_nodes": stats["nodes"],
                    "tree_max_depth": stats["max_depth"],
                    "target_matched": target is not None,
                    "html_field": field,
                    "annotation_id": source.get("annotation_id")
                    or instance.get("annotation_id"),
                    "action_uid": source.get("action_uid") or action.get("action_uid"),
                    "split": source.get("split") or split,
                    "task": context.task,
                    "website": source.get("website") or instance.get("website"),
                    "domain": source.get("domain") or instance.get("domain"),
                    "target_action": context.target_action,
                    "source_id": source.get("id"),
                },
            }
        )
    return rows, skipped


# --------------------------------------------------------------------------- #
# Whole split
# --------------------------------------------------------------------------- #
def build_split(
    split: str = "test_task",
    *,
    examples_path: str | Path | None = None,
    limit: int | None = 5,
    html_field: str = "cleaned",
    levels: Iterable[str] = LEVELS,
    out_dir: str | Path | None = None,
    max_tree_nodes: int = 800,
    max_lines: int = 60,
    include_missing_images: bool = False,
) -> dict[str, Any]:
    """Build ``examples.jsonl`` + ``stats.json`` for one split and return a summary."""
    limit = normalize_limit(limit)
    exported = load_exported(split, path=examples_path, limit=limit, export_if_missing=False)
    if not exported:
        raise FileNotFoundError(
            f"no exported examples for {split!r}. Run "
            f"`uv run m2w-mm-export --split {split} --limit {limit or 0}` first."
        )

    rows: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    level_counts: Counter[str] = Counter()
    matched = 0
    tree_nodes: list[int] = []
    tree_depths: list[int] = []

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
            levels=levels,
            max_tree_nodes=max_tree_nodes,
            max_lines=max_lines,
        )
        skipped.update(reasons)
        if not built:
            skipped["no_rows"] += 1
        for row in built:
            level_counts[row["level"]] += 1
            tree_nodes.append(int(row["meta"]["tree_nodes"]))
            tree_depths.append(int(row["meta"]["tree_max_depth"]))
        rows.extend(built)

    skipped["unmatched_exported"] += len(exported) - matched
    dest = Path(out_dir) if out_dir else PROCESSED_DIR / "treevqg" / split
    examples = tree_io.write_jsonl(rows, dest / "examples.jsonl")

    stats: dict[str, Any] = {
        "split": split,
        "html_field": HTML_FIELDS.get(html_field, html_field),
        "levels": list(levels),
        "exported_examples": len(exported),
        "matched_actions": matched,
        "built_examples": len(rows),
        "by_level": {level: level_counts.get(level, 0) for level in LEVELS},
        "skipped": {name: count for name, count in skipped.most_common() if count},
        "tree": {
            "mean_nodes": round(sum(tree_nodes) / len(tree_nodes), 1) if tree_nodes else 0.0,
            "max_nodes": max(tree_nodes) if tree_nodes else 0,
            "mean_max_depth": (
                round(sum(tree_depths) / len(tree_depths), 1) if tree_depths else 0.0
            ),
        },
        "max_tree_nodes": max_tree_nodes,
        "output": str(examples),
    }
    tree_io.write_json(stats, dest / "stats.json")
    return {"examples": examples, "stats": stats, "stats_path": dest / "stats.json"}


def print_summary(result: dict[str, Any]) -> None:
    stats = result["stats"]
    print(
        f"[treevqg] {stats['matched_actions']}/{stats['exported_examples']} action(s) matched, "
        f"{stats['built_examples']} example(s)"
    )
    print("  " + "  ".join(f"{level}={count}" for level, count in stats["by_level"].items()))
    if stats["skipped"]:
        shown = ", ".join(f"{name}={count}" for name, count in stats["skipped"].items())
        print(f"  skipped: {shown}")
    print(f"[treevqg] -> {result['examples']}")
    print(f"[treevqg] -> {result['stats_path']}")


def main(argv: list[str] | None = None) -> int:
    ensure_utf8()
    parser = argparse.ArgumentParser(
        description="Build TreeVQG examples (3 structural levels) from Mind2Web HTML."
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
        "--out", type=Path, default=None, help="default: data/processed/treevqg/<split>"
    )
    parser.add_argument("--html", default="cleaned", choices=sorted(HTML_FIELDS))
    parser.add_argument(
        "--levels", nargs="+", default=list(LEVELS), choices=list(LEVELS), help="levels to build"
    )
    parser.add_argument(
        "--max-tree-nodes", type=int, default=800, help="stored tree is pruned to this size"
    )
    parser.add_argument(
        "--max-lines", type=int, default=60, help="tree outline lines in the text prompt"
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
        levels=args.levels,
        out_dir=args.out,
        max_tree_nodes=args.max_tree_nodes,
        max_lines=args.max_lines,
        include_missing_images=args.include_missing_images,
    )
    print_summary(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
