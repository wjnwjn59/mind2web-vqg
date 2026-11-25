"""Export the self-supervised VQG set to ms-swift's multimodal SFT format.

ms-swift (``swift sft``) reads one JSON object per line::

    {"messages": [{"role": "user", "content": "<image>... prompt ..."},
                  {"role": "assistant", "content": "... reference question ..."}],
     "images": ["/abs/path/to/screenshot.jpg"]}

ms-swift replaces the ``<image>`` placeholder with the model's own image token, so the
same file works for Qwen-VL and for any other VLM it supports.

Why this module exists
----------------------
``data/vqg.py`` emits ``{id, input, target, answer, image, meta}`` — deliberately
framework-neutral. This module is the **only** place that knows the ms-swift schema, so
swapping trainers means writing one converter, not rewriting the dataset code
(see ``docs/train-framework.md``).

Three details that matter for a real run:

* **Images are absolute.** A relative path is resolved against the JSONL's folder and
  then written absolute, so moving the file to the lab server cannot silently break it.
  ``--image-root`` remaps everything when the screenshots live elsewhere.
* **The split is grouped by task (``annotation_id``).** Neighbouring actions of one task
  share a page state; a per-row split would put near-identical screenshots in both
  splits and inflate every metric. One task is one group, and a group never straddles.
* **A missing screenshot is dropped and counted**, never written as a broken path (that
  would only fail hours later, inside the trainer).

Usage::

    uv run m2w-swift-data --split train --val-fraction 0.05
    # -> data/processed/vqg/train/swift/{train.jsonl,val.jsonl,stats.json}
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from mind2web_vqg.cli import ensure_utf8
from mind2web_vqg.data.multimodal import SPLITS
from mind2web_vqg.data.vqg import (
    ALL_CATEGORIES,
    SCREENSHOT_GROUNDED,
    build_examples,
    load_exported,
    write_examples,
)
from mind2web_vqg.paths import PROCESSED_DIR

#: Placeholder ms-swift expands into the model's image token.
IMAGE_PLACEHOLDER = "<image>"

#: Second user turn used by ``--with-answer`` (question generation + answering).
ANSWER_PROMPT = "Answer the question you just asked, using only the screenshot."


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested, no I/O)
# --------------------------------------------------------------------------- #
def group_key(example: dict[str, Any]) -> str:
    """Group id for splitting: one Mind2Web task = one group.

    Prefers ``annotation_id`` (a whole task), then the source action, then the row id.
    """
    meta = example.get("meta") or {}
    for key in ("annotation_id", "source_id"):
        value = meta.get(key)
        if value:
            return str(value)
    return str(example.get("id") or "")


def category_of(example: dict[str, Any]) -> str:
    """VQG category of an example (``target_label`` / ``element_type`` / ...)."""
    return str((example.get("meta") or {}).get("category") or "unknown")


def resolve_image_path(
    image: Any,
    *,
    image_root: str | Path | None = None,
    examples_path: str | Path | None = None,
) -> Path | None:
    """Absolute path of one screenshot, remapped if needed.

    Order: ``image_root/<basename>`` > absolute path > relative to ``examples.jsonl`` >
    relative to the current directory.
    """
    if not image:
        return None
    path = Path(str(image))
    if image_root is not None:
        return (Path(image_root) / path.name).resolve()
    if path.is_absolute():
        return path
    if examples_path is not None:
        return (Path(examples_path).parent / path).resolve()
    return path.resolve()


def filter_missing_images(
    examples: list[dict[str, Any]], *, image_root: str | Path | None = None
) -> tuple[list[dict[str, Any]], list[str]]:
    """Split into ``(kept, missing_paths)`` so callers can report what was dropped."""
    kept: list[dict[str, Any]] = []
    missing: list[str] = []
    for example in examples:
        resolved = resolve_image_path(example.get("image"), image_root=image_root)
        if resolved is not None and resolved.is_file():
            kept.append(example)
        else:
            missing.append(str(example.get("image")))
    return kept, missing


def sample_per_category(
    examples: list[dict[str, Any]],
    max_per_category: int | None,
    *,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Cap each VQG category at ``max_per_category`` rows, deterministically.

    The EDA reports how many questions each category can supply; this is the knob that
    implements its sampling budget. Rows are sorted by id before shuffling so the result
    does not depend on file order.
    """
    if not max_per_category or max_per_category <= 0:
        return list(examples)

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for example in examples:
        buckets[category_of(example)].append(example)

    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    for category in sorted(buckets):
        rows = sorted(buckets[category], key=lambda ex: str(ex.get("id") or ""))
        rng.shuffle(rows)
        out.extend(rows[:max_per_category])
    return out


def split_examples(
    examples: list[dict[str, Any]], *, val_fraction: float = 0.05, seed: int = 42
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Group-aware ``(train, val)`` split; groups are whole tasks, never rows.

    With fewer than two groups there is nothing to hold out, so everything goes to train
    and the caller gets an empty validation list.
    """
    if not examples:
        return [], []
    if val_fraction <= 0:
        return list(examples), []

    groups = sorted({group_key(example) for example in examples})
    if len(groups) < 2:
        return list(examples), []

    rng = random.Random(seed)
    rng.shuffle(groups)
    n_val = max(1, round(len(groups) * val_fraction))
    n_val = min(n_val, len(groups) - 1)  # always keep at least one training group
    held_out = set(groups[:n_val])

    train = [ex for ex in examples if group_key(ex) not in held_out]
    val = [ex for ex in examples if group_key(ex) in held_out]
    return train, val


def to_swift_example(
    example: dict[str, Any],
    *,
    image_root: str | Path | None = None,
    examples_path: str | Path | None = None,
    with_answer: bool = False,
) -> dict[str, Any]:
    """One VQG example -> one ms-swift row (``messages`` + ``images``).

    The reference question is the assistant turn. With ``with_answer`` a second exchange
    is appended (ask -> answer), which trains the model to answer its own question too;
    keep it off when the fine-tuned model is evaluated as a pure question generator.
    """
    image = resolve_image_path(
        example.get("image"), image_root=image_root, examples_path=examples_path
    )
    prompt = str(example.get("input") or "").strip()
    target = str(example.get("target") or "").strip()

    messages: list[dict[str, str]] = [
        {"role": "user", "content": f"{IMAGE_PLACEHOLDER}{prompt}"},
        {"role": "assistant", "content": target},
    ]
    answer = str(example.get("answer") or "").strip()
    if with_answer and answer and target:
        messages.append({"role": "user", "content": ANSWER_PROMPT})
        messages.append({"role": "assistant", "content": answer})

    return {"messages": messages, "images": [str(image)] if image else []}


def to_swift_examples(
    examples: list[dict[str, Any]],
    *,
    image_root: str | Path | None = None,
    examples_path: str | Path | None = None,
    with_answer: bool = False,
) -> list[dict[str, Any]]:
    """Convert a list of VQG examples (see :func:`to_swift_example`)."""
    return [
        to_swift_example(
            example,
            image_root=image_root,
            examples_path=examples_path,
            with_answer=with_answer,
        )
        for example in examples
    ]


def dataset_stats(
    train: list[dict[str, Any]], val: list[dict[str, Any]], *, missing: list[str]
) -> dict[str, Any]:
    """Counts that belong next to the data (and in the thesis appendix)."""

    def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
        categories = Counter(category_of(ex) for ex in rows)
        return {
            "examples": len(rows),
            "screenshots": len({str(ex.get("image")) for ex in rows}),
            "tasks": len({group_key(ex) for ex in rows}),
            "categories": {name: categories.get(name, 0) for name in ALL_CATEGORIES},
        }

    return {
        "train": summarise(train),
        "val": summarise(val),
        "missing_images": len(missing),
        "missing_examples": missing[:20],
    }


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def write_jsonl(rows: list[dict[str, Any]], path: str | Path) -> Path:
    """Write ``rows`` as JSONL (UTF-8) and return the path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return out


def load_or_build_vqg(
    split: str,
    *,
    categories: list[str] | tuple[str, ...] | None = None,
    with_task: bool = True,
    limit: int | None = None,
    path: str | Path | None = None,
) -> tuple[list[dict[str, Any]], Path]:
    """Read the VQG JSONL for ``split``, building it first if it does not exist."""
    jsonl = Path(path) if path else PROCESSED_DIR / "vqg" / split / "examples.jsonl"
    if jsonl.exists():
        with open(jsonl, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        if categories:
            wanted = set(categories)
            rows = [row for row in rows if category_of(row) in wanted]
        return rows, jsonl

    exported = load_exported(split, limit=limit)
    examples = build_examples(exported, categories=categories, with_task=with_task)
    write_examples(examples, jsonl.parent)
    return examples, jsonl


def prepare_split(
    split: str = "train",
    *,
    out_dir: str | Path | None = None,
    categories: list[str] | tuple[str, ...] | None = None,
    with_task: bool = True,
    with_answer: bool = False,
    val_fraction: float = 0.05,
    max_per_category: int | None = None,
    seed: int = 42,
    image_root: str | Path | None = None,
    examples_path: str | Path | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Build ``train.jsonl`` / ``val.jsonl`` / ``stats.json`` for one split.

    Returns ``{"train": Path, "val": Path | None, "stats_path": Path, "stats": dict}``.
    """
    examples, source = load_or_build_vqg(
        split,
        categories=categories,
        with_task=with_task,
        limit=limit,
        path=examples_path,
    )
    kept, missing = filter_missing_images(examples, image_root=image_root)
    if missing:
        print(f"[swift] WARNING: {len(missing)} example(s) dropped, image not found")
        print(f"[swift]          e.g. {missing[0]}")

    balanced = sample_per_category(kept, max_per_category, seed=seed)
    train, val = split_examples(balanced, val_fraction=val_fraction, seed=seed)

    dest = Path(out_dir) if out_dir else PROCESSED_DIR / "vqg" / split / "swift"
    total = len(balanced)

    train_rows = to_swift_examples(
        train,
        image_root=image_root,
        examples_path=source,
        with_answer=with_answer,
    )
    val_rows = to_swift_examples(
        val,
        image_root=image_root,
        examples_path=source,
        with_answer=with_answer,
    )

    train_path = write_jsonl(train_rows, dest / "train.jsonl")
    val_path = write_jsonl(val_rows, dest / "val.jsonl") if val_rows else None

    stats = dataset_stats(train, val, missing=missing)
    stats.update(
        {
            "split": split,
            "source": str(source),
            "val_fraction": val_fraction if total else 0.0,
            "max_per_category": max_per_category,
            "with_answer": with_answer,
            "seed": seed,
        }
    )
    stats_path = dest / "stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    return {"train": train_path, "val": val_path, "stats_path": stats_path, "stats": stats}


def print_summary(result: dict[str, Any]) -> None:
    """Console summary shared by ``m2w-swift-data`` and ``m2w-train``."""
    stats = result["stats"]
    for side in ("train", "val"):
        row = stats[side]
        print(
            f"[swift] {side:5s} {row['examples']:6d} example(s) "
            f"{row['screenshots']:5d} screenshot(s) {row['tasks']:4d} task(s)"
        )
        counts = ", ".join(f"{name}={count}" for name, count in row["categories"].items())
        print(f"[swift]        {counts}")
    print(f"[swift] train -> {result['train']}")
    if result["val"]:
        print(f"[swift] val   -> {result['val']}")
    print(f"[swift] stats -> {result['stats_path']}")


def main(argv: list[str] | None = None) -> int:
    ensure_utf8()
    parser = argparse.ArgumentParser(
        description="Convert the VQG set to ms-swift's multimodal SFT JSONL."
    )
    parser.add_argument("--split", default="train", choices=SPLITS)
    parser.add_argument("--examples", type=Path, default=None, help="input VQG examples.jsonl")
    parser.add_argument(
        "--out", type=Path, default=None, help="default: data/processed/vqg/<split>/swift"
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        choices=list(ALL_CATEGORIES),
        help=f"subset of {list(ALL_CATEGORIES)} (default: {list(SCREENSHOT_GROUNDED)})",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help="look screenshots up as <image-root>/<basename> (moving data between machines)",
    )
    parser.add_argument("--val-fraction", type=float, default=0.05, help="0 disables val")
    parser.add_argument(
        "--max-per-category", type=int, default=None, help="cap each category (EDA budget)"
    )
    parser.add_argument("--with-answer", action="store_true", help="add a second Q/A turn")
    parser.add_argument("--no-task", action="store_true", help="omit the user task from the prompt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="only when building from scratch")
    args = parser.parse_args(argv)

    result = prepare_split(
        args.split,
        out_dir=args.out,
        categories=args.categories,
        with_task=not args.no_task,
        with_answer=args.with_answer,
        val_fraction=args.val_fraction,
        max_per_category=args.max_per_category,
        seed=args.seed,
        image_root=args.image_root,
        examples_path=args.examples,
        limit=args.limit,
    )
    print_summary(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
