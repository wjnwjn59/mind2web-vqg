"""Read Multimodal-Mind2Web parquet rows and export screenshots to disk.

The dataset is **action-level**: every row is one action of one Mind2Web task,
and carries the full-page screenshot captured at the moment of that action.

Row fields used here::

    annotation_id        task id (joins to the JSON dataset)
    action_uid           action id (joins to actions[i].action_uid)
    screenshot           {"bytes": b"...", "path": None}  (PIL-decodable)
    confirmed_task       natural-language task
    action_reprs         list[str]  the action history, e.g. ['CLICK [Search]']
    target_action_index  index of this action inside action_reprs
    target_action_reprs  str        the action to predict
    operation            JSON str   {"op": "CLICK", "value": "", ...}
    pos_candidates       JSON str[] candidate elements, each with attributes

Usage::

    uv run m2w-mm-export --split test_task --limit 5
    uv run m2w-mm-export --split test_task --limit 5 --max-width 1600 --force

Output::

    data/processed/multimodal/test_task/
    ├── images/0f1c..._00.jpg
    └── examples.jsonl
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from mind2web_vqg.cli import ensure_utf8
from mind2web_vqg.paths import MULTIMODAL_DIR, PROCESSED_DIR

#: Split name -> glob used by the parquet files (``data/<split>-*.parquet``).
SPLITS = ("train", "test_task", "test_website", "test_domain")


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested, no pyarrow/PIL needed)
# --------------------------------------------------------------------------- #
def parquet_files(split: str, root: str | Path | None = None) -> list[Path]:
    """Return the parquet files of ``split``, sorted."""
    if split not in SPLITS:
        raise ValueError(f"invalid split: {split!r}. Choose one of {list(SPLITS)}")
    data_dir = Path(root or MULTIMODAL_DIR) / "data"
    files = sorted(data_dir.glob(f"{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"no parquet files for {split!r} in {data_dir}. "
            "Run `uv run m2w-download-mm --splits " + split + "` first."
        )
    return files


def normalize_limit(limit: int | None) -> int | None:
    """``0`` (or negative) means "no limit", matching the ``m2w-eda --limit 0`` convention."""
    return None if limit is not None and limit <= 0 else limit


def as_json(value: Any, default: Any = None) -> Any:
    """The dataset stores nested fields as JSON *strings*; decode them tolerantly."""
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


def parse_bbox(rect: str | None) -> list[float] | None:
    """``"x,y,w,h"`` -> ``[x, y, w, h]`` (floats), or ``None``."""
    if not rect:
        return None
    try:
        parts = [float(p) for p in str(rect).split(",")]
    except ValueError:
        return None
    return parts if len(parts) == 4 else None


def target_element(pos_candidates: list[Any]) -> dict[str, Any]:
    """Pick the original target from ``pos_candidates`` (falls back to the first)."""
    candidates = [c for c in (as_json(c, {}) for c in pos_candidates or []) if isinstance(c, dict)]
    for cand in candidates:
        if cand.get("is_original_target"):
            return cand
    return candidates[0] if candidates else {}


def clean_task(value: Any) -> str:
    """``confirmed_task`` sometimes ships with a stray surrounding quote; strip it."""
    return str(value or "").strip().strip('"').strip()


def image_name(split: str, annotation_id: str, action_index: int) -> str:
    """Filesystem-safe image file name (no directories, no spaces)."""
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(annotation_id))
    return f"{split}_{safe}_{int(action_index):02d}.jpg"


def step_index(row: dict[str, Any], action_reprs: list[Any]) -> int:
    """0-based position of this action inside the task's ``action_reprs``.

    ``target_action_index`` is shipped by the dataset. Falling back to a text match is
    only a safety net: if the index is unknown we return 0, because putting *future*
    actions into the prompt would leak the rest of the trajectory.
    """
    for raw in (row.get("target_action_index"), row.get("row_index")):
        try:
            return max(0, int(raw))
        except (TypeError, ValueError):
            continue
    target = row.get("target_action_reprs")
    if target in action_reprs:
        return action_reprs.index(target)
    return 0


def from_row(row: dict[str, Any], *, index: int, split: str) -> dict[str, Any]:
    """Convert one parquet row into a plain example dict (no image decoding)."""
    annotation_id = str(row.get("annotation_id") or "unknown")
    operation = as_json(row.get("operation"), {}) or {}
    target = target_element(row.get("pos_candidates") or [])
    attrs = as_json(target.get("attributes"), {}) or {}
    action_reprs = list(row.get("action_reprs") or [])
    # ``action_history`` is *completed* steps, so it stops before the current action.
    # The previous version stored the whole trajectory, which leaked future actions into
    # every prompt and made the L3 "how many steps" answer constant per task.
    completed = action_reprs[: step_index(row, action_reprs)]

    return {
        "id": f"{annotation_id}:{index}",
        "annotation_id": annotation_id,
        "action_uid": row.get("action_uid"),
        "split": split,
        "row_index": index,
        "image_name": image_name(split, annotation_id, index),
        "task": clean_task(row.get("confirmed_task")),
        "website": row.get("website"),
        "domain": row.get("domain"),
        "subdomain": row.get("subdomain"),
        "action_history": completed,
        "target_action": row.get("target_action_reprs") or "",
        "op": operation.get("op"),
        "value": operation.get("value") or "",
        "target_tag": target.get("tag"),
        "target_type": attrs.get("type"),
        "target_role": attrs.get("role"),
        "target_text": attrs.get("text") or attrs.get("aria_label") or attrs.get("title") or "",
        "target_bbox": parse_bbox(attrs.get("bounding_box_rect")),
    }


def fit_width(image: Any, max_width: int) -> Any:
    """Downscale a full-page screenshot so its width is at most ``max_width`` px."""
    if image.width <= max_width:
        return image
    from PIL import Image

    resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS", 1)
    height = round(image.height * max_width / image.width)
    return image.resize((max_width, height), resample)


# --------------------------------------------------------------------------- #
# Parquet / image I/O (lazy imports)
# --------------------------------------------------------------------------- #
def iter_rows(
    split: str,
    *,
    root: str | Path | None = None,
    limit: int | None = None,
    batch_size: int = 4,
) -> Iterator[dict[str, Any]]:
    """Yield rows one at a time, stopping after ``limit`` (keeps memory flat)."""
    import pyarrow.parquet as pq

    seen = 0
    for path in parquet_files(split, root):
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=batch_size):
            for row in batch.to_pylist():
                yield row
                seen += 1
                if limit is not None and seen >= limit:
                    return


def decode_image(value: Any) -> Any:
    """Decode the ``screenshot`` column into a PIL image.

    Depending on the reader, the column is either a PIL image (``datasets``),
    a ``{"bytes": ...}`` / ``{"path": ...}`` dict (pyarrow), or raw bytes.
    """
    from PIL import Image

    # Full-page screenshots can exceed PIL's decompression-bomb threshold (89M px) while
    # still being legitimate Mind2Web captures (a long product list easily reaches it).
    # The images come from a trusted dataset, so the guard is lifted here, not globally.
    Image.MAX_IMAGE_PIXELS = None

    if value is None:
        raise ValueError("screenshot column is None")
    if hasattr(value, "resize") and hasattr(value, "convert"):  # already a PIL image
        return value
    if isinstance(value, dict):
        if value.get("bytes"):
            return Image.open(io.BytesIO(value["bytes"]))
        if value.get("path"):
            return Image.open(value["path"])
    if isinstance(value, (bytes, bytearray)):
        return Image.open(io.BytesIO(value))
    raise ValueError(f"cannot decode screenshot of type {type(value)!r}")


def load_screenshot(
    split: str,
    annotation_id: str,
    action_uid: str,
    *,
    root: str | Path | None = None,
) -> Any | None:
    """Find one screenshot by ``(annotation_id, action_uid)``; ``None`` if missing.

    Uses parquet predicate pushdown so only the matching row group is read.
    """
    import pyarrow.parquet as pq

    for path in parquet_files(split, root):
        table = pq.read_table(
            path,
            columns=["annotation_id", "action_uid", "screenshot"],
            filters=[("annotation_id", "=", annotation_id)],
        )
        for row in table.to_pylist():
            if row.get("action_uid") == action_uid:
                return decode_image(row["screenshot"])
    return None


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
def export_samples(
    split: str = "test_task",
    *,
    limit: int | None = 5,
    root: str | Path | None = None,
    dest: str | Path | None = None,
    max_width: int = 1280,
    quality: int = 82,
    force: bool = False,
) -> list[dict[str, Any]]:
    """Export ``limit`` rows of ``split`` as JPEGs + ``examples.jsonl``.

    ``limit=0`` means "the whole split" (the convention used by ``m2w-eda``).
    """
    limit = normalize_limit(limit)
    out_dir = Path(dest) if dest else PROCESSED_DIR / "multimodal" / split
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Image sizes must survive a resumed export (``--force`` is expensive), and the raw
    # size cannot be recovered from the downscaled JPEG. A tiny sidecar keeps both.
    meta_path = out_dir / "image_meta.json"
    image_meta: dict[str, dict[str, list[int]]] = {}
    if meta_path.is_file():
        try:
            image_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            image_meta = {}

    examples: list[dict[str, Any]] = []
    skipped = 0
    for index, row in enumerate(iter_rows(split, root=root, limit=limit)):
        example = from_row(row, index=index, split=split)
        path = images_dir / example["image_name"]

        if force or not path.exists():
            try:
                image = decode_image(row["screenshot"]).convert("RGB")
            except (ValueError, OSError, TypeError):
                # A single broken/absent screenshot must not abort a multi-thousand-row
                # split. Drop the row here, where it is cheap, and report the count.
                skipped += 1
                continue
            # ``source_size`` is the raw screenshot size; keep it so downstream tasks can
            # map DOM coordinates (``target_bbox``) onto the (possibly downscaled) JPEG.
            example["source_size"] = [image.width, image.height]
            image = fit_width(image, max_width)
            image.save(path, format="JPEG", quality=quality, optimize=True)
            example["image_size"] = [image.width, image.height]
            image_meta[example["image_name"]] = {
                "source_size": example["source_size"],
                "image_size": example["image_size"],
            }
        elif example["image_name"] in image_meta:
            example.update(image_meta[example["image_name"]])
        else:
            # Older exports have no sidecar: recover the JPEG size; treat it as the
            # source size so region questions still use a consistent coordinate space.
            from PIL import Image

            with Image.open(path) as saved:
                example["image_size"] = [saved.width, saved.height]
            example["source_size"] = example["image_size"]

        example["image"] = str(path)
        examples.append(example)

    if image_meta:
        meta_path.write_text(json.dumps(image_meta, indent=0), encoding="utf-8")

    with open(out_dir / "examples.jsonl", "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    if skipped:
        print(f"[mm] skipped {skipped} row(s) with a missing/undecodable screenshot")
    print(f"[mm] {len(examples)} example(s) -> {out_dir}")
    return examples


def main(argv: list[str] | None = None) -> int:
    ensure_utf8()
    parser = argparse.ArgumentParser(
        description="Export Multimodal-Mind2Web screenshots + metadata to disk."
    )
    parser.add_argument("--split", default="test_task", choices=SPLITS)
    parser.add_argument(
        "--limit", type=int, default=5, help="number of rows (default 5; 0 = the whole split)"
    )
    parser.add_argument(
        "--dest", type=Path, default=None, help="default: data/processed/multimodal/<split>"
    )
    parser.add_argument(
        "--root", type=Path, default=None, help="parquet root (default $MIND2WEB_MM_DIR)"
    )
    parser.add_argument(
        "--max-width", type=int, default=1280, help="downscale width (default 1280)"
    )
    parser.add_argument("--quality", type=int, default=82, help="JPEG quality (default 82)")
    parser.add_argument(
        "--force", action="store_true", help="re-encode images that already exist"
    )
    args = parser.parse_args(argv)

    export_samples(
        args.split,
        limit=args.limit,
        root=args.root,
        dest=args.dest,
        max_width=args.max_width,
        quality=args.quality,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
