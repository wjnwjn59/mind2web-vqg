"""Load the Mind2Web splits from the downloaded JSON files.

Data layout (after download, see ``README.md``)::

    data/raw/data/
    ├── train/train_*.json
    ├── test_task/test_task_*.json
    ├── test_website/test_website_*.json
    └── test_domain/test_domain_*.json

Each JSON file is a list of instances; see the README for the field descriptions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mind2web_vqg.paths import RAW_DIR

#: Split name -> subdirectory under the data root
SPLITS: dict[str, str] = {
    "train": "train",
    "test_task": "test_task",
    "test_website": "test_website",
    "test_domain": "test_domain",
}

DEFAULT_ROOT = RAW_DIR / "data"


def _iter_json_files(split_dir: Path) -> list[Path]:
    return sorted(split_dir.glob("*.json"))


def load_split(split: str, root: str | Path = DEFAULT_ROOT) -> list[dict[str, Any]]:
    """Load one split as a list of instances.

    Args:
        split: one of ``SPLITS`` (e.g. ``"test_task"``).
        root: directory containing ``train/``, ``test_task/``, ...

    Returns:
        A list of instances (each item is a dict following the Mind2Web schema).
    """
    if split not in SPLITS:
        raise ValueError(f"invalid split: {split!r}. Choose one of {list(SPLITS)}")

    split_dir = Path(root) / SPLITS[split]
    if not split_dir.is_dir():
        raise FileNotFoundError(
            f"{split_dir} not found. Run `uv run m2w-download` first (see README)."
        )

    instances: list[dict[str, Any]] = []
    for path in _iter_json_files(split_dir):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        # each file may be a list, or a dict wrapping a list
        if isinstance(data, dict):
            data = data.get("data", data.get("instances", []))
        instances.extend(data)
    return instances


def load_all_splits(root: str | Path = DEFAULT_ROOT) -> dict[str, list[dict[str, Any]]]:
    """Load every available split (skipping splits that were not downloaded)."""
    out: dict[str, list[dict[str, Any]]] = {}
    for split in SPLITS:
        try:
            out[split] = load_split(split, root)
        except FileNotFoundError:
            continue
    return out


if __name__ == "__main__":  # quick sanity check
    for name, items in load_all_splits().items():
        print(f"{name:14s} {len(items):5d} instances")
