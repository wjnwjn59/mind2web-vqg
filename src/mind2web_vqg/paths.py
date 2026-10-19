"""Project-wide paths, resolved from the repository root.

Import paths from here instead of hard-coding them, so relocating data only
requires changing one place (or setting the MIND2WEB_DATA_DIR env var).
"""

from __future__ import annotations

import os
from pathlib import Path

# src/mind2web_vqg/paths.py -> three levels up is the repository root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: Root data directory (override with MIND2WEB_DATA_DIR).
#: - Local : ./data
#: - Server: /mnt/data/datasets/Mind2Web  (datasets are grouped by name, see README section 7)
DATA_DIR = Path(os.environ.get("MIND2WEB_DATA_DIR", PROJECT_ROOT / "data"))

RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

#: Root of the Multimodal-Mind2Web parquet files (override with MIND2WEB_MM_DIR).
#: - Local : ./data/multimodal
#: - Server: /mnt/data/datasets/Multimodal-Mind2Web  (grouped by dataset name)
MULTIMODAL_DIR = Path(os.environ.get("MIND2WEB_MM_DIR", DATA_DIR / "multimodal"))

#: Where model weights are stored (override with MIND2WEB_HF_CACHE).
#: - Local : ./hf-cache
#: - Server: /mnt/data/hf-cache  (folder name <namespace>_<repo_name>)
HF_CACHE_DIR = Path(os.environ.get("MIND2WEB_HF_CACHE", PROJECT_ROOT / "hf-cache"))

CONFIG_DIR = PROJECT_ROOT / "configs"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PROMPT_DIR = Path(__file__).resolve().parent / "prompts"


def ensure_dirs() -> None:
    """Create the required directories if missing (idempotent)."""
    for path in (RAW_DIR, PROCESSED_DIR, OUTPUT_DIR):
        path.mkdir(parents=True, exist_ok=True)


def main() -> int:
    """Print the resolved paths — run this before a multi-GB download.

    ``uv run python -m mind2web_vqg.paths``

    The values below are the *same constants* the download/export/eval code imports, so
    this is exactly where data will be written and read from in this shell.
    """
    overrides = ("MIND2WEB_DATA_DIR", "MIND2WEB_MM_DIR", "MIND2WEB_HF_CACHE")
    print("env var -> resolved directory")
    for name in overrides:
        value = os.environ.get(name)
        state = f"set ({value})" if value else "unset -> default"
        print(f"  {name:20s} {state}")
    print("\ndirectory            state     path")
    for label, path in (
        ("raw", RAW_DIR),
        ("multimodal", MULTIMODAL_DIR),
        ("processed", PROCESSED_DIR),
        ("hf-cache", HF_CACHE_DIR),
    ):
        state = "exists" if path.exists() else "missing"
        print(f"  {label:16s} {state:9s} {path}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
