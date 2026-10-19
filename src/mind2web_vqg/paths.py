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
