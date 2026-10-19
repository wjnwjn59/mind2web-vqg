"""Download model weights following the lab server convention.

Convention (AIVN-VLAI): use the HF CLI, store under ``/mnt/data/hf-cache`` with
folder names ``<namespace>_<repo_name>`` (replace ``/`` with ``_``)::

    AIDC-AI/Ovis2.5-9B  ->  hf-cache/AIDC-AI_Ovis2.5-9B

Usage::

    uv run m2w-download-model google/flan-t5-base
    # -> $MIND2WEB_HF_CACHE/google_flan-t5-base (default ./hf-cache)

Equivalent hf CLI command::

    hf download google/flan-t5-base --local-dir /mnt/data/hf-cache/google_flan-t5-base
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

from mind2web_vqg.cli import ensure_utf8
from mind2web_vqg.paths import HF_CACHE_DIR


def flatten_repo_id(repo_id: str) -> str:
    """``<namespace>/<repo_name>`` -> ``<namespace>_<repo_name>``."""
    return repo_id.replace("/", "_")


def target_dir(repo_id: str, cache_dir: Path | None = None) -> Path:
    """Target directory per convention: ``<cache_dir>/<namespace>_<repo_name>``."""
    return (cache_dir or HF_CACHE_DIR) / flatten_repo_id(repo_id)


def download_model(
    repo_id: str,
    *,
    cache_dir: Path | None = None,
    force: bool = False,
) -> Path:
    """Download the weights of ``repo_id`` into ``target_dir(repo_id)`` and return the folder."""
    dest = target_dir(repo_id, cache_dir)
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[model] downloading {repo_id} -> {dest}")
    path = snapshot_download(
        repo_id=repo_id,
        local_dir=str(dest),
        force_download=force,
    )
    print(f"[model] done -> {path}")
    return Path(path)


def main(argv: list[str] | None = None) -> int:
    ensure_utf8()
    parser = argparse.ArgumentParser(
        description="Download model weights following the hf-cache convention."
    )
    parser.add_argument("repo_id", help="HuggingFace repo, in <namespace>/<repo_name> form")
    parser.add_argument(
        "--cache-dir", type=Path, default=None, help="defaults to $MIND2WEB_HF_CACHE"
    )
    parser.add_argument(
        "--force", action="store_true", help="re-download even if the files already exist"
    )
    args = parser.parse_args(argv)

    download_model(args.repo_id, cache_dir=args.cache_dir, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
