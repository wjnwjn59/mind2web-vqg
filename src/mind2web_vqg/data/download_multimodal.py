"""Download Multimodal-Mind2Web (screenshots aligned with Mind2Web HTML).

Source: HuggingFace dataset ``osunlp/Multimodal-Mind2Web``.
Unlike ``osunlp/Mind2Web`` (task-level JSON), this dataset is **action-level parquet**
with the webpage screenshot embedded as an image column.

Lab convention (AIVN-VLAI): datasets live under ``/mnt/data/datasets`` grouped by
dataset name, so the default destination on the server is::

    /mnt/data/datasets/Multimodal-Mind2Web

Locally it falls back to ``./data/multimodal`` (gitignored).

Where does the default come from? ``MULTIMODAL_DIR`` = ``$MIND2WEB_MM_DIR`` if set,
otherwise ``$MIND2WEB_DATA_DIR/multimodal``. On the lab server ``MIND2WEB_DATA_DIR`` is
usually ``/mnt/data/datasets/Mind2Web``, which would nest the screenshots **inside** the
Mind2Web folder — the script warns about this. Set ``MIND2WEB_MM_DIR`` to keep the two
datasets side by side.

Usage::

    # 1) see what would be downloaded (no network writes, no data)
    uv run m2w-download-mm --dry-run

    # 2) start with the smallest split
    uv run m2w-download-mm --splits test_task

    # 3) add the rest later (resumable, existing files are skipped)
    uv run m2w-download-mm --splits test_website test_domain

    # 4) everything (train included)
    uv run m2w-download-mm --splits all

On the lab server::

    export MIND2WEB_MM_DIR=/mnt/data/datasets/Multimodal-Mind2Web
    uv run m2w-download-mm --splits test_task

NOTE: never pass ``--force`` unless you really want to re-download several GB.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

from mind2web_vqg.cli import ensure_utf8
from mind2web_vqg.data.multimodal import SPLITS
from mind2web_vqg.paths import MULTIMODAL_DIR

REPO_ID = "osunlp/Multimodal-Mind2Web"

#: Rough download size per split, in GB (from the Hub file listing).
SPLIT_SIZE_GB = {
    "train": 7.6,
    "test_task": 1.3,
    "test_website": 1.1,
    "test_domain": 3.6,
}


def patterns_for(splits: list[str]) -> list[str]:
    """Glob patterns selecting ``data/<split>-*.parquet`` for each split."""
    return [f"data/{split}-*.parquet" for split in splits]


def estimate(patterns: list[str], token: str | None) -> tuple[int, float]:
    """Return ``(num_files, total_gb)`` for the given patterns without downloading."""
    infos = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        allow_patterns=patterns,
        token=token,
        dry_run=True,
    )
    files = [info for info in infos if getattr(info, "will_download", True)]
    total = sum(info.file_size for info in infos)
    return len(files), total / 1e9


def _free_gb(path: Path) -> float:
    """Free space (GB) on the filesystem that will hold ``path``."""
    probe = path
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    return shutil.disk_usage(probe).free / 1e9


def dest_has_data(dest: Path) -> bool:
    """True when ``dest`` already holds downloaded parquet files (i.e. is in use)."""
    return any(dest.glob("data/*.parquet"))


def nested_dest_hint(dest: Path, env: dict[str, str], has_data: bool = False) -> str | None:
    """Warn when the default destination lands *inside* another dataset's folder.

    ``MULTIMODAL_DIR`` defaults to ``$MIND2WEB_DATA_DIR/multimodal``, so on the lab
    server (where ``MIND2WEB_DATA_DIR=/mnt/data/datasets/Mind2Web``) the screenshots
    end up nested inside the Mind2Web dataset folder instead of next to it.

    Nags only before the first file lands: pass ``has_data=True`` (see
    :func:`dest_has_data`) once the destination is populated, since by then the
    nesting is clearly intentional and the reminder is just noise on resume runs.
    """
    if env.get("MIND2WEB_MM_DIR") or has_data:
        return None
    data_dir = env.get("MIND2WEB_DATA_DIR")
    if not data_dir or Path(dest) != Path(data_dir) / "multimodal":
        return None
    return (
        f"MIND2WEB_MM_DIR is not set, so the dataset goes inside the Mind2Web folder ({dest}). "
        "The lab convention keeps datasets side by side; set "
        "MIND2WEB_MM_DIR=/mnt/data/datasets/Multimodal-Mind2Web to do the same. "
        "Ignore this if the nesting is intentional."
    )


def download(
    splits: list[str],
    *,
    dest: Path | None = None,
    force: bool = False,
    dry_run: bool = False,
    token: str | None = None,
) -> Path:
    """Download the selected splits into ``dest`` (default ``MULTIMODAL_DIR``)."""
    dest = dest or MULTIMODAL_DIR
    patterns = patterns_for(splits)

    print(f"[mm] repo      : {REPO_ID}")
    print(f"[mm] splits    : {', '.join(splits)}")
    print(f"[mm] destination: {dest}")

    hint = nested_dest_hint(dest, dict(os.environ), has_data=dest_has_data(dest))
    if hint:
        print(f"[mm] note: {hint}", file=sys.stderr)

    n_files, total_gb = estimate(patterns, token)
    print(f"[mm] would download {n_files} file(s), ~{total_gb:.1f} GB")

    free = _free_gb(dest)
    print(f"[mm] free space on destination: {free:.1f} GB")
    if free < total_gb * 1.15:
        print(
            f"[mm] WARNING: only {free:.1f} GB free but ~{total_gb:.1f} GB needed. "
            "Free some space or pick another --dest.",
            file=sys.stderr,
        )

    if dry_run:
        print("[mm] dry run: nothing downloaded.")
        return dest

    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        allow_patterns=patterns,
        local_dir=str(dest),
        force_download=force,
        token=token,
        max_workers=8,
    )
    print(f"[mm] done -> {dest}")
    return dest


def verify(dest: Path) -> int:
    """Report parquet files and row counts (reads metadata only, not images)."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("[mm] pyarrow not installed; skipping row counts.", file=sys.stderr)
        return 0

    files = sorted(dest.glob("data/*.parquet"))
    if not files:
        print(f"[mm] no parquet files under {dest / 'data'}", file=sys.stderr)
        return 0

    per_split: dict[str, list[Path]] = {}
    for path in files:
        split = path.name.split("-")[0]
        per_split.setdefault(split, []).append(path)

    print(f"[mm] layout under {dest / 'data'}")
    total_rows = 0
    for split in SPLITS:
        paths = per_split.get(split, [])
        if not paths:
            continue
        rows = 0
        has_image = False
        for path in paths:
            meta = pq.ParquetFile(path).metadata
            rows += meta.num_rows
            if not has_image:
                names = pq.ParquetFile(path).schema_arrow.names
                has_image = "screenshot" in names
        total_rows += rows
        flag = "screenshot=yes" if has_image else "screenshot=NO"
        print(f"  {split:14s} {len(paths):2d} file(s)  {rows:6d} rows  ({flag})")
    print(f"  {'total':14s} {len(files):2d} file(s)  {total_rows:6d} rows")
    return total_rows


def main(argv: list[str] | None = None) -> int:
    ensure_utf8()
    parser = argparse.ArgumentParser(
        description="Download Multimodal-Mind2Web (screenshots) following the lab layout."
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["test_task"],
        choices=[*SPLITS, "all"],
        help="splits to download (default: test_task)",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="target directory (default: $MIND2WEB_MM_DIR or ./data/multimodal)",
    )
    parser.add_argument(
        "--force", action="store_true", help="re-download even if files already exist"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="show file count/size, download nothing"
    )
    parser.add_argument(
        "--token", default=None, help="HF token (default: $HF_TOKEN); optional for public data"
    )
    parser.add_argument(
        "--no-verify", action="store_true", help="skip the row-count report afterwards"
    )
    args = parser.parse_args(argv)

    splits = list(SPLITS) if "all" in args.splits else args.splits
    token = args.token or os.environ.get("HF_TOKEN")

    dest = download(
        splits,
        dest=args.dest,
        force=args.force,
        dry_run=args.dry_run,
        token=token,
    )
    if not args.dry_run and not args.no_verify:
        verify(dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
