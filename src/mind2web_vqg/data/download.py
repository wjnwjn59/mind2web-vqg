"""Download Mind2Web (train + test) into ``RAW_DIR`` (default ``data/raw``).

Source: HuggingFace dataset ``osunlp/Mind2Web``.
- Train: ``data/train/train_*.json``
- Test : ``test.zip`` (password ``mind2web``) -> test_task/test_website/test_domain

Usage::

    uv run m2w-download            # download everything
    uv run m2w-download --skip-test
    uv run m2w-download --skip-train

On the lab server, point it at the shared datasets directory (see README section 7)::

    export MIND2WEB_DATA_DIR=/mnt/data/datasets/Mind2Web
    uv run m2w-download            # -> /mnt/data/datasets/Mind2Web/raw

NOTE: data under ``data/`` is gitignored -> do NOT commit it to GitHub.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

from huggingface_hub import snapshot_download

from mind2web_vqg.cli import ensure_utf8
from mind2web_vqg.paths import RAW_DIR, ensure_dirs

REPO_ID = "osunlp/Mind2Web"
TEST_ZIP_NAME = "test.zip"
TEST_ZIP_PASSWORD = b"mind2web"


def _unzip_test(zip_path: Path, dest: Path) -> None:
    print(f"[download] extracting {zip_path.name} (password: mind2web) ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(path=dest, pwd=TEST_ZIP_PASSWORD)
    print(f"[download] done -> {dest}")


def download(
    dest: Path | None = None,
    *,
    skip_train: bool = False,
    skip_test: bool = False,
    force: bool = False,
) -> Path:
    """Download Mind2Web into ``dest`` (default ``RAW_DIR``)."""
    ensure_dirs()
    dest = dest or RAW_DIR
    dest.mkdir(parents=True, exist_ok=True)

    if not skip_train:
        print(f"[download] downloading train split from {REPO_ID} ...")
        # allow_patterns limits the download to the train split only
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            local_dir=str(dest),
            allow_patterns=["data/train/*.json"],
            force_download=force,
        )

    if not skip_test:
        print(f"[download] downloading test split ({TEST_ZIP_NAME}) from {REPO_ID} ...")
        path = snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            local_dir=str(dest),
            allow_patterns=[TEST_ZIP_NAME],
            force_download=force,
        )
        zip_path = Path(path) / TEST_ZIP_NAME
        if zip_path.exists():
            _unzip_test(zip_path, dest)

    print(f"[download] finished. Data at: {dest / 'data'}")
    return dest


def main(argv: list[str] | None = None) -> int:
    ensure_utf8()
    parser = argparse.ArgumentParser(description="Download the Mind2Web dataset.")
    parser.add_argument(
        "--dest", type=Path, default=None, help="target directory (default $MIND2WEB_DATA_DIR/raw)"
    )
    parser.add_argument("--skip-train", action="store_true", help="do not download the train split")
    parser.add_argument(
        "--skip-test", action="store_true", help="do not download/extract the test split"
    )
    parser.add_argument(
        "--force", action="store_true", help="re-download even if the files already exist"
    )
    args = parser.parse_args(argv)

    download(
        args.dest,
        skip_train=args.skip_train,
        skip_test=args.skip_test,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
