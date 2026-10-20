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
import json
import shutil
import sys
import zipfile
from pathlib import Path

from huggingface_hub import snapshot_download

from mind2web_vqg.cli import ensure_utf8
from mind2web_vqg.data.load import SPLITS, available_splits
from mind2web_vqg.paths import RAW_DIR, ensure_dirs

REPO_ID = "osunlp/Mind2Web"
TEST_ZIP_NAME = "test.zip"
TEST_ZIP_PASSWORD = b"mind2web"


def _normalise_split_layout(dest: Path) -> list[Path]:
    """Merge every loose split directory into ``dest/data/<split>`` — the layout loaders read.

    ``test.zip`` stores ``test_task/`` at its own root, whereas the train split arrives as
    ``data/train/`` (that is its path inside the HF repo), so extracting both into ``dest``
    leaves the splits in two different roots and ``available_splits()`` only sees train.

    Same-named files are **replaced**: re-extracting the archive is the repair path for a
    truncated file, so keeping the old copy would silently preserve the corruption. Idempotent:
    a loose directory that does not exist is simply skipped.
    """
    data_dir = dest / "data"
    moved: list[Path] = []
    for split in SPLITS:
        loose = dest / split
        if not loose.is_dir():
            continue

        target = data_dir / split
        target.mkdir(parents=True, exist_ok=True)
        for path in sorted(loose.rglob("*")):
            if not path.is_file():
                continue
            destination = target / path.relative_to(loose)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.unlink(missing_ok=True)
            shutil.move(str(path), str(destination))

        shutil.rmtree(loose, ignore_errors=True)
        moved.append(target)
    return moved


def _unzip_test(zip_path: Path, dest: Path) -> None:
    print(f"[download] extracting {zip_path.name} (password: mind2web) ...")
    try:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(path=dest, pwd=TEST_ZIP_PASSWORD)
    except zipfile.BadZipFile as error:
        raise RuntimeError(
            f"{zip_path} is corrupt ({error}). Delete that file and re-run "
            "`uv run m2w-download --skip-train --force`."
        ) from error
    _normalise_split_layout(dest)
    print(f"[download] done -> {dest / 'data'}")


def verify_layout(dest: Path | None = None) -> int:
    """Parse every JSON file under ``dest/data`` and report rows per split.

    A truncated JSON file is the classic symptom of an interrupted extraction, and the
    traceback from a deep loader only names a line and column. This check names the file,
    so a broken corpus is found in seconds instead of after a long EDA pass.

    Returns 0 when every present file parses, 1 when any file is broken. A split that was
    never downloaded is reported but is not an error: ``m2w-download --skip-test`` is a
    legitimate state.
    """
    dest = dest or RAW_DIR
    root = dest / "data"
    print(f"[download] verifying {root}")

    problems = 0
    missing = 0
    total = 0
    for split in SPLITS:
        directory = root / split
        if not directory.is_dir():
            print(f"  {split:14s} MISSING ({directory})")
            missing += 1
            continue

        files = sorted(directory.glob("*.json"))
        rows = 0
        broken = 0
        for path in files:
            try:
                with open(path, encoding="utf-8-sig") as handle:
                    data = json.load(handle)
            except (OSError, json.JSONDecodeError) as error:
                size = path.stat().st_size if path.exists() else 0
                print(f"  {split:14s} BROKEN {path.name} ({size:,} bytes): {error}")
                broken += 1
                continue
            if isinstance(data, dict):
                data = data.get("data", data.get("instances", []))
            rows += len(data)

        total += rows
        problems += broken
        state = "ok" if broken == 0 else "BROKEN"
        print(f"  {split:14s} {len(files):3d} file(s) {rows:6d} row(s)  {state}")

    print(
        f"[download] {total:,} row(s), {problems} broken file(s), {missing} missing split(s)"
    )
    return 0 if problems == 0 else 1


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
    found = available_splits(dest / "data")
    print(f"[download] splits available: {', '.join(found) or 'none'}")
    leftovers = [split for split in SPLITS if (dest / split).is_dir()]
    if leftovers:
        print(
            f"[download] warning: {', '.join(leftovers)} also exists directly under {dest}; "
            f"the loaders read only {dest / 'data'}"
        )
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
    parser.add_argument(
        "--verify",
        action="store_true",
        help="no download: parse every JSON file and report rows + broken files",
    )
    args = parser.parse_args(argv)

    if args.verify:
        return verify_layout(args.dest)

    download(
        args.dest,
        skip_train=args.skip_train,
        skip_test=args.skip_test,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
