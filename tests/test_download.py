"""Tests for the Mind2Web JSON download helpers (no network)."""

import zipfile
from pathlib import Path

from mind2web_vqg.data.download import _normalise_split_layout, _unzip_test


def _make_zip(path: Path, names: list[str]) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for name in names:
            zf.writestr(name, "[]")
    return path


def test_unzip_moves_the_test_splits_under_data(tmp_path: Path) -> None:
    """`test.zip` has the splits at its root; the loaders expect them under `data/`."""
    archive = _make_zip(
        tmp_path / "test.zip",
        ["test_task/a.json", "test_website/b.json", "test_domain/c.json"],
    )
    raw = tmp_path / "raw"

    _unzip_test(archive, raw)

    for split in ("test_task", "test_website", "test_domain"):
        assert (raw / "data" / split).is_dir()
        assert not (raw / split).exists()


def test_unzip_keeps_an_archive_that_already_has_the_data_prefix(tmp_path: Path) -> None:
    archive = _make_zip(tmp_path / "test.zip", ["data/test_task/a.json"])
    raw = tmp_path / "raw"

    _unzip_test(archive, raw)

    assert (raw / "data" / "test_task" / "a.json").is_file()
    assert not (raw / "test_task").exists()


def test_normalise_leaves_the_train_split_alone(tmp_path: Path) -> None:
    (tmp_path / "data" / "train").mkdir(parents=True)
    (tmp_path / "test_task").mkdir()

    _normalise_split_layout(tmp_path)

    assert (tmp_path / "data" / "train").is_dir()
    assert (tmp_path / "data" / "test_task").is_dir()


def test_normalise_is_idempotent(tmp_path: Path) -> None:
    (tmp_path / "test_task").mkdir()
    assert len(_normalise_split_layout(tmp_path)) == 1
    assert _normalise_split_layout(tmp_path) == []


def test_normalise_never_overwrites_an_existing_split(tmp_path: Path) -> None:
    (tmp_path / "data" / "test_task").mkdir(parents=True)
    (tmp_path / "data" / "test_task" / "kept.json").write_text("[]", encoding="utf-8")
    (tmp_path / "test_task").mkdir()
    (tmp_path / "test_task" / "loose.json").write_text("[]", encoding="utf-8")

    assert _normalise_split_layout(tmp_path) == []

    assert (tmp_path / "data" / "test_task" / "kept.json").is_file()
    assert (tmp_path / "test_task" / "loose.json").is_file()


def test_normalise_creates_the_data_directory_when_missing(tmp_path: Path) -> None:
    (tmp_path / "test_domain").mkdir()

    _normalise_split_layout(tmp_path)

    assert (tmp_path / "data" / "test_domain").is_dir()
