"""Tests for the Mind2Web JSON download helpers (no network)."""

import zipfile
from pathlib import Path

import pytest

from mind2web_vqg.data.download import _normalise_split_layout, _unzip_test, verify_layout
from mind2web_vqg.data.load import iter_split

TRAIN_ROWS = [{"annotation_id": "a"}, {"annotation_id": "b"}]


def _make_zip(path: Path, names: list[str]) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for name in names:
            zf.writestr(name, "[]")
    return path


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Split layout
# --------------------------------------------------------------------------- #
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


def test_unzip_explains_a_corrupt_archive(tmp_path: Path) -> None:
    archive = tmp_path / "test.zip"
    archive.write_bytes(b"not a zip at all")

    with pytest.raises(RuntimeError) as error:
        _unzip_test(archive, tmp_path / "raw")

    assert "corrupt" in str(error.value)
    assert "--force" in str(error.value)


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


def test_normalise_repairs_an_existing_split(tmp_path: Path) -> None:
    """Re-extracting is the repair path: the fresh file must replace the broken copy."""
    _write(tmp_path / "data" / "test_task" / "broken.json", "[{")
    _write(tmp_path / "data" / "test_task" / "other.json", "[]")
    _write(tmp_path / "test_task" / "broken.json", "[]")
    _write(tmp_path / "test_task" / "new.json", "[]")

    assert _normalise_split_layout(tmp_path) == [tmp_path / "data" / "test_task"]

    assert (tmp_path / "data" / "test_task" / "broken.json").read_text() == "[]"
    assert (tmp_path / "data" / "test_task" / "new.json").is_file()
    assert (tmp_path / "data" / "test_task" / "other.json").is_file()
    assert not (tmp_path / "test_task").exists()


def test_normalise_creates_the_data_directory_when_missing(tmp_path: Path) -> None:
    (tmp_path / "test_domain").mkdir()

    _normalise_split_layout(tmp_path)

    assert (tmp_path / "data" / "test_domain").is_dir()


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
def test_verify_accepts_a_healthy_layout(tmp_path: Path, capsys) -> None:
    _write(tmp_path / "data" / "train" / "train_0.json", '[{"a": 1}, {"a": 2}]')
    _write(tmp_path / "data" / "test_task" / "task_0.json", "[]")

    assert verify_layout(tmp_path) == 0
    out = capsys.readouterr().out
    assert "train" in out and "2 row(s)" in out
    assert "0 broken file(s)" in out
    assert "2 missing split(s)" in out  # the two test splits are simply not downloaded


def test_verify_names_a_truncated_file(tmp_path: Path, capsys) -> None:
    _write(tmp_path / "data" / "train" / "train_0.json", '[{"clean'[:8])

    assert verify_layout(tmp_path) == 1
    out = capsys.readouterr().out
    assert "BROKEN train_0.json" in out
    assert "1 broken file(s)" in out


def test_verify_reports_a_missing_split_without_failing(tmp_path: Path, capsys) -> None:
    _write(tmp_path / "data" / "train" / "train_0.json", "[]")

    assert verify_layout(tmp_path) == 0
    assert "MISSING" in capsys.readouterr().out


def test_verify_counts_a_wrapped_instance_list(tmp_path: Path, capsys) -> None:
    _write(tmp_path / "data" / "train" / "train_0.json", '{"data": [{}, {}, {}]}')

    assert verify_layout(tmp_path) == 0
    assert "3 row(s)" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def test_iter_split_reads_instances(tmp_path: Path) -> None:
    _write(tmp_path / "train" / "train_0.json", '[{"id": 1}, {"id": 2}]')

    assert [row["id"] for row in iter_split("train", tmp_path)] == [1, 2]


def test_iter_split_names_a_truncated_file(tmp_path: Path) -> None:
    _write(tmp_path / "train" / "train_0.json", '[{"annotation_id": "x"')

    with pytest.raises(ValueError) as error:
        list(iter_split("train", tmp_path))

    message = str(error.value)
    assert "train_0.json" in message
    assert "m2w-download --verify" in message


def test_iter_split_explains_a_missing_split(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError) as error:
        list(iter_split("test_task", tmp_path))

    assert "m2w-download" in str(error.value)
