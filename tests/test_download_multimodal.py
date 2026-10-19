"""Tests for the Multimodal-Mind2Web download helpers (no network)."""

from pathlib import Path

from mind2web_vqg.data.download_multimodal import (
    dest_has_data,
    nested_dest_hint,
    patterns_for,
)

LAB_DATA = "/mnt/data/datasets/Mind2Web"


def test_patterns_for() -> None:
    assert patterns_for(["test_task"]) == ["data/test_task-*.parquet"]
    assert patterns_for(["train", "test_domain"]) == [
        "data/train-*.parquet",
        "data/test_domain-*.parquet",
    ]


def test_hint_when_default_nests_inside_mind2web() -> None:
    dest = Path(LAB_DATA) / "multimodal"
    hint = nested_dest_hint(dest, {"MIND2WEB_DATA_DIR": LAB_DATA})
    assert hint is not None
    assert "MIND2WEB_MM_DIR" in hint and "Multimodal-Mind2Web" in hint


def test_no_hint_once_destination_has_data() -> None:
    dest = Path(LAB_DATA) / "multimodal"
    env = {"MIND2WEB_DATA_DIR": LAB_DATA}
    assert nested_dest_hint(dest, env, has_data=True) is None


def test_dest_has_data_detects_parquet(tmp_path: Path) -> None:
    assert dest_has_data(tmp_path) is False
    (tmp_path / "data").mkdir()
    assert dest_has_data(tmp_path) is False
    (tmp_path / "data" / "test_task-00000-of-00005.parquet").touch()
    assert dest_has_data(tmp_path) is True


def test_no_hint_when_mm_dir_is_explicit() -> None:
    dest = Path(LAB_DATA) / "multimodal"
    env = {
        "MIND2WEB_DATA_DIR": LAB_DATA,
        "MIND2WEB_MM_DIR": "/mnt/data/datasets/Multimodal-Mind2Web",
    }
    assert nested_dest_hint(dest, env) is None


def test_no_hint_for_local_default() -> None:
    assert nested_dest_hint(Path("./data/multimodal"), {}) is None
    other = Path(LAB_DATA) / "something-else"
    assert nested_dest_hint(other, {"MIND2WEB_DATA_DIR": LAB_DATA}) is None
