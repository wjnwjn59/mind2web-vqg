"""Tests for the model download helpers (no network)."""

from pathlib import Path

from mind2web_vqg.models.download import (
    flatten_repo_id,
    hf_cache_hint,
    target_dir,
)


def test_flatten_repo_id() -> None:
    assert flatten_repo_id("Qwen/Qwen3.5-9B") == "Qwen_Qwen3.5-9B"
    assert flatten_repo_id("google/flan-t5-base") == "google_flan-t5-base"


def test_target_dir_uses_namespace_repo_name() -> None:
    assert target_dir("Qwen/Qwen3.5-9B", cache_dir=Path("/mnt/data/hf-cache")) == Path(
        "/mnt/data/hf-cache/Qwen_Qwen3.5-9B"
    )


def test_hint_when_env_is_unset() -> None:
    hint = hf_cache_hint({})
    assert hint is not None
    assert "MIND2WEB_HF_CACHE" in hint and "/mnt/data/hf-cache" in hint


def test_no_hint_when_env_is_set() -> None:
    assert hf_cache_hint({"MIND2WEB_HF_CACHE": "/mnt/data/hf-cache"}) is None
