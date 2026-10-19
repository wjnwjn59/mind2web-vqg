"""Smoke tests: the project imports and the config loads."""

from mind2web_vqg import __version__
from mind2web_vqg.config import load_config
from mind2web_vqg.paths import PROJECT_ROOT


def test_version() -> None:
    assert __version__


def test_config_loads() -> None:
    cfg = load_config()
    assert cfg.get("model.name")


def test_project_root() -> None:
    assert (PROJECT_ROOT / "pyproject.toml").exists()
