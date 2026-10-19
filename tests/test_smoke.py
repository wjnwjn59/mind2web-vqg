"""Smoke tests: the project imports and the config loads."""

from mind2web_vqg import __version__
from mind2web_vqg.config import load_config
from mind2web_vqg.paths import PROJECT_ROOT
from mind2web_vqg.paths import main as paths_main


def test_version() -> None:
    assert __version__


def test_config_loads() -> None:
    cfg = load_config()
    assert cfg.get("model.name")


def test_project_root() -> None:
    assert (PROJECT_ROOT / "pyproject.toml").exists()


def test_paths_main_reports_resolved_dirs(capsys) -> None:
    assert paths_main() == 0
    out = capsys.readouterr().out
    for name in ("MIND2WEB_DATA_DIR", "MIND2WEB_MM_DIR", "MIND2WEB_HF_CACHE"):
        assert name in out
    assert "hf-cache" in out
