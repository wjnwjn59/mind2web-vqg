"""Simple YAML configuration loader (no hydra dependency)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from mind2web_vqg.paths import CONFIG_DIR


@dataclass
class Config:
    """Run configuration with nested groups merged for easy access."""

    raw: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Access nested values like ``cfg.get("model.name")``."""
        node: Any = self.raw
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


def load_config(path: str | Path | None = None) -> Config:
    """Read a YAML file; defaults to ``configs/default.yaml``."""
    cfg_path = Path(path) if path is not None else CONFIG_DIR / "default.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return Config(raw=data)
