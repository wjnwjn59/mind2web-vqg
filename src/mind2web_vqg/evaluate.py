"""Evaluation loop: generate a question for each example and write the results.

The skeleton is ready; you only need to edit ``build_examples`` (data/prepare.py)
and add the metrics that match your task.

Usage::

    uv run m2w-eval --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mind2web_vqg.cli import ensure_utf8
from mind2web_vqg.config import load_config
from mind2web_vqg.data.load import load_split
from mind2web_vqg.data.prepare import build_examples
from mind2web_vqg.models import build_model
from mind2web_vqg.paths import OUTPUT_DIR


def run_eval(config_path: str | Path | None = None) -> Path:
    cfg = load_config(config_path)

    split = cfg.get("data.splits", ["test_task"])[0]
    root = cfg.get("data.root") or None
    max_samples = cfg.get("data.max_samples")

    instances = load_split(split) if root is None else load_split(split, root)
    examples = build_examples(instances)
    if max_samples:
        examples = examples[: int(max_samples)]

    model = build_model(cfg.get("model.name", "api/gpt-4o-mini"))

    run_dir = OUTPUT_DIR / f"eval_{split}"
    run_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for ex in examples:
        pred = model.generate(ex["input"])
        results.append({**ex, "prediction": pred})

    out_file = run_dir / "predictions.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[eval] {len(results)} examples -> {out_file}")
    return out_file


def main(argv: list[str] | None = None) -> int:
    ensure_utf8()
    parser = argparse.ArgumentParser(description="Evaluate a question generation model.")
    parser.add_argument(
        "--config", default=None, help="YAML file (defaults to configs/default.yaml)"
    )
    args = parser.parse_args(argv)
    run_eval(args.config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
