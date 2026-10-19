"""Evaluation loop: generate a question for each example and write the results.

The task is screenshot VQG (see ``data/prepare.py`` / ``data/vqg.py``): every example is
an exported screenshot, and the model is asked to produce the question that ``input``
describes. Predictions are compared against the reference question/answer by hand for now;
add the metrics that match your task below.

Usage::

    uv run m2w-eval --config configs/default.yaml
    uv run m2w-eval --model vlm/Qwen/Qwen3.5-9B --max-samples 6
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mind2web_vqg.cli import ensure_utf8
from mind2web_vqg.config import load_config
from mind2web_vqg.data.prepare import build_examples, load_exported
from mind2web_vqg.metrics import compute_metrics
from mind2web_vqg.models import build_model
from mind2web_vqg.paths import OUTPUT_DIR


def run_eval(
    config_path: str | Path | None = None,
    *,
    model_name: str | None = None,
    max_samples: int | None = None,
) -> Path:
    cfg = load_config(config_path)

    split = cfg.get("data.splits", ["test_task"])[0]
    if max_samples is None:
        max_samples = cfg.get("data.max_samples")

    # Screenshots, not the JSON HTML: read what ``m2w-mm-export`` produced.
    exported = load_exported(split, path=cfg.get("data.examples") or None)
    examples = build_examples(exported)
    built = len(examples)
    if max_samples:
        examples = examples[: int(max_samples)]

    print(f"[eval] {len(exported)} screenshot(s) -> {built} example(s), evaluating {len(examples)}")
    if not examples:
        print("[eval] nothing to evaluate — run `uv run m2w-build-vqg` to see why")

    model = build_model(model_name or cfg.get("model.name", "api/gpt-4o-mini"))

    run_dir = OUTPUT_DIR / f"eval_{split}"
    run_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for ex in examples:
        pred = model.generate(ex["input"], image=ex.get("image"))
        results.append({**ex, "prediction": pred})

    out_file = run_dir / "predictions.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    metrics_file = run_dir / "metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(compute_metrics(results), f, ensure_ascii=False, indent=2)

    print(f"[eval] {len(results)} examples -> {out_file}")
    print(f"[eval] offline metrics -> {metrics_file}")
    return out_file


def main(argv: list[str] | None = None) -> int:
    ensure_utf8()
    parser = argparse.ArgumentParser(description="Evaluate a question generation model.")
    parser.add_argument(
        "--config", default=None, help="YAML file (defaults to configs/default.yaml)"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="override model.name, e.g. vlm/Qwen/Qwen3.5-9B or api/gpt-4o-mini",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="cap the number of examples (overrides config)",
    )
    args = parser.parse_args(argv)
    run_eval(args.config, model_name=args.model, max_samples=args.max_samples)
    return 0


if __name__ == "__main__":
    sys.exit(main())
