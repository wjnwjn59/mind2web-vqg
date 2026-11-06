"""Evaluate a model on the TreeVQG JSONL: predictions, offline metrics, per level.

This is the *second half* of the MVP: once ``m2w-build-treevqg`` has produced
``examples.jsonl``, the existing API / HuggingFace / VLM backends can be scored without
any new model code. The same file will later be consumed by the structural encoder.

The offline metrics come from :mod:`mind2web_vqg.metrics`; the builder writes
``meta.category = level`` so that report already contains an L1/L2/L3 breakdown.

Usage::

    uv run m2w-eval-treevqg --split test_task --model api/gpt-4o-mini --max-samples 6
    uv run m2w-eval-treevqg --examples data/processed/treevqg/train/examples.jsonl \\
        --model vlm/Qwen/Qwen3.5-9B --max-samples 6
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from mind2web_vqg.cli import ensure_utf8
from mind2web_vqg.metrics import compute_metrics
from mind2web_vqg.models import build_model
from mind2web_vqg.paths import OUTPUT_DIR
from mind2web_vqg.treevqg import LEVEL_NAMES, LEVELS
from mind2web_vqg.treevqg import io as tree_io


def run_eval(
    *,
    examples: str | Path | None = None,
    split: str = "test_task",
    model_name: str = "api/gpt-4o-mini",
    max_samples: int | None = None,
    out_dir: str | Path | None = None,
) -> Path:
    """Generate a question for each TreeVQG row and write predictions + metrics."""
    path = Path(examples) if examples else tree_io.default_examples_path(split)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Run `uv run m2w-build-treevqg --split {split}` first."
        )
    rows = tree_io.read_jsonl(path)
    if max_samples:
        rows = rows[: int(max_samples)]
    if not rows:
        raise ValueError(f"{path} has no rows")

    model = build_model(model_name)
    results: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        prediction = model.generate(str(row.get("input") or ""), image=row.get("image"))
        results.append({**row, "prediction": prediction})
        if index % 10 == 0 or index == len(rows):
            print(f"[treevqg-eval] {index}/{len(rows)}")

    out = Path(out_dir) if out_dir else OUTPUT_DIR / f"treevqg_eval_{split}"
    out.mkdir(parents=True, exist_ok=True)
    predictions = out / "predictions.json"
    predictions.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    metrics = compute_metrics(results)
    # The metrics module groups by ``meta.category``; the builder sets it to the level,
    # so this alias is exact and costs nothing.
    metrics["by_level"] = metrics.get("by_category", {})
    metrics["level_names"] = LEVEL_NAMES
    metrics["levels_present"] = [
        level for level in LEVELS if level in metrics["by_level"]
    ]
    metrics_path = out / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"[treevqg-eval] {len(results)} example(s) -> {predictions}")
    for level in metrics["levels_present"]:
        block = metrics["by_level"][level]
        print(
            f"  {level} ({LEVEL_NAMES[level]:9s}) n={block['n']:4d} "
            f"well-formed={block['well_formed_rate']:.2f} "
            f"ROUGE-L={block['rouge_l']:.2f} chrF={block['chrf']:.2f}"
        )
    print(f"[treevqg-eval] metrics -> {metrics_path}")
    return predictions


def main(argv: list[str] | None = None) -> int:
    ensure_utf8()
    parser = argparse.ArgumentParser(
        description="Evaluate a model on the TreeVQG 3-level dataset."
    )
    parser.add_argument("--split", default="test_task", help="used to locate the default JSONL")
    parser.add_argument("--examples", type=Path, default=None, help="TreeVQG examples.jsonl")
    parser.add_argument(
        "--model",
        default="api/gpt-4o-mini",
        help="api/..., hf/... or vlm/... backend (see models/)",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="cap the rows evaluated")
    parser.add_argument(
        "--out", type=Path, default=None, help="default: outputs/treevqg_eval_<split>"
    )
    args = parser.parse_args(argv)

    run_eval(
        examples=args.examples,
        split=args.split,
        model_name=args.model,
        max_samples=args.max_samples,
        out_dir=args.out,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
