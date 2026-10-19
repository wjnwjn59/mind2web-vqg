"""Pipeline smoke test: screenshot -> VLM -> short answer.

This is **not** the final VQG task. It runs a handful of simple probes over a few
exported screenshots so you can check three things before designing anything:

1. the screenshots are readable (render quality, downscaling),
2. the model can see them (no silent image-dropping in the processor),
3. latency and answer style are acceptable.

The probes deliberately follow the classic VQA taxonomy (what / how / where /
yes-no / count / descriptive), because the interesting research question is how
much of that survives in a GUI-agent setting.

Usage::

    uv run m2w-mm-export --split test_task --limit 5
    uv run m2w-vlm-smoke --split test_task --limit 5
    uv run m2w-vlm-smoke --split test_task --limit 2 --model api/Qwen/Qwen3.5-9B

With a served model (vLLM/SGLang) set ``OPENAI_BASE_URL=http://localhost:8000/v1``
and use the ``api/`` prefix — no local weights, much faster.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from mind2web_vqg.cli import ensure_utf8
from mind2web_vqg.config import load_config
from mind2web_vqg.data.multimodal import export_samples
from mind2web_vqg.models import build_model
from mind2web_vqg.paths import OUTPUT_DIR, PROCESSED_DIR

#: Simple probes. ``{task}`` is filled from the example; others are static.
PROBES: dict[str, dict[str, str]] = {
    "what_heading": {
        "type": "what",
        "question": "What is the main heading of this web page? Reply with the text only.",
    },
    "descriptive": {
        "type": "descriptive",
        "question": "In one sentence, what can a user do on this page?",
    },
    "yes_no": {
        "type": "yes/no",
        "question": "Is a search box visible on this page? Answer yes or no.",
    },
    "count": {
        "type": "count",
        "question": "How many items are in the top navigation bar? Answer with a number.",
    },
    "how_next": {
        "type": "how",
        "question": (
            "The user's task is: {task}\n"
            "What is the next action (click, type, or select), and on which element? "
            "Answer in one sentence."
        ),
    },
    "where_target": {
        "type": "where",
        "question": (
            "The user's task is: {task}\n"
            "Where on the page is the element they should use next? "
            "Describe its position (e.g. 'top-right header')."
        ),
    },
}


def build_prompt(probe: dict[str, str], example: dict[str, Any]) -> str:
    """Fill ``{task}`` in a probe; other braces are left untouched."""
    return probe["question"].replace("{task}", example.get("task") or "")


def load_examples(
    split: str, limit: int, examples_path: str | Path | None = None
) -> list[dict[str, Any]]:
    """Read the exported ``examples.jsonl`` (exporting it first if missing)."""
    path = Path(examples_path) if examples_path else (
        PROCESSED_DIR / "multimodal" / split / "examples.jsonl"
    )
    if not path.exists():
        export_samples(split, limit=limit)
    with open(path, encoding="utf-8") as f:
        examples = [json.loads(line) for line in f if line.strip()]
    return examples[:limit]


def run_smoke(
    split: str = "test_task",
    *,
    limit: int = 5,
    model_name: str = "vlm/Qwen/Qwen3.5-9B",
    probes: list[str] | None = None,
    out_dir: Path | None = None,
    model_kwargs: dict[str, Any] | None = None,
    examples_path: str | Path | None = None,
) -> Path:
    """Run every probe on every example, write ``results.jsonl`` + ``report.md``."""
    names = probes or list(PROBES)
    unknown = sorted(set(names) - set(PROBES))
    if unknown:
        raise ValueError(f"unknown probe(s): {unknown}. Choose from {list(PROBES)}")

    examples = load_examples(split, limit, examples_path)
    model = build_model(model_name, **(model_kwargs or {}))

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = out_dir or OUTPUT_DIR / "smoke_vlm" / f"{split}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for example in examples:
        for name in names:
            probe = PROBES[name]
            prompt = build_prompt(probe, example)
            started = time.perf_counter()
            answer = model.generate(prompt, image=example["image"])
            elapsed = time.perf_counter() - started
            results.append(
                {
                    "id": example["id"],
                    "annotation_id": example.get("annotation_id"),
                    "probe": name,
                    "probe_type": probe["type"],
                    "question": prompt,
                    "answer": answer,
                    "seconds": round(elapsed, 2),
                    "image": example["image"],
                    "target_action": example.get("target_action"),
                }
            )
            print(f"[smoke] {example['id']} / {name:14s} {elapsed:5.1f}s  {answer[:70]}")

    with open(out_dir / "results.jsonl", "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    (out_dir / "report.md").write_text(
        render_report(results, split=split, model_name=model_name), encoding="utf-8"
    )
    print(f"[smoke] {len(results)} answer(s) -> {out_dir}")
    return out_dir


def render_report(results: list[dict[str, Any]], *, split: str, model_name: str) -> str:
    """Markdown table: one row per example, one column per probe."""
    by_id: dict[str, dict[str, str]] = {}
    order: list[str] = []
    for row in results:
        key = row["id"]
        if key not in by_id:
            by_id[key] = {}
            order.append(key)
        by_id[key][row["probe"]] = row["answer"].replace("\n", " ").replace("|", "\\|")

    probes = list(dict.fromkeys(row["probe"] for row in results))
    lines = [
        f"# VLM smoke test — {split}",
        "",
        f"- model: `{model_name}`",
        f"- examples: {len(order)}",
        f"- probes: {', '.join(probes)}",
        "",
        "| example | " + " | ".join(probes) + " |",
        "| --- | " + " | ".join("---" for _ in probes) + " |",
    ]
    for key in order:
        lines.append(f"| `{key}` | " + " | ".join(by_id[key].get(p, "") for p in probes) + " |")

    empty = sum(1 for r in results if not r["answer"].strip())
    avg = sum(r["seconds"] for r in results) / max(len(results), 1)
    lines += ["", f"- empty answers: {empty}/{len(results)}", f"- mean latency: {avg:.1f}s", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ensure_utf8()
    parser = argparse.ArgumentParser(description="Screenshot -> VLM smoke test.")
    parser.add_argument("--config", type=Path, default=None, help="YAML (see configs/smoke.yaml)")
    parser.add_argument("--split", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--model",
        default=None,
        help="vlm/<repo> for local transformers, api/<model> for a served endpoint",
    )
    parser.add_argument(
        "--probes", nargs="+", default=None, choices=list(PROBES), help="subset of probes"
    )
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config) if args.config else None
    get = (lambda key, default: cfg.get(key, default)) if cfg else (lambda key, default: default)

    split = args.split or get("data.split", "test_task")
    limit = args.limit if args.limit is not None else int(get("data.limit", 5))
    model_name = args.model or get("model.name", "vlm/Qwen/Qwen3.5-9B")
    probes = args.probes or get("probes", None)
    max_new_tokens = args.max_new_tokens or int(get("model.max_new_tokens", 256))

    kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens}
    if model_name.startswith("vlm/"):
        kwargs["enable_thinking"] = bool(get("model.enable_thinking", False))

    run_smoke(
        split,
        limit=limit,
        model_name=model_name,
        probes=probes,
        out_dir=args.out,
        model_kwargs=kwargs,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
