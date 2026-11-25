"""Run a question-generation model on TreeVQG-Bench and score what it writes.

``m2w-build-bench`` writes rows with two different views::

    condition   task + UI-tree outline (the anchor is marked ``*``)  -> what the model sees
    question    the gold question                                     -> what we score

This module feeds ``condition`` (and, with ``--with-image``, the screenshot) to any
backend from :mod:`mind2web_vqg.models`, appends every prediction to
``predictions.jsonl`` as it is produced, then scores the run per question type.

The scores are intentionally simple and explainable — a benchmark you cannot read is
not a benchmark:

``asks``
    fraction of predictions that are actually questions (contain ``?``).
``answer mention``
    fraction whose text names the gold answer (a section, a label, a region…).
``token F1`` / ``ROUGE-L``
    lexical overlap with the gold question; low for a zero-shot model is expected,
    high while ``answer mention`` is low usually means the question is too vague.

Usage::

    # 1) weights -> $MIND2WEB_HF_CACHE/<namespace>_<repo>/
    uv run m2w-download-model Qwen/Qwen2.5-VL-7B-Instruct

    # 2) generate + score (start with --per-type 5 before a full run)
    uv run m2w-eval-bench --split test_task --per-type 5 \\
        --model vlm/Qwen/Qwen2.5-VL-7B-Instruct

    # 3) same command again: finished ids are skipped, scoring is redone
    #    -> outputs/bench_eval/<split>/<model>/{predictions.jsonl,metrics.json,report.md}

A bare Hub id (``--model Qwen/Qwen2.5-VL-7B-Instruct``) is read as ``vlm/...``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from mind2web_vqg.cli import ensure_utf8
from mind2web_vqg.paths import OUTPUT_DIR, PROCESSED_DIR
from mind2web_vqg.treevqg import io as tree_io
from mind2web_vqg.treevqg.bench import REQUIRES_TREE, TYPES

#: Small enough for a 24 GB card in bf16; swap for ``vlm/Qwen/Qwen3.5-9B`` (needs
#: ``--load-in-4bit`` there) or any other ``*ForConditionalGeneration`` checkpoint.
DEFAULT_MODEL = "vlm/Qwen/Qwen2.5-VL-7B-Instruct"

#: What the model is asked to produce, per question type. The answer is never shown:
#: only the *kind* of question is described, exactly like a task instruction.
TYPE_PROMPTS: dict[str, str] = {
    "container_membership": (
        "Ask WHICH SECTION of the page — a card, form, panel or navigation area — contains "
        "the control marked *, so the answer is the name of that section."
    ),
    "sibling_disambiguation": (
        "Several controls on the page share the same label. Ask WHICH one the user's task "
        "needs, so the answer tells them apart by section or by region."
    ),
    "absence": (
        "Ask a YES/NO question: does a particular section of the page contain a control the "
        "user could use for this task?"
    ),
    "consequence": (
        "Ask WHAT HAPPENS after the user activates the control marked * — which page, panel "
        "or state appears next — so the answer names the next state."
    ),
    "target_label": (
        "Ask WHAT TEXT is written on the control marked *, so the answer is that visible label."
    ),
    "target_region": (
        "Ask WHERE on the page the control marked * is, so the answer is a coarse 3x3 region "
        "(top-left, top-center, …, bottom-right)."
    ),
}

GENERIC_PROMPT = (
    "Ask ONE specific question about the control marked *, so the answer is something a "
    "person can state after looking at the page."
)

_RULES = (
    "Rules:\n"
    "- Ask about one thing only; use just the task and the page outline.\n"
    "- Do not answer the question, do not explain, do not add quotation marks.\n"
    "- Output exactly one question, ending with '?'."
)

#: ``bench.py`` prefixes every ``condition`` with this generic instruction; the runner
#: replaces it with the type-specific one above.
_PREAMBLE = "You are building ONE question"

_WORD_RE = re.compile(r"[a-z0-9]+")

#: Unicode dashes/minus are not folded by NFKD; map them so ``top–left`` == ``top-left``.
_DASHES = str.maketrans({**{chr(code): "-" for code in range(0x2010, 0x2016)}, "\u2212": "-"})


# --------------------------------------------------------------------------- #
# Prompt building (pure)
# --------------------------------------------------------------------------- #
def condition_body(condition: Any) -> str:
    """``condition`` without the generic generator preamble (task + outline remain)."""
    text = str(condition or "").strip()
    if text.startswith(_PREAMBLE):
        _, _, rest = text.partition("\n\n")
        return rest.strip()
    return text


def build_prompt(row: dict[str, Any], *, with_image: bool = False) -> str:
    """The full instruction for one bench row (uses ``condition``, never ``answer``)."""
    kind = str(row.get("type") or "")
    head = ["You write evaluation questions for a web-page question-answering benchmark."]
    head.append(TYPE_PROMPTS.get(kind, GENERIC_PROMPT))
    if with_image:
        head.append(
            "A screenshot of the same page is attached; the outline below is its "
            "accessible UI structure."
        )
    head.append(_RULES)
    body = condition_body(row.get("condition") or row.get("input") or "")
    return "\n".join(head) + "\n\n" + body


# --------------------------------------------------------------------------- #
# Scoring helpers (pure)
# --------------------------------------------------------------------------- #
def normalize(text: Any) -> str:
    """Lowercase, accent-folded, whitespace-collapsed text."""
    raw = unicodedata.normalize("NFKD", str(text or ""))
    ascii_text = "".join(ch for ch in raw if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", ascii_text.translate(_DASHES)).strip().lower()


def tokens(text: Any) -> list[str]:
    """Word-level tokens (hyphens split, so ``top-left`` == ``top left``)."""
    return _WORD_RE.findall(normalize(text).replace("-", " "))


def token_f1(prediction: Any, gold: Any) -> float:
    """Bag-of-words F1 against the gold question."""
    pred, ref = Counter(tokens(prediction)), Counter(tokens(gold))
    if not pred or not ref:
        return 0.0
    overlap = sum((pred & ref).values())
    if not overlap:
        return 0.0
    precision = overlap / sum(pred.values())
    recall = overlap / sum(ref.values())
    return round(2 * precision * recall / (precision + recall), 4)


def _lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    """Length of the longest common subsequence (two rolling rows, O(len(right)))."""
    if not left or not right:
        return 0
    previous = [0] * (len(right) + 1)
    for token in left:
        current = [0]
        for index, other in enumerate(right, start=1):
            if token == other:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def rouge_l(prediction: Any, gold: Any) -> float:
    """Token-level ROUGE-L F-measure against the gold question."""
    pred, ref = tokens(prediction), tokens(gold)
    if not pred or not ref:
        return 0.0
    lcs = _lcs_length(pred, ref)
    if not lcs:
        return 0.0
    precision, recall = lcs / len(pred), lcs / len(ref)
    return round(2 * precision * recall / (precision + recall), 4)


def answer_mentioned(prediction: Any, answer: Any) -> bool | None:
    """Whether the prediction names the gold answer (whole words, hyphen-insensitive).

    ``None`` when there is no gold answer to check (the row is then left out of the mean).
    """
    gold = normalize(answer).replace("-", " ")
    if not gold:
        return None
    text = normalize(prediction).replace("-", " ")
    if re.search(rf"(?<![a-z0-9]){re.escape(gold)}(?![a-z0-9])", text):
        return True
    gold_tokens = tokens(answer)
    return bool(gold_tokens) and all(token in set(tokens(prediction)) for token in gold_tokens)


def is_question(text: Any) -> bool:
    """A cheap sanity check: did the model actually write a question?"""
    return "?" in str(text or "")


# --------------------------------------------------------------------------- #
# Row selection
# --------------------------------------------------------------------------- #
def select_rows(
    rows: Iterable[dict[str, Any]],
    *,
    types: Iterable[str] | None = None,
    per_type: int | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Filter by type and cap each type, keeping the file order (already sampled)."""
    wanted = set(types) if types else None
    picked: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for row in rows:
        kind = str(row.get("type"))
        if wanted is not None and kind not in wanted:
            continue
        if per_type is not None and counts[kind] >= per_type:
            continue
        counts[kind] += 1
        picked.append(row)
        if limit is not None and len(picked) >= limit:
            break
    return picked


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def _mean(values: Iterable[float]) -> float:
    materialized = [float(value) for value in values]
    return round(sum(materialized) / len(materialized), 3) if materialized else 0.0


def _block(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate one group of scored records."""
    return {
        "n": len(records),
        "asks_rate": _mean(1.0 if record["asks"] else 0.0 for record in records),
        "answer_mention": _mean(
            record["mention"] for record in records if record["mention"] is not None
        ),
        "token_f1": _mean(record["token_f1"] for record in records),
        "rouge_l": _mean(record["rouge_l"] for record in records),
        "exact_match": _mean(1.0 if record["exact"] else 0.0 for record in records),
    }


def score_predictions(
    rows: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Score ``predictions`` (keyed by ``id``) against the gold rows."""
    records: list[dict[str, Any]] = []
    for row in rows:
        row_id = str(row.get("id"))
        prediction = str(predictions.get(row_id, {}).get("prediction") or "")
        gold_question = str(row.get("question") or "")
        gold_answer = str(row.get("answer") or "")
        records.append(
            {
                "id": row_id,
                "type": str(row.get("type")),
                "requires_tree": bool(row.get("meta", {}).get("requires_tree")),
                "prediction": prediction,
                "gold_question": gold_question,
                "gold_answer": gold_answer,
                "asks": is_question(prediction),
                "mention": answer_mentioned(prediction, gold_answer),
                "token_f1": token_f1(prediction, gold_question),
                "rouge_l": rouge_l(prediction, gold_question),
                "exact": bool(prediction) and normalize(prediction) == normalize(gold_question),
            }
        )

    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_type[record["type"]].append(record)

    ordered = [kind for kind in TYPES if kind in by_type]
    ordered += sorted(kind for kind in by_type if kind not in TYPES)
    tree_yes = [record for record in records if record["requires_tree"]]
    tree_no = [record for record in records if not record["requires_tree"]]

    evaluation = {
        "rows": len(rows),
        "predicted": sum(1 for row in rows if str(row.get("id")) in predictions),
        "totals": _block(records),
        "by_type": {kind: _block(by_type[kind]) for kind in ordered},
        "by_track": {
            "tree_yes": _block(tree_yes),
            "tree_no": _block(tree_no),
        },
    }
    evaluation["warnings"] = build_warnings(evaluation, records)
    return evaluation


def build_warnings(evaluation: dict[str, Any], records: list[dict[str, Any]]) -> list[str]:
    """Human-readable red flags; each one has a known cause and a known fix."""
    warnings: list[str] = []
    totals = evaluation["totals"]
    if not evaluation["rows"]:
        return ["no rows selected: check --types / --per-type and the examples path"]

    missing = evaluation["rows"] - evaluation["predicted"]
    if missing:
        warnings.append(
            f"{missing} row(s) have no prediction yet; run the command again (resume) "
            "before trusting the means"
        )
    empty = sum(1 for record in records if not record["prediction"])
    if empty:
        warnings.append(
            f"{empty} prediction(s) are empty — the model returned nothing, often a "
            "tokenizer/chat-template problem rather than a hard question"
        )
    if totals["asks_rate"] < 0.8:
        warnings.append(
            f"only {totals['asks_rate']:.0%} of predictions contain '?': the model is "
            "probably answering or explaining instead of asking; tighten the prompt or "
            "raise --max-new-tokens"
        )
    for kind, block in evaluation["by_type"].items():
        if kind in REQUIRES_TREE and block["answer_mention"] == 0.0:
            warnings.append(
                f"{kind}: no prediction names the gold answer — this type is a "
                "compositional question, a flat caption model is expected to fail; "
                "check the prompt before blaming the model"
            )
    if totals["token_f1"] < 0.10 and totals["asks_rate"] >= 0.8:
        warnings.append(
            f"token F1 {totals['token_f1']:.2f} is low: wording differs from the reference "
            "templates, which is normal zero-shot; read answer_mention and report.md first"
        )
    return warnings


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def render_markdown(evaluation: dict[str, Any], *, split: str, settings: dict[str, Any]) -> str:
    totals = evaluation["totals"]
    lines = [
        f"# TreeVQG-Bench v1 — question generation — `{split}`",
        "",
        f"- model: `{settings.get('model')}`",
        f"- examples: `{settings.get('examples')}`",
        f"- screenshot given to the model: {'yes' if settings.get('with_image') else 'no'}",
        f"- rows: {evaluation['rows']} (predicted: {evaluation['predicted']})",
        "",
        "| type | n | asks | answer mention | token F1 | ROUGE-L |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for kind, block in evaluation["by_type"].items():
        lines.append(
            f"| `{kind}` | {block['n']} | {block['asks_rate']:.2f} | "
            f"{block['answer_mention']:.2f} | {block['token_f1']:.2f} | {block['rouge_l']:.2f} |"
        )
    lines += [
        "",
        "## Aggregate",
        "",
        "| group | n | asks | answer mention | token F1 | ROUGE-L |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, block in [("all", totals), *evaluation["by_track"].items()]:
        if not block["n"]:
            continue
        lines.append(
            f"| {name} | {block['n']} | {block['asks_rate']:.2f} | "
            f"{block['answer_mention']:.2f} | {block['token_f1']:.2f} | {block['rouge_l']:.2f} |"
        )
    lines += ["", "## Warnings", ""]
    warnings = evaluation["warnings"]
    lines += [f"- {warning}" for warning in warnings] if warnings else ["- none"]
    lines += [
        "",
        "## How to read this",
        "",
        "- `asks` should be ~1.0; below that the run is invalid, not just weak.",
        "- `answer mention` = the generated question names the same section / label / region "
        "as the gold one; the metric that matters for `tree=yes` types.",
        "- token F1 / ROUGE-L compare wording with one reference template: useful for "
        "regressions, not as the headline score.",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def resolve_model_name(name: str) -> str:
    """``Qwen/...`` -> ``vlm/Qwen/...``; known prefixes pass through."""
    if name.startswith(("api/", "hf/", "vlm/")):
        return name
    return f"vlm/{name}"


def load_model(
    name: str,
    *,
    max_new_tokens: int = 96,
    load_in_4bit: bool = False,
    max_pixels: int | None = None,
) -> Any:
    """Build a backend from :mod:`mind2web_vqg.models` (imports torch only here)."""
    from mind2web_vqg.models import build_model

    resolved = resolve_model_name(name)
    kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens}
    if resolved.startswith("vlm/"):
        kwargs["load_in_4bit"] = load_in_4bit
        if max_pixels:
            kwargs["max_pixels"] = max_pixels
    return build_model(resolved, **kwargs)


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def run_dir_for(model_name: str, split: str, out_dir: str | Path | None = None) -> Path:
    if out_dir:
        return Path(out_dir)
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name)
    return OUTPUT_DIR / "bench_eval" / split / slug


def read_predictions(path: str | Path) -> dict[str, dict[str, Any]]:
    """``{id: record}`` from a (possibly incomplete) ``predictions.jsonl``."""
    path = Path(path)
    if not path.is_file():
        return {}
    return {str(row.get("id")): row for row in tree_io.read_jsonl(path) if row.get("id")}


def generate(
    rows: list[dict[str, Any]],
    *,
    out_dir: str | Path,
    model: Any,
    model_name: str,
    with_image: bool = False,
    resume: bool = True,
    progress: Any = None,
) -> Path:
    """Write one prediction per row to ``<out_dir>/predictions.jsonl``, resuming by id.

    Each line is flushed immediately, so an interrupted 7B run keeps its work and the
    next invocation only redoes the missing ids.
    """
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "predictions.jsonl"
    done = set(read_predictions(path)) if resume else set()
    todo = [row for row in rows if str(row.get("id")) not in done]
    print(f"[bench-eval] {len(rows)} row(s), {len(done)} already predicted, {len(todo)} to go")

    missing_images = 0
    with open(path, "a" if resume else "w", encoding="utf-8") as handle:
        for index, row in enumerate(todo, start=1):
            image = row.get("image") if with_image else None
            if image and not Path(str(image)).is_file():
                missing_images += 1
                image = None
            prediction = model.generate(build_prompt(row, with_image=bool(image)), image=image)
            record = {
                "id": str(row.get("id")),
                "type": row.get("type"),
                "website": row.get("meta", {}).get("website"),
                "prediction": prediction,
                "gold_question": row.get("question"),
                "gold_answer": row.get("answer"),
                "image": image,
                "model": model_name,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            if progress is not None:
                progress(index, len(todo), record)

    if missing_images:
        print(
            f"[bench-eval] warning: {missing_images} screenshot(s) were missing; those rows "
            "ran text-only (check the `image` paths in examples.jsonl)"
        )
    return path


def print_summary(evaluation: dict[str, Any], *, model_name: str, with_image: bool) -> None:
    print(f"[bench-eval] model: {model_name} | screenshot: {'yes' if with_image else 'no'}")
    print(f"{'type':26s} {'n':>4s} {'asks':>5s} {'mention':>8s} {'F1':>6s} {'ROUGE-L':>8s}")
    for kind, block in evaluation["by_type"].items():
        print(
            f"{kind:26s} {block['n']:4d} {block['asks_rate']:5.2f} "
            f"{block['answer_mention']:8.2f} {block['token_f1']:6.2f} {block['rouge_l']:8.2f}"
        )
    totals = evaluation["totals"]
    print(
        f"{'ALL':26s} {totals['n']:4d} {totals['asks_rate']:5.2f} "
        f"{totals['answer_mention']:8.2f} {totals['token_f1']:6.2f} {totals['rouge_l']:8.2f}"
    )
    for name, block in evaluation["by_track"].items():
        if block["n"]:
            print(
                f"{name:26s} {block['n']:4d} {block['asks_rate']:5.2f} "
                f"{block['answer_mention']:8.2f} {block['token_f1']:6.2f} "
                f"{block['rouge_l']:8.2f}"
            )
    for warning in evaluation["warnings"]:
        print(f"[bench-eval] warning: {warning}")


def default_examples_path(split: str) -> Path:
    return PROCESSED_DIR / "bench" / "v1" / split / "examples.jsonl"


def run_eval(
    examples_path: str | Path | None = None,
    *,
    split: str = "test_task",
    types: Iterable[str] | None = None,
    per_type: int | None = None,
    limit: int | None = None,
    model_name: str = DEFAULT_MODEL,
    with_image: bool = False,
    max_new_tokens: int = 96,
    load_in_4bit: bool = False,
    max_pixels: int | None = None,
    out_dir: str | Path | None = None,
    resume: bool = True,
    model: Any = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Select rows, generate the missing predictions, score, and write the three files."""
    path = Path(examples_path) if examples_path else default_examples_path(split)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Run `uv run m2w-build-bench --split {split}` first."
        )

    rows = select_rows(tree_io.read_jsonl(path), types=types, per_type=per_type, limit=limit)
    if not rows:
        raise ValueError(
            f"no rows selected from {path}; check --types {list(types or TYPES)} / --per-type"
        )

    if dry_run:
        print(f"[bench-eval] dry run: {len(rows)} row(s) from {path} (no model loaded)")
        for row in rows[:3]:
            print(f"\n--- {row.get('type')} {row.get('id')} ---")
            print(build_prompt(row, with_image=with_image))
        return {
            "dry_run": True,
            "examples": str(path),
            "rows": len(rows),
            "model": model_name,
        }

    destination = run_dir_for(model_name, split, out_dir)
    backend = model or load_model(
        model_name,
        max_new_tokens=max_new_tokens,
        load_in_4bit=load_in_4bit,
        max_pixels=max_pixels,
    )
    predictions_path = generate(
        rows,
        out_dir=destination,
        model=backend,
        model_name=model_name,
        with_image=with_image,
        resume=resume,
    )

    predictions = read_predictions(predictions_path)
    evaluation = score_predictions(rows, predictions)
    settings = {
        "split": split,
        "model": model_name,
        "examples": str(path),
        "with_image": with_image,
        "types": list(types or TYPES),
        "per_type": per_type,
        "limit": limit,
        "predictions": str(predictions_path),
    }
    evaluation["settings"] = settings

    tree_io.write_json(evaluation, destination / "metrics.json")
    report = render_markdown(evaluation, split=split, settings=settings)
    (destination / "report.md").write_text(report, encoding="utf-8")

    print_summary(evaluation, model_name=model_name, with_image=with_image)
    print(f"[bench-eval] predictions -> {predictions_path}")
    print(f"[bench-eval] metrics     -> {destination / 'metrics.json'}")
    print(f"[bench-eval] report      -> {destination / 'report.md'}")
    return evaluation


def main(argv: list[str] | None = None) -> int:
    ensure_utf8()
    parser = argparse.ArgumentParser(
        description="Generate questions for TreeVQG-Bench with a local model and score them."
    )
    parser.add_argument("--examples", default=None, help="bench examples.jsonl")
    parser.add_argument("--split", default="test_task", help="used to find the default path")
    parser.add_argument("--types", nargs="+", default=None, choices=list(TYPES))
    parser.add_argument("--per-type", type=int, default=None, help="cap rows per type")
    parser.add_argument("--limit", type=int, default=None, help="cap rows in total")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"api/..., hf/... or vlm/... (default {DEFAULT_MODEL}); a bare repo id = vlm/",
    )
    parser.add_argument(
        "--with-image", action="store_true", help="also give the screenshot to the model"
    )
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--load-in-4bit", action="store_true", help="QLoRA-style 4-bit load")
    parser.add_argument("--max-pixels", type=int, default=None, help="per-image pixel budget")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--no-resume", action="store_true", help="ignore existing predictions")
    parser.add_argument("--dry-run", action="store_true", help="print prompts, load no model")
    args = parser.parse_args(argv)

    run_eval(
        args.examples,
        split=args.split,
        types=args.types,
        per_type=args.per_type,
        limit=args.limit,
        model_name=args.model,
        with_image=args.with_image,
        max_new_tokens=args.max_new_tokens,
        load_in_4bit=args.load_in_4bit,
        max_pixels=args.max_pixels,
        out_dir=args.out_dir,
        resume=not args.no_resume,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
