"""Utility metrics for TreeVQG-Clarify predictions.

The benchmark is *selective asking*, so accuracy alone is not enough. This module
scores a prediction file with four families of numbers:

``Ask``
    Precision / recall / F1 of *whether* to ask. A false positive interrogates the
    user for no reason; a false negative guesses when it should have asked.

``Option recall``
    Fraction of the gold candidates named in the generated question. A question
    that omits an option cannot be answered with that option.

``Answer accuracy``
    Optional field ``answer``: normalized match with the gold container / role /
    region / ``yes``. This is what a user simulator would return.

``Realized information gain``
    ``log2(n_candidates)`` credited only when the question names every option, so a
    vague question earns nothing. Averaged over gold positives; this is the closest
    offline stand-in for EVPI without a downstream agent.

No model is run here — pass a JSONL with ``{id, question, should_ask?, answer?}``.
Missing ids count as ``NO_QUESTION`` (the conservative baseline).

Usage::

    uv run m2w-eval-clarify --split test_task \
        --predictions outputs/preds/clarify_test_task.jsonl
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from mind2web_vqg.cli import ensure_utf8
from mind2web_vqg.paths import OUTPUT_DIR, PROCESSED_DIR
from mind2web_vqg.treevqg import io as tree_io
from mind2web_vqg.treevqg.audit import normalize
from mind2web_vqg.treevqg.clarify import NO_QUESTION
from mind2web_vqg.treevqg.clarify_audit import options_for_row


def info_gain_bits(n_candidates: int) -> float:
    """Entropy reduction from naming one of ``n`` uniformly likely candidates."""
    return math.log2(n_candidates) if n_candidates > 1 else 0.0


def option_recall(question: str, options: Iterable[str]) -> float | None:
    """Fraction of candidate labels present in the question (``None`` if no options)."""
    cleaned = [option for option in options if normalize(option)]
    if not cleaned:
        return None
    text = normalize(question)
    return sum(normalize(option) in text for option in cleaned) / len(cleaned)


def ask_prf(gold: Iterable[bool], pred: Iterable[bool]) -> dict[str, Any]:
    """Confusion-matrix scores for the ask/no-ask decision."""
    pairs = list(zip(gold, pred, strict=True))
    tp = sum(g and p for g, p in pairs)
    fp = sum(not g and p for g, p in pairs)
    fn = sum(g and not p for g, p in pairs)
    tn = sum(not g and not p for g, p in pairs)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "accuracy": round((tp + tn) / len(pairs), 3) if pairs else 0.0,
    }


def _prediction_question(prediction: dict[str, Any]) -> str:
    return str(prediction.get("question") or prediction.get("target") or "").strip()


def _asks(prediction: dict[str, Any]) -> bool:
    question = _prediction_question(prediction)
    if not question or question.strip().upper() == NO_QUESTION:
        return False
    return bool(prediction.get("should_ask", True))


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 3) if values else 0.0


def evaluate_rows(
    rows: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Score predictions against the clarification gold rows."""
    records: list[dict[str, Any]] = []
    for row in rows:
        row_id = str(row.get("id"))
        prediction = predictions.get(row_id, {})
        question = _prediction_question(prediction)
        pred_ask = _asks(prediction)
        gold_ask = bool(row.get("meta", {}).get("should_ask"))
        gold_options = options_for_row(row)
        recall = option_recall(question, gold_options) if pred_ask else None
        gold_answer = row.get("answer")
        pred_answer = prediction.get("answer")
        answer_correct: bool | None = None
        if gold_ask and pred_ask and pred_answer not in (None, ""):
            answer_correct = normalize(pred_answer) == normalize(gold_answer)
        gain = float(row.get("meta", {}).get("info_gain_bits") or 0)
        realized = gain if (gold_ask and pred_ask and recall == 1.0) else 0.0
        records.append(
            {
                "id": row_id,
                "track": row.get("meta", {}).get("track") or row.get("track"),
                "level": row.get("level"),
                "gold_ask": gold_ask,
                "pred_ask": pred_ask,
                "question": question,
                "recall": recall,
                "answer_correct": answer_correct,
                "gold_info_gain": gain,
                "realized_info_gain": realized,
            }
        )

    asks = ask_prf(
        [record["gold_ask"] for record in records],
        [record["pred_ask"] for record in records],
    )
    recalls = [record["recall"] for record in records if record["recall"] is not None]
    option_positives = sum(1 for row in rows if options_for_row(row))
    answered = [record for record in records if record["answer_correct"] is not None]
    positives = [record for record in records if record["gold_ask"]]

    by_track: dict[str, Any] = {}
    for track in sorted({str(record["track"]) for record in records}):
        block = [record for record in records if str(record["track"]) == track]
        block_asks = ask_prf(
            [record["gold_ask"] for record in block],
            [record["pred_ask"] for record in block],
        )
        block_recalls = [record["recall"] for record in block if record["recall"] is not None]
        block_answered = [record for record in block if record["answer_correct"] is not None]
        by_track[track] = {
            "n": len(block),
            "gold_positives": sum(record["gold_ask"] for record in block),
            "pred_positives": sum(record["pred_ask"] for record in block),
            **block_asks,
            "option_recall": _mean([float(value) for value in block_recalls]),
            "answer_accuracy": (
                round(
                    sum(bool(record["answer_correct"]) for record in block_answered)
                    / len(block_answered),
                    3,
                )
                if block_answered
                else None
            ),
            "realized_info_gain_bits": round(
                sum(record["realized_info_gain"] for record in block), 2
            ),
        }

    return {
        "totals": {
            "rows": len(records),
            "gold_positives": len(positives),
            "pred_positives": sum(record["pred_ask"] for record in records),
            "ask": asks,
            "option_recall": _mean([float(value) for value in recalls]),
            "full_option_recall_rate": (
                round(sum(value == 1.0 for value in recalls) / len(recalls), 3)
                if recalls
                else (0.0 if option_positives else 1.0)
            ),
            "answer_accuracy": (
                round(
                    sum(bool(record["answer_correct"]) for record in answered) / len(answered),
                    3,
                )
                if answered
                else None
            ),
            "answered": len(answered),
            "mean_question_words": _mean(
                [
                    float(len(normalize(record["question"]).split()))
                    for record in records
                    if record["pred_ask"]
                ]
            ),
            "realized_info_gain_bits": round(
                sum(record["realized_info_gain"] for record in records), 2
            ),
            "mean_realized_info_gain_bits": _mean(
                [record["realized_info_gain"] for record in positives]
            ),
            "max_realized_info_gain_bits": round(
                sum(record["gold_info_gain"] for record in positives), 2
            ),
            "wasted_questions": asks["fp"],
            "missed_questions": asks["fn"],
        },
        "by_track": by_track,
        "records": records,
    }


def build_warnings(evaluation: dict[str, Any]) -> list[str]:
    totals = evaluation["totals"]
    warnings: list[str] = []
    if totals["ask"]["f1"] < 0.5:
        warnings.append(
            f"Ask-F1 is {totals['ask']['f1']:.2f}: the model either asks at the wrong "
            "steps or misses most opportunities"
        )
    if totals["wasted_questions"]:
        warnings.append(
            f"{totals['wasted_questions']} false-positive question(s): asking when the "
            "screen is unambiguous"
        )
    if totals["missed_questions"]:
        warnings.append(
            f"{totals['missed_questions']} missed question(s): the model guesses where the "
            "gold label asks the user"
        )
    if totals["full_option_recall_rate"] < 1.0:
        warnings.append(
            f"only {totals['full_option_recall_rate']:.0%} of generated questions name every "
            "candidate, so the user cannot select the gold option"
        )
    if totals["mean_realized_info_gain_bits"] < 0.5 * (
        totals["max_realized_info_gain_bits"] / totals["gold_positives"]
        if totals["gold_positives"]
        else 0
    ):
        warnings.append("average realized information gain is below half the gold ceiling")
    return warnings


def render_markdown(evaluation: dict[str, Any], *, split: str) -> str:
    totals = evaluation["totals"]
    ask = totals["ask"]
    lines = [
        f"# TreeVQG-Clarify evaluation — `{split}`",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| rows | {totals['rows']:,} |",
        (
            f"| gold / predicted positives | {totals['gold_positives']:,} / "
            f"{totals['pred_positives']:,} |"
        ),
        (
            f"| Ask precision / recall / F1 | {ask['precision']:.2f} / "
            f"{ask['recall']:.2f} / {ask['f1']:.2f} |"
        ),
        f"| Ask accuracy | {ask['accuracy']:.3f} |",
        f"| option recall (mean) | {totals['option_recall']:.1%} |",
        f"| full option recall | {totals['full_option_recall_rate']:.1%} |",
        f"| answer accuracy | "
        f"{totals['answer_accuracy'] if totals['answer_accuracy'] is not None else 'n/a'} |",
        f"| realized info gain | {totals['realized_info_gain_bits']:.1f} / "
        f"{totals['max_realized_info_gain_bits']:.1f} bits |",
        f"| mean realized info gain | {totals['mean_realized_info_gain_bits']:.3f} bits |",
        (
            f"| wasted / missed questions | {totals['wasted_questions']} / "
            f"{totals['missed_questions']} |"
        ),
        "",
        "## By track",
        "",
        "| track | n | gold pos | pred pos | P | R | F1 | option recall | "
        "answer acc | realized bits |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for track, block in sorted(evaluation["by_track"].items()):
        lines.append(
            f"| {track} | {block['n']:,} | {block['gold_positives']:,} | "
            f"{block['pred_positives']:,} | {block['precision']:.2f} | {block['recall']:.2f} | "
            f"{block['f1']:.2f} | {block['option_recall']:.1%} | "
            f"{block['answer_accuracy'] if block['answer_accuracy'] is not None else 'n/a'} | "
            f"{block['realized_info_gain_bits']:.1f} |"
        )
    warnings = build_warnings(evaluation)
    lines += ["", "## Warnings", ""]
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- none")
    return "\n".join(lines).rstrip() + "\n"


def evaluate_split(
    split: str = "test_task",
    *,
    predictions_path: str | Path,
    examples_path: str | Path | None = None,
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    examples = (
        Path(examples_path)
        if examples_path
        else PROCESSED_DIR / "clarify" / split / "examples.jsonl"
    )
    rows = tree_io.read_jsonl(examples)
    predictions = {str(row.get("id")): row for row in tree_io.read_jsonl(predictions_path)}
    evaluation = evaluate_rows(rows, predictions)
    dest = Path(out_dir) if out_dir else OUTPUT_DIR / "clarify_eval" / split
    tree_io.write_json(evaluation, dest / "eval.json")
    (dest / "eval.md").write_text(render_markdown(evaluation, split=split), encoding="utf-8")
    return {"evaluation": evaluation, "json": dest / "eval.json", "markdown": dest / "eval.md"}


def main(argv: list[str] | None = None) -> int:
    ensure_utf8()
    parser = argparse.ArgumentParser(description="Evaluate TreeVQG-Clarify predictions.")
    parser.add_argument("--split", default="test_task")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--examples", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    result = evaluate_split(
        args.split,
        predictions_path=args.predictions,
        examples_path=args.examples,
        out_dir=args.out,
    )
    totals = result["evaluation"]["totals"]
    print(
        f"[clarify-eval] rows={totals['rows']} gold_pos={totals['gold_positives']} "
        f"pred_pos={totals['pred_positives']} "
        f"Ask-F1={totals['ask']['f1']:.3f} "
        f"option_recall={totals['option_recall']:.1%} "
        f"realized={totals['realized_info_gain_bits']:.1f}/"
        f"{totals['max_realized_info_gain_bits']:.1f} bits"
    )
    for warning in build_warnings(result["evaluation"]):
        print(f"  ! {warning}")
    print(f"[clarify-eval] -> {result['markdown']}")
    print(f"[clarify-eval] -> {result['json']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
