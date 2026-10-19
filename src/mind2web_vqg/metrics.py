"""Quality metrics for the screenshot VQG task.

Two families:

* **Offline** — pure functions over ``predictions.json``: format validity, diversity
  (unique ratio, Distinct-1/2, self-BLEU) and overlap with the reference question
  (ROUGE-L, chrF). Cheap and deterministic, but the references are *templates*, so a
  correct rephrasing scores low: read these for regressions, not as the goal.
* **Round-trip answerability** — opt-in, needs a model: ask the model the *generated*
  question against the same screenshot, then compare its answer with the ground-truth
  ``answer``. This is the metric that matches the task, so it is always reported next to
  its *ceiling* (the same loop with the reference question).

Usage::

    uv run m2w-metrics --split test_task                # offline only
    uv run m2w-metrics --split test_task --round-trip   # + answerability (needs weights)
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from mind2web_vqg.cli import ensure_utf8
from mind2web_vqg.paths import OUTPUT_DIR

# --------------------------------------------------------------------------- #
# Answer comparison (round-trip)
# --------------------------------------------------------------------------- #
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def normalize(text: Any) -> str:
    """Lowercase, drop punctuation and collapse whitespace."""
    return " ".join(_PUNCT.sub(" ", str(text or "").lower()).split())


def exact_match(prediction: Any, gold: Any) -> float:
    """1.0 when the normalized strings are identical (and non-empty)."""
    pred, want = normalize(prediction), normalize(gold)
    return 1.0 if pred and pred == want else 0.0


def containment(prediction: Any, gold: Any) -> float:
    """1.0 when the normalized gold appears inside the prediction (lenient match)."""
    pred, want = normalize(prediction), normalize(gold)
    return 1.0 if want and want in pred else 0.0


def token_f1(prediction: Any, gold: Any) -> float:
    """Multiset token F1: partial credit for answers that overlap the gold."""
    pred, want = normalize(prediction).split(), normalize(gold).split()
    if not pred or not want:
        return 0.0
    overlap = sum((Counter(pred) & Counter(want)).values())
    if not overlap:
        return 0.0
    precision, recall = overlap / len(pred), overlap / len(want)
    return 2 * precision * recall / (precision + recall)


# --------------------------------------------------------------------------- #
# Question-text overlap (offline)
# --------------------------------------------------------------------------- #
def _lcs_length(a: list[str], b: list[str]) -> int:
    """Longest common subsequence length (rolling row, O(min) memory)."""
    if not a or not b:
        return 0
    previous = [0] * (len(b) + 1)
    for left in a:
        current = [0]
        for j, right in enumerate(b, start=1):
            if left == right:
                current.append(previous[j - 1] + 1)
            else:
                current.append(max(previous[j], current[j - 1]))
        previous = current
    return previous[-1]


def rouge_l(prediction: Any, gold: Any) -> float:
    """ROUGE-L F1 (longest common subsequence) against the reference question."""
    pred, want = normalize(prediction).split(), normalize(gold).split()
    if not pred or not want:
        return 0.0
    lcs = _lcs_length(pred, want)
    if not lcs:
        return 0.0
    precision, recall = lcs / len(pred), lcs / len(want)
    return 2 * precision * recall / (precision + recall)


def _char_ngrams(text: str, n: int) -> Counter:
    return Counter(text[i : i + n] for i in range(max(0, len(text) - n + 1)))


def chrf(prediction: Any, gold: Any, *, n: int = 6, beta: float = 2.0) -> float:
    """chrF: character n-gram F-score, robust for the short template references."""
    pred, want = normalize(prediction), normalize(gold)
    if not pred or not want:
        return 0.0
    total = 0.0
    for k in range(1, n + 1):
        pred_grams, want_grams = _char_ngrams(pred, k), _char_ngrams(want, k)
        if not pred_grams or not want_grams:
            continue
        overlap = sum((pred_grams & want_grams).values())
        if not overlap:
            continue
        precision = overlap / sum(pred_grams.values())
        recall = overlap / sum(want_grams.values())
        beta_sq = beta * beta
        total += (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)
    return total / n


# --------------------------------------------------------------------------- #
# Diversity (offline)
# --------------------------------------------------------------------------- #
def distinct_n(texts: Iterable[str], n: int) -> float:
    """Unique n-grams / total n-grams over a set of texts."""
    grams: Counter = Counter()
    for text in texts:
        tokens = normalize(text).split()
        grams.update(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))
    total = sum(grams.values())
    return len(grams) / total if total else 0.0


def unique_ratio(texts: Iterable[str]) -> float:
    """Fraction of non-empty texts that are distinct (1.0 = no repeats)."""
    kept = [normalize(text) for text in texts]
    kept = [text for text in kept if text]
    if not kept:
        return 0.0
    return len(set(kept)) / len(kept)


def _sentence_bleu(hypothesis: str, references: list[str], *, n: int = 4) -> float:
    """Clip n-gram precision with a brevity penalty, against several references.

    Questions are short, so the order is capped at the hypothesis length (a 3-word
    question is still scored, on 1- to 3-grams, instead of collapsing to 0 on 4-grams).
    """
    hyp = normalize(hypothesis).split()
    refs = [normalize(ref).split() for ref in references]
    refs = [ref for ref in refs if ref]
    if not hyp or not refs:
        return 0.0
    max_n = min(n, len(hyp))
    precisions: list[float] = []
    for k in range(1, max_n + 1):
        hyp_grams = Counter(tuple(hyp[i : i + k]) for i in range(len(hyp) - k + 1))
        best: Counter = Counter()
        for ref in refs:
            for gram, count in Counter(
                tuple(ref[i : i + k]) for i in range(len(ref) - k + 1)
            ).items():
                best[gram] = max(best[gram], count)
        clipped = sum(min(count, best[gram]) for gram, count in hyp_grams.items())
        precisions.append(clipped / sum(hyp_grams.values()))
    if min(precisions) <= 0:
        return 0.0
    hyp_len = len(hyp)
    ref_len = min((len(ref) for ref in refs), key=lambda size: abs(size - hyp_len))
    penalty = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / hyp_len)
    return penalty * math.exp(sum(math.log(p) for p in precisions) / max_n)


def self_bleu(texts: list[str], *, n: int = 4, limit: int = 200) -> float:
    """Mean sentence-BLEU of each question against all others; high = mode collapse."""
    kept = [text for text in texts if normalize(text)]
    if len(kept) < 2:
        return 0.0
    kept = kept[:limit]
    scores = [
        _sentence_bleu(text, kept[:i] + kept[i + 1 :], n=n) for i, text in enumerate(kept)
    ]
    return statistics.fmean(scores)


# --------------------------------------------------------------------------- #
# Format (offline)
# --------------------------------------------------------------------------- #
def is_well_formed(question: Any, *, min_words: int = 3, max_words: int = 40) -> bool:
    """A usable question: non-empty, asks something, sane length, no leaked thinking."""
    text = str(question or "").strip()
    if not text.endswith("?"):
        return False
    if "thinking process" in text.lower() or "here is a" in text.lower():
        return False
    return min_words <= len(text.split()) <= max_words


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return round(statistics.fmean(values), 4) if values else 0.0


def compute_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Offline quality metrics over one ``predictions.json`` (no model needed)."""
    predictions = [str(record.get("prediction") or "") for record in records]
    references = [str(record.get("target") or "") for record in records]
    categories = [
        str((record.get("meta") or {}).get("category") or "unknown") for record in records
    ]

    def block(preds: list[str], refs: list[str]) -> dict[str, Any]:
        return {
            "n": len(preds),
            "empty_rate": _mean([float(not pred.strip()) for pred in preds]),
            "well_formed_rate": _mean([float(is_well_formed(pred)) for pred in preds]),
            "mean_words": _mean([float(len(pred.split())) for pred in preds]),
            "unique_ratio": round(unique_ratio(preds), 4),
            "rouge_l": _mean([rouge_l(p, r) for p, r in zip(preds, refs, strict=True)]),
            "chrf": _mean([chrf(p, r) for p, r in zip(preds, refs, strict=True)]),
        }

    metrics = {
        **block(predictions, references),
        "distinct_1": round(distinct_n(predictions, 1), 4),
        "distinct_2": round(distinct_n(predictions, 2), 4),
        "self_bleu": round(self_bleu(predictions), 4),
        "by_category": {},
    }
    for category in sorted(set(categories)):
        rows = [i for i, name in enumerate(categories) if name == category]
        metrics["by_category"][category] = block(
            [predictions[i] for i in rows], [references[i] for i in rows]
        )
    return metrics


# --------------------------------------------------------------------------- #
# Round-trip answerability (needs a model)
# --------------------------------------------------------------------------- #
ANSWER_INSTRUCTION = (
    "You are shown a screenshot of a web page. Answer the question using only what is "
    "visible in the screenshot. Reply with a short, direct answer.\n"
    "Question: {question}\nAnswer:"
)


def round_trip(
    records: list[dict[str, Any]],
    model: Any,
    *,
    use_reference: bool = False,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Ask ``model`` the generated (or reference) question and score its answer."""
    rounds: list[dict[str, Any]] = []
    for record in records[:limit] if limit else records:
        question = str(record.get("target") if use_reference else record.get("prediction") or "")
        answered = model.generate(
            ANSWER_INSTRUCTION.format(question=question), image=record.get("image")
        )
        gold = record.get("answer")
        rounds.append(
            {
                "id": record.get("id"),
                "category": (record.get("meta") or {}).get("category"),
                "question": question,
                "gold": gold,
                "answered": answered,
                "image": record.get("image"),
                "em": exact_match(answered, gold),
                "containment": containment(answered, gold),
                "f1": token_f1(answered, gold),
            }
        )
    return rounds


def answer_scores(rounds: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate round-trip rounds: EM / containment / F1, overall and per category."""

    def block(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "n": len(rows),
            "em": _mean([row["em"] for row in rows]),
            "containment": _mean([row["containment"] for row in rows]),
            "f1": _mean([row["f1"] for row in rows]),
        }

    scores: dict[str, Any] = block(rounds)
    scores["by_category"] = {
        category: block([row for row in rounds if row["category"] == category])
        for category in sorted({row["category"] for row in rounds})
    }
    return scores


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def predictions_path(split: str) -> Path:
    """Where ``m2w-eval`` writes its output for a split."""
    return OUTPUT_DIR / f"eval_{split}" / "predictions.json"


def main(argv: list[str] | None = None) -> int:
    ensure_utf8()
    parser = argparse.ArgumentParser(description="Question-quality metrics for an eval run.")
    parser.add_argument("--split", default="test_task")
    parser.add_argument(
        "--predictions", type=Path, default=None, help="default: outputs/eval_<split>/..."
    )
    parser.add_argument("--out", type=Path, default=None, help="default: next to predictions")
    parser.add_argument(
        "--round-trip",
        action="store_true",
        help="also ask the model the generated questions (needs weights; slow)",
    )
    parser.add_argument("--model", default="vlm/Qwen/Qwen3.5-9B", help="model for --round-trip")
    parser.add_argument("--limit", type=int, default=None, help="cap round-trip records")
    args = parser.parse_args(argv)

    path = args.predictions or predictions_path(args.split)
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    metrics = compute_metrics(records)

    if args.round_trip:
        from mind2web_vqg.models import build_model

        model = build_model(args.model)
        metrics["round_trip"] = {
            "prediction": answer_scores(round_trip(records, model, limit=args.limit)),
            "reference": answer_scores(
                round_trip(records, model, use_reference=True, limit=args.limit)
            ),
        }

    out = args.out or Path(path).with_name("metrics.json")
    Path(out).write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[metrics] {metrics['n']} predictions -> {out}")
    print(
        f"  well-formed {metrics['well_formed_rate']:.2f} | unique {metrics['unique_ratio']:.2f}"
        f" | self-BLEU {metrics['self_bleu']:.2f} | ROUGE-L {metrics['rouge_l']:.2f}"
        f" | chrF {metrics['chrf']:.2f}"
    )
    if "round_trip" in metrics:
        predicted = metrics["round_trip"]["prediction"]
        ceiling = metrics["round_trip"]["reference"]
        print(
            f"  answer EM {predicted['em']:.2f} (ceiling {ceiling['em']:.2f})"
            f" | F1 {predicted['f1']:.2f} (ceiling {ceiling['f1']:.2f})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
