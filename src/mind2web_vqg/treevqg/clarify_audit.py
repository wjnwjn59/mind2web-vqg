"""Audit the H2 clarification dataset before using it as a benchmark.

The clarification track has a different failure surface from the first TreeVQG pass:
the answer is *supposed* to be one of the options named in the question, so the old
"answer in question" leak check does not apply. What matters instead is:

* every positive row has a valid ``answer_index`` and an answer that equals the
  candidate field it points at (container / role / region);
* the question names *all* options, otherwise the user cannot pick the gold one;
* negatives are exactly ``NO_QUESTION`` and positives never are;
* referents are pronounceable (no URL / bare-digit containers);
* the same template is not 99% of the corpus.

Writes ``audit.json`` + ``audit.md``. Usage::

    uv run m2w-audit-clarify --split test_task
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from mind2web_vqg.cli import ensure_utf8
from mind2web_vqg.paths import OUTPUT_DIR, PROCESSED_DIR
from mind2web_vqg.treevqg import io as tree_io
from mind2web_vqg.treevqg.audit import normalize, skeleton
from mind2web_vqg.treevqg.clarify import NO_QUESTION, is_weak_container, is_weak_label

_QUOTED = re.compile(r'"[^"]*"')


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 3) if values else 0.0


def _top(counter: Counter[str], k: int = 10) -> list[dict[str, Any]]:
    return [{"value": value, "count": count} for value, count in counter.most_common(k)]


def _track(row: dict[str, Any]) -> str:
    return str(row.get("meta", {}).get("track") or row.get("track") or "?")


def _ambiguity(row: dict[str, Any]) -> str:
    return str(row.get("meta", {}).get("ambiguity_type") or "?")


def options_for_row(row: dict[str, Any]) -> list[str]:
    """The candidate labels the user has to choose between (empty for safety)."""
    ambiguity = _ambiguity(row)
    candidates = row.get("candidates") or []
    if ambiguity in {"container", "role", "region"}:
        return [str(candidate.get(ambiguity) or "") for candidate in candidates]
    return []


def answer_matches(row: dict[str, Any]) -> bool | None:
    """Does ``answer`` equal the field selected by ``answer_index``?"""
    ambiguity = _ambiguity(row)
    if ambiguity not in {"container", "role", "region"}:
        return None
    candidates = row.get("candidates") or []
    index = row.get("answer_index")
    if not isinstance(index, int) or not 0 <= index < len(candidates):
        return False
    return normalize(candidates[index].get(ambiguity)) == normalize(row.get("answer"))


def option_coverage(row: dict[str, Any]) -> float | None:
    """Fraction of the candidate options literally named in the question."""
    options = [option for option in options_for_row(row) if normalize(option)]
    if not options:
        return None
    question = normalize(row.get("target"))
    return sum(normalize(option) in question for option in options) / len(options)


def input_option_coverage(row: dict[str, Any]) -> float | None:
    """Fraction of referents visible in the prompt.

    Region labels are derived from bboxes and intentionally excluded: they belong to
    the screenshot, not to the DOM outline. Container and role labels must be visible,
    otherwise the model cannot discover the ambiguity in the first place.
    """
    if _ambiguity(row) not in {"container", "role"}:
        return None
    options = [option for option in options_for_row(row) if normalize(option)]
    if not options:
        return None
    prompt = normalize(row.get("input"))
    return sum(normalize(option) in prompt for option in options) / len(options)


def audit_rows(rows: list[dict[str, Any]], *, samples: int = 2) -> dict[str, Any]:
    if not rows:
        raise ValueError("no clarification rows to audit")

    tracks = Counter(_track(row) for row in rows)
    levels = Counter(str(row.get("level") or "?") for row in rows)
    ambiguities = Counter(_ambiguity(row) for row in rows)
    websites = Counter(str(row.get("meta", {}).get("website") or "?") for row in rows)
    answers = Counter(
        str(row.get("answer")) for row in rows if row.get("meta", {}).get("should_ask")
    )
    positives = [row for row in rows if row.get("meta", {}).get("should_ask")]
    negatives = [row for row in rows if not row.get("meta", {}).get("should_ask")]

    checks: dict[str, dict[str, Any]] = {}

    def flag(name: str, row: dict[str, Any]) -> None:
        block = checks.setdefault(name, {"count": 0, "examples": []})
        block["count"] += 1
        if len(block["examples"]) < 3:
            block["examples"].append(str(row.get("id")))

    for row in rows:
        should_ask = bool(row.get("meta", {}).get("should_ask"))
        target = row.get("target")
        if not target:
            flag("missing_target", row)
        if should_ask:
            if target == NO_QUESTION:
                flag("positive_is_no_question", row)
            if row.get("answer") in (None, ""):
                flag("missing_answer", row)
        else:
            if target != NO_QUESTION:
                flag("negative_not_no_question", row)

        candidates = row.get("candidates") or []
        if int(row.get("meta", {}).get("n_candidates") or 0) != len(candidates):
            flag("n_candidates_mismatch", row)
        nodes = [candidate.get("node") for candidate in candidates]
        if len(nodes) != len(set(nodes)):
            flag("duplicate_candidate_nodes", row)

        ambiguity = _ambiguity(row)
        if ambiguity in {"container", "role", "region"}:
            index = row.get("answer_index")
            if not isinstance(index, int) or not 0 <= index < len(candidates):
                flag("invalid_answer_index", row)
            elif not answer_matches(row):
                flag("answer_index_mismatch", row)
            coverage = option_coverage(row)
            if coverage is not None and coverage < 1.0:
                flag("option_not_in_question", row)
            visible = input_option_coverage(row)
            if visible is not None and visible < 1.0:
                flag("referents_not_in_prompt", row)
            if ambiguity == "container" and any(
                is_weak_container(str(candidate.get("container") or "")) for candidate in candidates
            ):
                flag("weak_container", row)
            label = str(row.get("meta", {}).get("anchor_label") or "")
            if is_weak_label(label):
                flag("weak_anchor_label", row)

    coverages = [value for row in rows if (value := option_coverage(row)) is not None]
    input_coverages = [value for row in rows if (value := input_option_coverage(row)) is not None]
    target_in_input = sum(
        1
        for row in positives
        if str(row.get("target") or "").casefold() in str(row.get("input") or "").casefold()
    )
    images_missing = sum(
        1 for row in rows if not (row.get("image") and Path(str(row["image"])).is_file())
    )
    skeletons = Counter(
        skeleton(row.get("target"), row.get("meta", {}).get("task"), row.get("answer"))
        for row in positives
    )
    track_blocks: dict[str, Any] = {}
    for track in tracks:
        block = [row for row in rows if _track(row) == track]
        gains = [float(row.get("meta", {}).get("info_gain_bits") or 0) for row in block]
        candidates = [len(row.get("candidates") or []) for row in block]
        track_blocks[track] = {
            "n": len(block),
            "share": round(len(block) / len(rows), 3),
            "mean_candidates": _mean([float(value) for value in candidates]),
            "mean_info_gain_bits": _mean(gains),
            "total_info_gain_bits": round(sum(gains), 2),
        }

    audit = {
        "totals": {
            "examples": len(rows),
            "actions": len(
                {str(row.get("meta", {}).get("source_id") or row.get("id")) for row in rows}
            ),
            "tasks": len(
                {
                    str(row.get("meta", {}).get("annotation_id"))
                    for row in rows
                    if row.get("meta", {}).get("annotation_id")
                }
            ),
            "websites": len(websites),
            "positives": len(positives),
            "negatives": len(negatives),
            "positive_rate": round(len(positives) / len(rows), 3),
            "mean_candidates": _mean(
                [float(len(row.get("candidates") or [])) for row in positives]
            ),
            "mean_info_gain_bits": _mean(
                [float(row.get("meta", {}).get("info_gain_bits") or 0) for row in positives]
            ),
            "total_info_gain_bits": round(
                sum(float(row.get("meta", {}).get("info_gain_bits") or 0) for row in positives), 2
            ),
            "mean_option_coverage": _mean(coverages),
            "full_option_coverage_rate": round(
                sum(value == 1.0 for value in coverages) / len(coverages), 3
            )
            if coverages
            else 1.0,
            "mean_input_option_coverage": _mean(input_coverages),
            "full_input_option_coverage_rate": round(
                sum(value == 1.0 for value in input_coverages) / len(input_coverages), 3
            )
            if input_coverages
            else 1.0,
            "target_in_input": target_in_input,
            "images_missing": images_missing,
            "unique_answers": len(answers),
            "top_website_share": round(websites.most_common(1)[0][1] / len(rows), 3),
            "top_answer_share": round(answers.most_common(1)[0][1] / len(answers), 3)
            if answers
            else 0.0,
        },
        "tracks": dict(tracks),
        "levels": dict(levels),
        "ambiguities": dict(ambiguities),
        "by_track": track_blocks,
        "option_coverage": {
            "mean": _mean(coverages),
            "full_rate": round(sum(value == 1.0 for value in coverages) / len(coverages), 3)
            if coverages
            else 1.0,
        },
        "checks": checks,
        "top_answers": _top(answers, 10),
        "top_websites": _top(websites, 10),
        "skeletons": {
            "n": len(skeletons),
            "unique_rate": round(len(skeletons) / len(positives), 3) if positives else 0.0,
            "max_share": round(skeletons.most_common(1)[0][1] / len(positives), 3)
            if positives
            else 0.0,
            "top": _top(skeletons, 5),
        },
        "samples": _samples(rows, samples),
        "warnings": [],
    }
    audit["warnings"] = build_warnings(audit)
    return audit


def _samples(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str]] = Counter()
    picked: list[dict[str, Any]] = []
    for row in rows:
        key = (_track(row), _ambiguity(row))
        if counts[key] >= limit:
            continue
        counts[key] += 1
        picked.append(
            {
                "id": row.get("id"),
                "track": _track(row),
                "ambiguity": _ambiguity(row),
                "website": row.get("meta", {}).get("website"),
                "task": row.get("meta", {}).get("task"),
                "question": row.get("target"),
                "answer": row.get("answer"),
                "candidates": [
                    {
                        "role": candidate.get("role"),
                        "container": candidate.get("container"),
                        "region": candidate.get("region"),
                    }
                    for candidate in (row.get("candidates") or [])
                ],
            }
        )
    return picked


def build_warnings(audit: dict[str, Any]) -> list[str]:
    totals = audit["totals"]
    warnings: list[str] = []
    if totals["positives"] < 100:
        warnings.append(
            f"only {totals['positives']} positive rows: below the ~100 needed for a stable "
            "per-level test split"
        )
    if totals["images_missing"]:
        warnings.append(f"{totals['images_missing']} row(s) are missing their screenshot on disk")
    if totals["target_in_input"]:
        warnings.append(
            f"{totals['target_in_input']} positive row(s) contain the whole gold question in "
            "the prompt"
        )
    if totals["full_option_coverage_rate"] < 1.0:
        warnings.append(
            f"only {totals['full_option_coverage_rate']:.0%} of questions name every option; "
            "the rest are not answerable as written"
        )
    if totals["full_input_option_coverage_rate"] < 1.0:
        warnings.append(
            f"only {totals['full_input_option_coverage_rate']:.0%} of container/role rows show "
            "every referent in the prompt; those ambiguities are invisible to the model"
        )
    if totals["top_website_share"] > 0.2:
        warnings.append(f"one website holds {totals['top_website_share']:.0%} of the rows (>20%)")
    if totals["top_answer_share"] > 0.3:
        warnings.append(
            f"one answer accounts for {totals['top_answer_share']:.0%} of positives; "
            "check for a degenerate target"
        )
    for name, block in sorted(audit["checks"].items()):
        warnings.append(
            f"{block['count']} row(s) fail `{name}` (e.g. {', '.join(block['examples'])})"
        )
    if audit["skeletons"]["max_share"] > 0.9 and audit["skeletons"]["n"] > 1:
        warnings.append(
            f"one question skeleton is {audit['skeletons']['max_share']:.0%} of positives: "
            "template collapse (expected for template gold, but a baseline must not copy it)"
        )
    return warnings


def render_markdown(audit: dict[str, Any], *, split: str) -> str:
    totals = audit["totals"]
    lines = [
        f"# TreeVQG-Clarify audit â€” `{split}`",
        "",
        "H2 clarification questions: the answer is user intent, so the standard "
        "answer-in-prompt leakage check does not apply. This report checks instead "
        "that each question is answerable from its own options and that the gold "
        "answer points at the intended candidate.",
        "",
        "## Headline",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| examples | {totals['examples']:,} |",
        (
            f"| actions / tasks / websites | {totals['actions']:,} / "
            f"{totals['tasks']:,} / {totals['websites']:,} |"
        ),
        f"| positives / negatives | {totals['positives']:,} / {totals['negatives']:,} |",
        f"| positive rate | {totals['positive_rate']:.1%} |",
        f"| mean candidates (positive) | {totals['mean_candidates']:.2f} |",
        f"| mean info gain (positive) | {totals['mean_info_gain_bits']:.3f} bits |",
        f"| total info gain | {totals['total_info_gain_bits']:.1f} bits |",
        f"| full option coverage | {totals['full_option_coverage_rate']:.1%} |",
        (f"| referents visible in prompt | {totals['full_input_option_coverage_rate']:.1%} |"),
        f"| unique answers | {totals['unique_answers']:,} |",
        f"| missing screenshots | {totals['images_missing']:,} |",
        "",
        "## Rows by track",
        "",
        "| track | n | share | candidates | info gain |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for track, block in sorted(audit["by_track"].items()):
        lines.append(
            f"| {track} | {block['n']:,} | {block['share']:.1%} | "
            f"{block['mean_candidates']:.2f} | {block['mean_info_gain_bits']:.3f} |"
        )
    lines += [
        "",
        "## Levels and ambiguity types",
        "",
        "- levels: "
        + ", ".join(f"{key}={value}" for key, value in sorted(audit["levels"].items())),
        "- ambiguity: "
        + ", ".join(f"{key}={value}" for key, value in sorted(audit["ambiguities"].items())),
        "",
        "## Quality checks",
        "",
        "| check | rows | examples |",
        "| --- | ---: | --- |",
    ]
    if audit["checks"]:
        for name, block in sorted(audit["checks"].items()):
            lines.append(f"| {name} | {block['count']} | {', '.join(block['examples'])} |")
    else:
        lines.append("| â€” | 0 | â€” |")
    lines += [
        "",
        "## Warnings",
        "",
    ]
    if audit["warnings"]:
        lines.extend(f"- {warning}" for warning in audit["warnings"])
    else:
        lines.append("- none")
    lines += [
        "",
        "## Option coverage",
        "",
        f"Mean fraction of candidates named in the question: "
        f"**{audit['option_coverage']['mean']:.1%}**; "
        f"questions naming every option: **{audit['option_coverage']['full_rate']:.1%}**.",
        "",
        "## Question skeletons",
        "",
        f"{audit['skeletons']['n']} distinct skeleton(s); the most common covers "
        f"{audit['skeletons']['max_share']:.1%} of positives.",
        "",
    ]
    for item in audit["skeletons"]["top"][:3]:
        lines.append(f"- `{item['value']}` â€” {item['count']} row(s)")
    lines += ["", "## Top answers", ""]
    for item in audit["top_answers"][:8]:
        lines.append(f"- `{item['value']}` â€” {item['count']} row(s)")
    lines += ["", "## Samples", ""]
    for sample in audit["samples"][:16]:
        lines.append(f"### `{sample['track']}` / `{sample['ambiguity']}` â€” {sample['website']}")
        lines.append("")
        lines.append(f"- task: {sample['task']}")
        lines.append(f"- question: **{sample['question']}**")
        lines.append(f"- answer: `{sample['answer']}`")
        if sample["candidates"]:
            lines.append(
                "- candidates: "
                + "; ".join(
                    f"{candidate['role']} in {candidate['container']!r} ({candidate['region']})"
                    for candidate in sample["candidates"]
                )
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def audit_split(
    split: str = "test_task",
    *,
    examples_path: str | Path | None = None,
    out_dir: str | Path | None = None,
    samples: int = 2,
) -> dict[str, Any]:
    path = (
        Path(examples_path)
        if examples_path
        else PROCESSED_DIR / "clarify" / split / "examples.jsonl"
    )
    rows = tree_io.read_jsonl(path)
    audit = audit_rows(rows, samples=samples)
    dest = Path(out_dir) if out_dir else OUTPUT_DIR / "clarify_audit" / split
    tree_io.write_json(audit, dest / "audit.json")
    (dest / "audit.md").write_text(render_markdown(audit, split=split), encoding="utf-8")
    return {"audit": audit, "json": dest / "audit.json", "markdown": dest / "audit.md"}


def main(argv: list[str] | None = None) -> int:
    ensure_utf8()
    parser = argparse.ArgumentParser(description="Audit the TreeVQG-Clarify dataset.")
    parser.add_argument("--split", default="test_task")
    parser.add_argument("--examples", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--samples", type=int, default=2)
    args = parser.parse_args(argv)

    result = audit_split(
        args.split, examples_path=args.examples, out_dir=args.out, samples=args.samples
    )
    audit = result["audit"]
    totals = audit["totals"]
    print(
        f"[clarify-audit] {totals['examples']} row(s): "
        f"{totals['positives']} positive / {totals['negatives']} negative; "
        f"option coverage {totals['full_option_coverage_rate']:.0%}; "
        f"{len(audit['warnings'])} warning(s)"
    )
    for warning in audit["warnings"][:8]:
        print(f"  ! {warning}")
    print(f"[clarify-audit] -> {result['markdown']}")
    print(f"[clarify-audit] -> {result['json']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
