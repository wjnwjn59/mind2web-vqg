"""Tests for the bench question-generation runner and its offline scores.

No model, no GPU, no dataset: a fake backend stands in for Qwen and every row is
hand-written, so the whole file runs in milliseconds.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mind2web_vqg.treevqg import bench_eval
from mind2web_vqg.treevqg.bench_eval import (
    answer_mentioned,
    build_prompt,
    condition_body,
    generate,
    is_question,
    normalize,
    read_predictions,
    resolve_model_name,
    rouge_l,
    run_eval,
    score_predictions,
    select_rows,
    token_f1,
    tokens,
)

CONDITION = (
    "You are building ONE question about the web page shown by the UI tree below. "
    "The node marked with * is the element the question must be about. "
    "Output only the question.\n\n"
    "User's task: find a car to pick up at SFO.\n\n"
    "UI tree:\n  - <button> button \"Enter your pick-up location\" *"
)


def _row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": "abc:0",
        "type": "container_membership",
        "condition": CONDITION,
        "input": "Task: find a car.\nQuestion:",
        "question": "Which section of the page contains the pick-up location field?",
        "answer": "Customer ID",
        "image": None,
        "meta": {"requires_tree": True, "website": "cars.example"},
    }
    row.update(overrides)
    return row


class FakeModel:
    """Records prompts and returns a canned question; no torch import."""

    def __init__(self, reply: str = "Which section holds the pick-up location field?") -> None:
        self.reply = reply
        self.calls: list[tuple[str, Any]] = []

    def generate(self, prompt: str, image: Any = None) -> str:
        self.calls.append((prompt, image))
        return self.reply


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #
def test_condition_body_drops_the_generic_preamble() -> None:
    body = condition_body(CONDITION)
    assert not body.startswith("You are building ONE question")
    assert body.startswith("User's task: find a car")


def test_condition_body_keeps_text_without_a_preamble() -> None:
    assert condition_body("Task: x\nTree: y") == "Task: x\nTree: y"
    assert condition_body(None) == ""


def test_build_prompt_uses_the_type_instruction_and_hides_the_gold_question() -> None:
    prompt = build_prompt(_row())
    assert bench_eval.TYPE_PROMPTS["container_membership"] in prompt
    assert "Which section of the page contains the pick-up location field?" not in prompt
    assert "You are building ONE question" not in prompt
    assert "User's task: find a car to pick up at SFO." in prompt
    assert prompt.endswith('*')


def test_build_prompt_mentions_the_screenshot_only_when_used() -> None:
    assert "screenshot" not in build_prompt(_row()).lower()
    assert "screenshot" in build_prompt(_row(), with_image=True).lower()


def test_build_prompt_falls_back_to_input() -> None:
    prompt = build_prompt(_row(condition="", input="Task: find a car.\nQuestion:"))
    assert "Task: find a car." in prompt


def test_build_prompt_handles_unknown_types() -> None:
    prompt = build_prompt(_row(type="mystery"))
    assert bench_eval.GENERIC_PROMPT in prompt


# --------------------------------------------------------------------------- #
# Matching / scoring
# --------------------------------------------------------------------------- #
def test_normalize_and_tokens_are_accent_and_hyphen_insensitive() -> None:
    assert normalize("  Top–Left\nRésumé ") == "top-left resume"
    assert tokens("top-left") == ["top", "left"]


def test_token_f1_bounds() -> None:
    gold = "Which section contains the button?"
    assert token_f1(gold, gold) == 1.0
    assert token_f1("Totally unrelated words here", gold) == 0.0
    assert 0.0 < token_f1("Which button?", gold) < 1.0


def test_rouge_l_rewards_shared_subsequence() -> None:
    gold = "Which section of the page contains the pick-up field?"
    assert rouge_l(gold, gold) == 1.0
    assert rouge_l("", gold) == 0.0
    partial = rouge_l("Which section contains the pick-up field?", gold)
    assert 0.0 < partial < 1.0


def test_answer_mentioned_matches_words_not_substrings() -> None:
    assert answer_mentioned("Which section contains the Customer ID field?", "Customer ID")
    assert answer_mentioned("Is it in the top left?", "top-left")
    assert not answer_mentioned("Is the button in the north panel?", "No")
    assert answer_mentioned("Does it contain a search box? (yes)", "Yes")
    assert answer_mentioned("anything", "") is None


def test_is_question() -> None:
    assert is_question("Which section?")
    assert not is_question("The section is Customer ID.")


def test_select_rows_filters_and_caps_per_type() -> None:
    rows = [
        _row(id=f"a:{i}", type="absence") for i in range(4)
    ] + [_row(id=f"b:{i}", type="target_label") for i in range(2)]
    picked = select_rows(rows, types=["absence"], per_type=2)
    assert [row["id"] for row in picked] == ["a:0", "a:1"]
    assert len(select_rows(rows, per_type=3)) == 5
    assert len(select_rows(rows, limit=4)) == 4


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def test_score_predictions_aggregates_per_type_and_track() -> None:
    rows = [
        _row(id="a:0", type="container_membership"),
        _row(id="a:1", type="target_region", answer="top-left", meta={"requires_tree": False}),
        _row(id="a:2", type="target_region", answer="top-left", meta={"requires_tree": False}),
    ]
    predictions = {
        "a:0": {"prediction": "Which section contains the Customer ID field?"},
        "a:1": {"prediction": "In which region is the field?"},
    }
    evaluation = score_predictions(rows, predictions)

    assert evaluation["rows"] == 3
    assert evaluation["predicted"] == 2
    assert evaluation["by_type"]["container_membership"]["n"] == 1
    assert evaluation["by_type"]["container_membership"]["answer_mention"] == 1.0
    assert evaluation["by_type"]["target_region"]["n"] == 2
    assert evaluation["by_type"]["target_region"]["answer_mention"] == 0.0
    assert evaluation["by_track"]["tree_yes"]["n"] == 1
    assert evaluation["by_track"]["tree_no"]["n"] == 2
    assert any("no prediction yet" in warning for warning in evaluation["warnings"])


def test_score_predictions_flags_non_questions() -> None:
    rows = [_row(id="a:0")]
    evaluation = score_predictions(rows, {"a:0": {"prediction": "Customer ID."}})
    assert evaluation["totals"]["asks_rate"] == 0.0
    assert any("contain '?'" in warning for warning in evaluation["warnings"])


def test_empty_selection_is_reported() -> None:
    evaluation = score_predictions([], {})
    assert evaluation["rows"] == 0
    assert evaluation["warnings"] == [
        "no rows selected: check --types / --per-type and the examples path"
    ]


# --------------------------------------------------------------------------- #
# Generation loop
# --------------------------------------------------------------------------- #
def test_generate_writes_one_line_per_row_and_resumes(tmp_path: Path) -> None:
    rows = [_row(id=f"a:{i}") for i in range(3)]
    model = FakeModel()

    path = generate(rows, out_dir=tmp_path, model=model, model_name="vlm/fake")
    assert len(model.calls) == 3
    assert len(read_predictions(path)) == 3
    assert read_predictions(path)["a:1"]["prediction"].startswith("Which section")

    generate(rows, out_dir=tmp_path, model=model, model_name="vlm/fake")
    assert len(model.calls) == 3  # nothing re-run

    generate(rows, out_dir=tmp_path, model=model, model_name="vlm/fake", resume=False)
    assert len(model.calls) == 6  # --no-resume rewrites everything


def test_generate_falls_back_to_text_when_the_screenshot_is_gone(tmp_path: Path) -> None:
    rows = [_row(id="a:0", image=str(tmp_path / "missing.jpg"))]
    model = FakeModel()
    generate(rows, out_dir=tmp_path, model=model, model_name="vlm/fake", with_image=True)
    prompt, image = model.calls[0]
    assert image is None
    assert "screenshot" not in prompt.lower()


def test_run_eval_dry_run_needs_no_model(tmp_path: Path) -> None:
    examples = tmp_path / "examples.jsonl"
    examples.write_text(json.dumps(_row()) + "\n", encoding="utf-8")
    result = run_eval(examples, model_name="vlm/fake", dry_run=True)
    assert result["dry_run"] is True
    assert result["rows"] == 1


def test_run_eval_writes_metrics_and_report(tmp_path: Path) -> None:
    examples = tmp_path / "examples.jsonl"
    examples.write_text(
        "\n".join(json.dumps(_row(id=f"a:{i}")) for i in range(2)) + "\n", encoding="utf-8"
    )
    out_dir = tmp_path / "run"
    evaluation = run_eval(
        examples,
        model_name="vlm/fake",
        out_dir=out_dir,
        model=FakeModel(),
    )
    assert evaluation["totals"]["n"] == 2
    assert (out_dir / "predictions.jsonl").is_file()
    assert (out_dir / "metrics.json").is_file()
    report = (out_dir / "report.md").read_text(encoding="utf-8")
    assert "TreeVQG-Bench v1" in report
    assert "container_membership" in report


def test_run_eval_explains_a_missing_bench(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="m2w-build-bench"):
        run_eval(tmp_path / "nope.jsonl")


def test_resolve_model_name_defaults_to_vlm() -> None:
    assert resolve_model_name("Qwen/Qwen2.5-VL-7B-Instruct") == "vlm/Qwen/Qwen2.5-VL-7B-Instruct"
    assert resolve_model_name("api/gpt-4o-mini") == "api/gpt-4o-mini"
