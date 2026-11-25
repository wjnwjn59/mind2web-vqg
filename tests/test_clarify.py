"""Tests for the H2 clarification builder (no network, no images)."""

from __future__ import annotations

import json
from typing import Any

from mind2web_vqg.treevqg.clarify import (
    NO_QUESTION,
    build_rows_for_action,
    humanize_action,
)
from mind2web_vqg.treevqg.clarify_audit import (
    audit_rows,
    input_option_coverage,
    option_coverage,
    render_markdown,
)
from mind2web_vqg.treevqg.clarify_metrics import (
    ask_prf,
    evaluate_rows,
    info_gain_bits,
    option_recall,
)


def source_row(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "task1:0",
        "annotation_id": "task1",
        "action_uid": "uid1",
        "split": "test_task",
        "task": "Add the cheapest item to the cart",
        "website": "shop",
        "domain": "shopping",
        "action_history": [],
        "target_action": "",
        "op": "CLICK",
        "value": "",
        "image": "images/task1_00.jpg",
        "image_size": [1280, 1000],
        "source_size": [1280, 1000],
    }
    base.update(over)
    return base


def action(html: str, *, target_id: str = "2", target_action: str = "") -> dict[str, Any]:
    return {
        "action_uid": "uid1",
        "cleaned_html": html,
        "pos_candidates": [
            {
                "tag": "button",
                "backend_node_id": target_id,
                "is_original_target": True,
                "attributes": json.dumps(
                    {"backend_node_id": target_id, "bounding_box_rect": "100,100,40,20"}
                ),
            }
        ],
        "neg_candidates": [],
        "target_action_reprs": target_action,
    }


TWO_CARDS = """
<html><body>
  <main>
    <div aria-label="Recommended">
      <button backend_node_id="1">Add</button>
    </div>
    <div aria-label="Recently viewed">
      <button backend_node_id="2">Add</button>
    </div>
  </main>
</body></html>
"""

TWO_ROLES = """
<html><body>
  <main>
    <button backend_node_id="1">Search</button>
    <a href="/search" backend_node_id="2">Search</a>
  </main>
</body></html>
"""

SAME_CONTAINER = """
<html><body>
  <div aria-label="Results">
    <button backend_node_id="1">Add</button>
    <button backend_node_id="2">Add</button>
  </div>
</body></html>
"""

UNIQUE = """
<html><body>
  <main><button backend_node_id="2">Apply filter</button></main>
</body></html>
"""


def test_container_ambiguity_is_l2_and_the_answer_is_the_container() -> None:
    rows, skipped = build_rows_for_action(
        source_row(target_action="[button]  Add -> CLICK"),
        action(TWO_CARDS),
        split="test_task",
    )
    assert skipped == []
    assert len(rows) == 1
    row = rows[0]
    assert row["level"] == "L2"
    assert row["track"] == "ambiguity"
    assert row["meta"]["ambiguity_type"] == "container"
    assert row["meta"]["n_candidates"] == 2
    assert row["meta"]["info_gain_bits"] == 1.0
    assert row["answer"] == "Recently viewed"
    assert row["answer_index"] == 1
    assert "Recently viewed" in row["target"]
    assert "Recommended" in row["target"]
    # The prompt must not hand the model the gold question or the candidate list.
    assert row["target"] not in row["input"]
    assert "Which one do you mean" not in row["input"]
    # It does get the intended action, but without the tag or the container.
    assert 'Planned action: click "Add"' in row["input"]


def test_same_label_different_roles_is_l1() -> None:
    rows, _ = build_rows_for_action(source_row(), action(TWO_ROLES), split="test_task")
    assert len(rows) == 1
    row = rows[0]
    assert row["level"] == "L1"
    assert row["meta"]["ambiguity_type"] == "role"
    assert row["answer"] == "link"
    assert row["meta"]["n_candidates"] == 2


def test_indistinguishable_duplicates_are_skipped_not_negatives() -> None:
    rows, skipped = build_rows_for_action(source_row(), action(SAME_CONTAINER), split="test_task")
    assert rows == []
    assert "indistinguishable_candidates" in skipped


def test_unique_target_is_a_no_question_negative() -> None:
    rows, skipped = build_rows_for_action(
        source_row(target_action="[button]  Apply filter -> CLICK"),
        action(UNIQUE),
        split="test_task",
    )
    assert skipped == []
    assert len(rows) == 1
    row = rows[0]
    assert row["track"] == "negative"
    assert row["target"] == NO_QUESTION
    assert row["meta"]["should_ask"] is False
    assert row["answer"] is None


def test_irreversible_action_adds_a_safety_row() -> None:
    rows, _ = build_rows_for_action(
        source_row(target_action="[button]  Delete account -> CLICK"),
        action(UNIQUE, target_action="[button]  Delete account -> CLICK"),
        split="test_task",
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["level"] == "L3"
    assert row["track"] == "safety"
    assert row["answer"] == "yes"
    assert 'click "Delete account"' in row["target"]
    assert row["meta"]["info_gain_bits"] == 0.0


def test_humanize_action_reads_type_and_select() -> None:
    assert humanize_action("[button]  Delete -> CLICK") == 'click "Delete"'
    assert humanize_action("[textbox]  Mileage -> TYPE: 100") == 'type "100" into "Mileage"'
    assert (
        humanize_action("[select]  Pick-up Time -> SELECT: 02:30 PM")
        == 'select "02:30 PM" in "Pick-up Time"'
    )


def _valid_rows() -> list[dict[str, Any]]:
    container_rows, _ = build_rows_for_action(source_row(), action(TWO_CARDS))
    negative_rows, _ = build_rows_for_action(
        source_row(id="task2:0", target_action="[button]  Apply filter -> CLICK"),
        action(UNIQUE),
    )
    safety_rows, _ = build_rows_for_action(
        source_row(id="task3:0", target_action="[button]  Delete account -> CLICK"),
        action(UNIQUE, target_action="[button]  Delete account -> CLICK"),
    )
    return [*container_rows, *negative_rows, *safety_rows]


def test_audit_accepts_valid_rows_and_checks_option_coverage() -> None:
    rows = _valid_rows()
    audit = audit_rows(rows)
    assert audit["totals"]["positives"] == 2
    assert audit["totals"]["negatives"] == 1
    assert audit["totals"]["full_option_coverage_rate"] == 1.0
    assert audit["totals"]["full_input_option_coverage_rate"] == 1.0
    assert option_coverage(rows[0]) == 1.0
    assert input_option_coverage(rows[0]) == 1.0
    assert audit["checks"] == {}
    markdown = render_markdown(audit, split="test_task")
    assert "# TreeVQG-Clarify audit" in markdown
    assert "Quality checks" in markdown


def test_audit_flags_a_broken_answer_index() -> None:
    row = _valid_rows()[0]
    row["answer"] = "somewhere else"
    audit = audit_rows([row])
    assert audit["checks"]["answer_index_mismatch"]["count"] == 1
    assert any("answer_index_mismatch" in warning for warning in audit["warnings"])


def test_audit_flags_a_positive_that_says_do_not_ask() -> None:
    row = _valid_rows()[0]
    row["target"] = NO_QUESTION
    audit = audit_rows([row])
    assert audit["checks"]["positive_is_no_question"]["count"] == 1


def _gold_predictions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": row["id"],
            "question": row["target"],
            "should_ask": row["meta"]["should_ask"],
            "answer": row["answer"],
        }
        for row in rows
    ]


def test_metrics_perfect_gold_predictions() -> None:
    rows = _valid_rows()
    evaluation = evaluate_rows(rows, {str(row["id"]): row for row in _gold_predictions(rows)})
    totals = evaluation["totals"]
    assert totals["ask"]["f1"] == 1.0
    assert totals["ask"]["precision"] == 1.0
    assert totals["ask"]["recall"] == 1.0
    assert totals["full_option_recall_rate"] == 1.0
    assert totals["answer_accuracy"] == 1.0
    assert totals["realized_info_gain_bits"] == totals["max_realized_info_gain_bits"]
    assert totals["wasted_questions"] == 0
    assert totals["missed_questions"] == 0


def test_metrics_missing_predictions_mean_no_question() -> None:
    rows = _valid_rows()
    totals = evaluate_rows(rows, {})["totals"]
    assert totals["pred_positives"] == 0
    assert totals["ask"]["recall"] == 0.0
    assert totals["missed_questions"] == totals["gold_positives"]
    assert totals["full_option_recall_rate"] == 0.0
    assert totals["realized_info_gain_bits"] == 0.0


def test_ask_prf_and_option_recall_helpers() -> None:
    assert info_gain_bits(4) == 2.0
    assert info_gain_bits(1) == 0.0
    assert ask_prf([True, False, True], [True, True, False]) == {
        "tp": 1,
        "fp": 1,
        "fn": 1,
        "tn": 0,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
        "accuracy": 0.333,
    }
    assert option_recall('One in "A", one in "B".', ["A", "B"]) == 1.0
    assert option_recall('One in "A".', ["A", "B"]) == 0.5
    assert option_recall("Confirm?", []) is None
