"""Tests for the self-supervised VQG task (pure helpers, no images/wheels)."""

from pathlib import Path
from typing import Any

import pytest

from mind2web_vqg.data.vqg import (
    ALL_CATEGORIES,
    SCREENSHOT_GROUNDED,
    build_examples,
    category_stats,
    element_role,
    generate_questions,
    relative_region,
    write_examples,
)


def example(**over: Any) -> dict[str, Any]:
    """A ``from_row``-shaped example with sane defaults."""
    base: dict[str, Any] = {
        "id": "task1:0",
        "annotation_id": "task1",
        "action_uid": "uid1",
        "split": "test_task",
        "row_index": 0,
        "image_name": "test_task_task1_00.jpg",
        "task": "Search for a red shirt",
        "website": "shop",
        "domain": "shopping",
        "subdomain": "clothing",
        "action_history": [],
        "target_action": "CLICK [Search]",
        "op": "CLICK",
        "value": "",
        "target_tag": "button",
        "target_type": None,
        "target_role": None,
        "target_text": "Search",
        "target_bbox": [1200.0, 40.0, 80.0, 30.0],
        "source_size": [1280, 1000],
        "image_size": [1280, 1000],
        "image": "images/test_task_task1_00.jpg",
    }
    base.update(over)
    return base


# --------------------------------------------------------------------------- #
# element_role
# --------------------------------------------------------------------------- #
def test_role_from_tag() -> None:
    assert element_role(example(target_tag="button")) == "button"
    assert element_role(example(target_tag="a")) == "link"
    assert element_role(example(target_tag="select")) == "dropdown"


def test_role_refines_input_by_type() -> None:
    assert element_role(example(target_tag="input", target_type="search")) == "search field"
    assert element_role(example(target_tag="input", target_type="checkbox")) == "checkbox"
    assert element_role(example(target_tag="input", target_type="submit")) == "button"
    # no type attribute -> generic text field
    assert element_role(example(target_tag="input", target_type=None)) == "text field"


def test_role_none_for_unknown_or_missing_tag() -> None:
    assert element_role(example(target_tag="div")) is None
    assert element_role(example(target_tag=None)) is None


# --------------------------------------------------------------------------- #
# relative_region
# --------------------------------------------------------------------------- #
def test_region_thirds() -> None:
    size = [1000, 1000]
    assert relative_region([10, 10, 100, 20], source_size=size) == "top-left"
    assert relative_region([400, 400, 200, 200], source_size=size) == "middle-center"
    assert relative_region([900, 900, 100, 100], source_size=size) == "bottom-right"


def test_region_rescales_from_source_to_image() -> None:
    # half-width image: a box at the far right of the source stays far right.
    region = relative_region(
        [1800, 10, 100, 40], source_size=[2000, 1000], image_size=[1000, 500]
    )
    assert region == "top-right"


def test_region_none_when_coordinates_are_off_page() -> None:
    assert relative_region([5000, 10, 10, 10], source_size=[1000, 1000]) is None
    assert relative_region(None, source_size=[1000, 1000]) is None
    assert relative_region([1, 2, 3], source_size=[1000, 1000]) is None
    assert relative_region([1, 2, 3, 4], source_size=None, image_size=None) is None


# --------------------------------------------------------------------------- #
# generate_questions
# --------------------------------------------------------------------------- #
def test_default_categories_are_screenshot_grounded() -> None:
    qas = generate_questions(example())
    assert [qa["category"] for qa in qas] == list(SCREENSHOT_GROUNDED)
    assert all(qa["answer_source"] == "screenshot" for qa in qas)


def test_questions_and_answers() -> None:
    by_cat = {qa["category"]: qa for qa in generate_questions(example())}
    assert by_cat["target_label"]["answer"] == "Search"
    assert "Search" in by_cat["target_label"]["question"]
    assert by_cat["element_type"]["answer"] == "button"
    assert by_cat["target_region"]["answer"] == "top-right"


def test_field_value_is_opt_in_and_answered_from_task() -> None:
    row = example(op="TYPE", value="red shirt", target_tag="input", target_type="search")
    assert "field_value" not in {qa["category"] for qa in generate_questions(row)}
    by_cat = {qa["category"]: qa for qa in generate_questions(row, ["field_value"])}
    assert by_cat["field_value"]["answer"] == "red shirt"
    assert by_cat["field_value"]["answer_source"] == "task"


def test_categories_skipped_when_answer_missing() -> None:
    # no label -> nothing asks about the element's text/type/region
    assert generate_questions(example(target_text="")) == []
    # off-page box -> region dropped, the other two survive
    cats = {qa["category"] for qa in generate_questions(example(target_bbox=[9999, 0, 1, 1]))}
    assert cats == {"target_label", "element_type"}
    # unmapped tag -> element_type dropped (its answer would be meaningless)
    cats = {qa["category"] for qa in generate_questions(example(target_tag="div"))}
    assert cats == {"target_label", "target_region"}


def test_unknown_category_rejected() -> None:
    with pytest.raises(ValueError):
        generate_questions(example(), ["nope"])


# --------------------------------------------------------------------------- #
# build_examples
# --------------------------------------------------------------------------- #
def test_build_examples_ids_and_prompt() -> None:
    examples = build_examples([example()])
    assert len(examples) == len(SCREENSHOT_GROUNDED)
    assert {ex["id"] for ex in examples} == {
        "task1:0::target_label",
        "task1:0::element_type",
        "task1:0::target_region",
    }
    for ex in examples:
        assert ex["input"].startswith("You are shown a screenshot")
        assert "Search for a red shirt" in ex["input"]
        assert ex["target"] and ex["answer"]
        assert ex["image"] == "images/test_task_task1_00.jpg"
        assert ex["meta"]["source_id"] == "task1:0"


def test_build_examples_can_omit_task() -> None:
    examples = build_examples([example()], with_task=False)
    assert all("Search for a red shirt" not in ex["input"] for ex in examples)


def test_category_stats() -> None:
    examples = build_examples([example(), example(target_text="")], categories=ALL_CATEGORIES)
    stats = category_stats(examples)
    assert stats["target_label"] == 1
    assert stats["field_value"] == 0


def test_write_examples_round_trip(tmp_path: Path) -> None:
    import json

    examples = build_examples([example()])
    jsonl = write_examples(examples, tmp_path / "vqg")
    assert jsonl.exists()
    rows = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == len(examples)
    assert rows[0]["meta"]["category"] == "target_label"
