"""Tests for the Multimodal-Mind2Web helpers (no parquet, no network, no images)."""

import json

from mind2web_vqg.data.multimodal import (
    as_json,
    clean_task,
    from_row,
    image_name,
    parse_bbox,
    target_element,
)
from mind2web_vqg.smoke_vlm import PROBES, build_prompt


def test_as_json_handles_strings_and_passthrough() -> None:
    assert as_json('[{"a": 1}]') == [{"a": 1}]
    assert as_json('{"op": "CLICK"}') == {"op": "CLICK"}
    assert as_json({"already": "parsed"}) == {"already": "parsed"}
    assert as_json("not json", default=[]) == []


def test_parse_bbox() -> None:
    assert parse_bbox("534,45,151,22") == [534.0, 45.0, 151.0, 22.0]
    assert parse_bbox("1,2,3") is None
    assert parse_bbox(None) is None
    assert parse_bbox("a,b,c,d") is None


def test_clean_task_strips_the_stray_quote() -> None:
    # the parquet stores confirmed_task as '"Rent a truck ... April' (leading quote)
    assert clean_task('"Rent a truck on April') == "Rent a truck on April"
    assert clean_task("  Search for a red shirt  ") == "Search for a red shirt"
    assert clean_task(None) == ""


def test_target_element_prefers_original_target() -> None:
    candidates = [
        json.dumps({"tag": "div", "attributes": "{}"}),
        json.dumps(
            {
                "tag": "a",
                "attributes": json.dumps({"bounding_box_rect": "1,2,3,4", "text": "Sign in"}),
                "is_original_target": True,
            }
        ),
    ]
    assert target_element(candidates)["tag"] == "a"
    assert target_element([]) == {}


def test_from_row_parses_the_real_schema() -> None:
    row = {
        "annotation_id": "abc-123",
        "action_uid": "uid-1",
        "confirmed_task": "Find a hotel in Rome",
        "action_reprs": ["CLICK [Hotels]", "CLICK [Search]"],
        "target_action_reprs": "CLICK [Search]",
        "operation": json.dumps({"op": "CLICK", "value": ""}),
        "pos_candidates": [
            json.dumps(
                {
                    "tag": "button",
                    "attributes": json.dumps(
                        {"bounding_box_rect": "10,20,30,40", "aria_label": "Search"}
                    ),
                    "is_original_target": True,
                }
            )
        ],
        "screenshot": None,
    }

    example = from_row(row, index=0, split="test_task")

    assert example["id"] == "abc-123:0"
    assert example["image_name"] == "test_task_abc-123_00.jpg"
    assert example["task"] == "Find a hotel in Rome"
    assert example["op"] == "CLICK"
    assert example["target_tag"] == "button"
    assert example["target_text"] == "Search"
    assert example["target_bbox"] == [10.0, 20.0, 30.0, 40.0]
    assert example["action_history"] == ["CLICK [Hotels]", "CLICK [Search]"]


def test_image_name_is_filesystem_safe() -> None:
    assert "/" not in image_name("test_task", "a/b c", 7)
    assert image_name("test_task", "a/b c", 7).endswith("_07.jpg")


def test_probe_task_substitution() -> None:
    prompt = build_prompt(PROBES["how_next"], {"task": "Book a flight"})
    assert "Book a flight" in prompt
    assert "{task}" not in prompt
    # static probes keep their braces-free text
    assert "{task}" not in build_prompt(PROBES["yes_no"], {"task": "x"})
