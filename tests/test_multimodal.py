"""Tests for the Multimodal-Mind2Web helpers (no parquet, no network, no images)."""

import io
import json

from mind2web_vqg.data.multimodal import (
    as_json,
    clean_task,
    from_row,
    image_name,
    normalize_limit,
    parse_bbox,
    step_index,
    target_element,
)
from mind2web_vqg.smoke_vlm import PROBES, build_prompt


def test_as_json_handles_strings_and_passthrough() -> None:
    assert as_json('[{"a": 1}]') == [{"a": 1}]
    assert as_json('{"op": "CLICK"}') == {"op": "CLICK"}
    assert as_json({"already": "parsed"}) == {"already": "parsed"}
    assert as_json("not json", default=[]) == []


def test_normalize_limit_treats_zero_as_all() -> None:
    # `m2w-mm-export --limit 0` must mean "the whole split", like `m2w-eda --limit 0`.
    assert normalize_limit(0) is None
    assert normalize_limit(-3) is None
    assert normalize_limit(None) is None
    assert normalize_limit(5) == 5


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
    # Steps *before* the current action: the current one is the target, and future
    # actions must never leak into the prompt.
    assert example["action_history"] == ["CLICK [Hotels]"]


def test_step_index_prefers_the_dataset_index() -> None:
    reprs = ["CLICK [A]", "CLICK [B]", "CLICK [B]"]
    assert step_index({"target_action_index": "2", "target_action_reprs": "CLICK [B]"}, reprs) == 2
    # No index: fall back to the first textual match.
    assert step_index({"target_action_reprs": "CLICK [B]"}, reprs) == 1
    # Nothing matches: assume the first step rather than leaking the whole trajectory.
    assert step_index({"target_action_reprs": "missing"}, reprs) == 0


def test_export_samples_skips_missing_screenshots(tmp_path, monkeypatch) -> None:
    from PIL import Image

    import mind2web_vqg.data.multimodal as mm

    buffer = io.BytesIO()
    Image.new("RGB", (4, 2), "white").save(buffer, format="PNG")

    row = {
        "annotation_id": "abc-123",
        "action_uid": "uid-1",
        "confirmed_task": "Find a hotel in Rome",
        "action_reprs": ["CLICK [Search]"],
        "target_action_reprs": "CLICK [Search]",
        "operation": json.dumps({"op": "CLICK", "value": ""}),
        "pos_candidates": [],
    }
    monkeypatch.setattr(
        mm, "iter_rows", lambda *args, **kwargs: iter([{**row, "screenshot": None}])
    )
    examples = mm.export_samples("test_task", limit=1, dest=tmp_path, max_width=4)
    assert examples == []

    monkeypatch.setattr(
        mm,
        "iter_rows",
        lambda *args, **kwargs: iter([{**row, "screenshot": buffer.getvalue()}]),
    )
    examples = mm.export_samples("test_task", limit=1, dest=tmp_path, max_width=4)
    assert len(examples) == 1
    assert examples[0]["image_size"] == [4, 2]


def test_image_name_is_filesystem_safe() -> None:
    assert "/" not in image_name("test_task", "a/b c", 7)
    assert image_name("test_task", "a/b c", 7).endswith("_07.jpg")


def test_probe_task_substitution() -> None:
    prompt = build_prompt(PROBES["how_next"], {"task": "Book a flight"})
    assert "Book a flight" in prompt
    assert "{task}" not in prompt
    # static probes keep their braces-free text
    assert "{task}" not in build_prompt(PROBES["yes_no"], {"task": "x"})
