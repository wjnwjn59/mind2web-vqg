"""Tests for the Mind2Web HTML-tree EDA (stdlib parser, no data files)."""

import json
from pathlib import Path
from typing import Any

from mind2web_vqg.data.load import available_splits
from mind2web_vqg.eda import (
    accumulate,
    finalize,
    main,
    parse_html,
    recorded_target,
    render_markdown,
    render_summary,
    summarize,
    write_report,
)

SAMPLE_HTML = """
<html><body>
  <nav><a href="/">Home</a><a href="/x"><img src="x.png"></a></nav>
  <form>
    <input type="text" placeholder="Search">
    <input type="hidden" name="q">
    <select><option value="a">A</option></select>
    <button aria-label="Go">Go</button>
    <textarea></textarea>
  </form>
  <script>var x = '<a>';</script>
  <img src="hero.png" alt="Hero">
  <div><span>Some text</span></div>
</body></html>
"""


# --------------------------------------------------------------------------- #
# HTML parsing
# --------------------------------------------------------------------------- #
def test_parse_html_counts_tags_and_skips_scripts() -> None:
    parsed = parse_html(SAMPLE_HTML)
    assert parsed["elements"] == 16
    assert parsed["tags"]["a"] == 2
    assert parsed["tags"]["input"] == 2
    assert parsed["tags"]["img"] == 2
    assert parsed["tags"]["select"] == 1
    assert "script" not in parsed["tags"]  # script subtrees are ignored


def test_parse_html_detects_labels() -> None:
    parsed = parse_html(SAMPLE_HTML)
    assert parsed["labelled"]["a"] == 1  # <a>Home</a> yes, the icon link no
    assert parsed["labelled"]["input"] == 1  # placeholder, not the hidden one
    assert parsed["labelled"]["img"] == 1  # alt only
    assert parsed["labelled"]["button"] == 1  # aria-label + text
    assert parsed["labelled"].get("textarea", 0) == 0  # empty
    assert parsed["labelled"]["option"] == 1  # value / text


def test_parse_html_collects_input_types() -> None:
    assert parse_html(SAMPLE_HTML)["input_types"] == {"text": 1, "hidden": 1}
    assert parse_html('<input name="q">')["input_types"] == {"text": 1}


def test_parse_html_tolerates_unclosed_tags() -> None:
    parsed = parse_html("<div><a href='#'>x")
    assert parsed["tags"]["div"] == 1
    assert parsed["tags"]["a"] == 1
    assert parsed["labelled"]["a"] == 1


def test_parse_html_handles_empty_input() -> None:
    assert parse_html("")["elements"] == 0


# --------------------------------------------------------------------------- #
# Recorded targets
# --------------------------------------------------------------------------- #
def test_recorded_target_prefers_the_original_target() -> None:
    action = {
        "pos_candidates": [
            json.dumps({"tag": "div", "attributes": "{}"}),
            json.dumps(
                {
                    "tag": "button",
                    "attributes": json.dumps({"text": "Go"}),
                    "is_original_target": True,
                }
            ),
        ]
    }
    assert recorded_target(action)["tag"] == "button"


def test_recorded_target_falls_back_to_the_first_candidate() -> None:
    action = {"pos_candidates": [json.dumps({"tag": "a", "attributes": "{}"})]}
    assert recorded_target(action)["tag"] == "a"
    assert recorded_target({}) == {}


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def instance(
    html: str | None = SAMPLE_HTML, *, op: str = "CLICK", value: str = ""
) -> dict[str, Any]:
    action: dict[str, Any] = {
        "operation": {"op": op, "value": value},
        "pos_candidates": [
            json.dumps(
                {
                    "tag": "button",
                    "attributes": json.dumps(
                        {"text": "Go", "bounding_box_rect": "1,2,3,4"}
                    ),
                    "is_original_target": True,
                }
            )
        ],
        "neg_candidates": [{}, {}],
    }
    if html is not None:
        action["cleaned_html"] = html
    return {
        "annotation_id": "a",
        "confirmed_task": "Do the thing now",
        "website": "w",
        "domain": "d",
        "subdomain": "s",
        "actions": [action],
    }


def test_summarize_counts_elements_and_targets() -> None:
    result = summarize([instance(SAMPLE_HTML)], split="test_task")
    assert result["tasks"] == 1
    assert result["steps"] == 1
    assert result["steps_with_html"] == 1
    assert result["elements"] == 16
    assert result["targets"]["n"] == 1
    assert result["targets"]["label_rate"] == 1.0
    assert result["targets"]["bbox_rate"] == 1.0


def test_summarize_computes_the_target_lift() -> None:
    result = summarize([instance(SAMPLE_HTML)])
    row = next(r for r in result["targets"]["tags"] if r["tag"] == "button")
    assert row["count"] == 1
    assert row["share"] == 1.0
    assert row["tree_share"] == 0.0625  # 1 of 16 elements
    assert row["lift"] == 16.0


def test_summarize_estimates_vqg_potential() -> None:
    result = summarize([instance(SAMPLE_HTML)])
    assert result["potential"]["element_type"] == 12  # questionable tags only
    assert result["potential"]["target_label"] == 8  # ... that have text or a label
    assert result["potential"]["target_region"] == 1  # the recorded target's bbox


def test_summarize_reports_operations_and_candidates() -> None:
    result = summarize([instance(SAMPLE_HTML, op="TYPE", value="truck")])
    assert result["ops"]["TYPE"]["count"] == 1
    assert result["ops"]["TYPE"]["with_value"] == 1
    assert result["candidates"]["mean_neg"] == 2.0


def test_summarize_detects_repeated_page_states() -> None:
    result = summarize([instance(SAMPLE_HTML), instance(SAMPLE_HTML)])
    assert result["html_repeated_steps"] == 1
    assert result["html_unique_steps"] == 1


def test_summarize_handles_missing_html() -> None:
    result = summarize([instance(None)])
    assert result["steps_with_html"] == 0
    assert result["tasks_without_html"] == 1
    assert result["elements"] == 0


def test_summarize_marks_the_top_questionable_type() -> None:
    result = summarize([instance(SAMPLE_HTML)])
    assert result["questionable"][0]["tag"] == "a"  # 2 links, the most common mapped tag
    assert result["questionable"][0]["role"] == "link"


def test_merge_of_accumulators_matches_a_single_pass() -> None:
    rows = [instance(SAMPLE_HTML), instance(None)]
    left = accumulate(rows[:1])
    right = accumulate(rows[1:])
    assert finalize(left.merge([right]), split="all") == summarize(rows, split="all")


def test_merge_keeps_repeated_page_states_exact() -> None:
    page = instance(SAMPLE_HTML)
    merged = accumulate([page]).merge([accumulate([page])])
    assert merged.steps_with_html == 2
    assert len(merged.seen_html) == 1  # same page in both splits -> union, not sum
    assert finalize(merged)["html_repeated_steps"] == 1


# --------------------------------------------------------------------------- #
# Rendering / output
# --------------------------------------------------------------------------- #
def test_render_markdown_has_the_main_sections() -> None:
    markdown = render_markdown(summarize([instance(SAMPLE_HTML)], split="test_task"))
    assert markdown.startswith("# EDA — test_task")
    assert "## Findings" in markdown
    assert "## Element types — whole tree" in markdown
    assert "## Tree vs recorded targets" in markdown
    assert "## VQG potential" in markdown
    assert "`button`" in markdown


def test_write_report_creates_three_files(tmp_path: Path) -> None:
    result = summarize([instance(SAMPLE_HTML)], split="test_task")
    data_path, markdown_path, csv_path = write_report(result, tmp_path / "eda")
    assert data_path.name == "eda.json" and data_path.exists()
    assert markdown_path.name == "eda.md" and markdown_path.exists()
    assert csv_path.name == "tags.csv" and csv_path.exists()
    assert "button" in csv_path.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# Whole corpus
# --------------------------------------------------------------------------- #
def test_available_splits_lists_only_existing_directories(tmp_path: Path) -> None:
    (tmp_path / "test_task").mkdir()
    (tmp_path / "train").mkdir()
    assert available_splits(tmp_path) == ["train", "test_task"]  # SPLITS order


def test_render_summary_has_one_row_per_split_plus_all() -> None:
    task = summarize([instance(SAMPLE_HTML)], split="test_task")
    train = summarize([instance(None)], split="train")
    markdown = render_summary([task, train])
    assert markdown.startswith("# EDA — all splits")
    assert "`test_task`" in markdown
    assert "`train`" in markdown
    assert "**all**" in markdown


def test_main_all_splits_writes_a_summary(tmp_path: Path) -> None:
    for split in ("train", "test_task"):
        directory = tmp_path / "data" / split
        directory.mkdir(parents=True)
        rows = json.dumps([instance(SAMPLE_HTML)])
        (directory / f"{split}_0.json").write_text(rows, encoding="utf-8")

    out = tmp_path / "out"
    code = main(
        [
            "--all-splits",
            "--root",
            str(tmp_path / "data"),
            "--limit",
            "1",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    assert (out / "summary.md").exists()
    assert (out / "summary.json").exists()
    assert (out / "test_task" / "eda.md").exists()
    assert (out / "train" / "eda.md").exists()

    # Plus one merged report for the whole corpus, built without a second pass.
    merged = json.loads((out / "all" / "eda.json").read_text(encoding="utf-8"))
    assert merged["split"] == "all"
    assert merged["tasks"] == 2
    assert merged["html_repeated_steps"] == 1  # the same page in both splits
    assert (out / "all" / "tags.csv").exists()
