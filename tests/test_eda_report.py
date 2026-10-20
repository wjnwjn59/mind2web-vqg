"""Tests for the LaTeX EDA report (no data files, no TeX distribution needed)."""

import json
from pathlib import Path

from mind2web_vqg.eda import summarize, write_report
from mind2web_vqg.eda_report import (
    concentration,
    find_reports,
    main,
    num,
    pct,
    policy_rows,
    render_tex,
    tex,
    whole,
)
from mind2web_vqg.eda_report import write_report as write_tex

SAMPLE_HTML = """
<html><body>
  <table><tr><td>Route</td><td>Price</td></tr><tr><td>A</td><td>10</td></tr></table>
  <a href="/x">Book now</a>
  <input type="text" placeholder="Search">
  <select><option>One</option></select>
  <button aria-label="Go">Go</button>
</body></html>
"""


def instance(html: str | None = SAMPLE_HTML, *, op: str = "CLICK") -> dict:
    action = {
        "operation": {"op": op, "value": ""},
        "pos_candidates": [
            json.dumps(
                {
                    "tag": "button",
                    "attributes": json.dumps({"text": "Go", "bounding_box_rect": "1,2,3,4"}),
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
        "confirmed_task": "Do the thing",
        "website": "w",
        "domain": "d",
        "subdomain": "s",
        "actions": [action],
    }


def sample_result(name: str = "test_task") -> dict:
    return summarize([instance(), instance(op="TYPE")], split=name)


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #
def test_tex_escapes_every_special_character() -> None:
    assert tex("a_b") == r"a\_b"
    assert tex("100%") == r"100\%"
    assert tex("x&y") == r"x\&y"
    assert tex("2^3") == r"2\textasciicircum{}3"
    assert tex(r"a\b") == r"a\textbackslash{}b"
    assert tex("plain") == "plain"


def test_tex_does_not_double_escape_backslashes() -> None:
    assert tex(r"\{") == r"\textbackslash{}\{"


def test_number_formatting() -> None:
    assert num(14193) == "14,193"
    assert num(84.0) == "84"
    assert num(0.0625) == "0.06"
    assert whole(2858.4) == "2,858"
    assert pct(0.0625) == r"6.2\,\%"
    assert pct(1.0, digits=0) == r"100\,\%"


# --------------------------------------------------------------------------- #
# Derived analysis
# --------------------------------------------------------------------------- #
def test_concentration_of_an_even_pool() -> None:
    result = concentration([0.25, 0.25, 0.25, 0.25])
    assert result["n"] == 4
    assert round(result["hhi"], 3) == 0.25
    assert round(result["effective"], 1) == 4.0
    assert result["tags_for_50"] == 2
    assert result["tags_for_80"] == 4


def test_concentration_of_a_dominated_pool() -> None:
    result = concentration([0.9, 0.05, 0.03, 0.02])
    assert round(result["hhi"], 3) == 0.814
    assert round(result["effective"], 1) == 1.2
    assert result["tags_for_50"] == 1
    assert result["tags_for_80"] == 1


def test_concentration_ignores_empty_types() -> None:
    result = concentration([0.5, 0.5, 0.0])
    assert result["n"] == 2


def test_policy_caps_the_dominant_type_below_uncontrolled() -> None:
    policy = policy_rows(sample_result(), 1000)
    total = sum(row["share"] for row in policy["rows"])
    assert abs(total - 1.0) < 1e-3  # shares are rounded to 4 decimals by the EDA

    leader = max(policy["rows"], key=lambda row: row["share"])
    assert leader["uncontrolled"] == leader["share"] * 1000
    assert leader["cap"] <= policy["even"]
    assert leader["cap"] <= leader["available"]


def test_policy_reports_the_shortfall_when_the_corpus_is_too_small() -> None:
    policy = policy_rows(sample_result(), 10_000)
    # Two steps cannot supply 10,000/20 questions per type.
    assert policy["shortfall"] > 0
    assert policy["shortfall_share"] > 0.5


def test_policy_with_a_budget_the_corpus_can_supply() -> None:
    policy = policy_rows(sample_result(), 10)
    assert policy["shortfall"] == 0.0
    assert all(row["cap"] <= policy["even"] for row in policy["rows"])


# --------------------------------------------------------------------------- #
# Reading EDA output
# --------------------------------------------------------------------------- #
def test_find_reports_separates_the_merged_report(tmp_path: Path) -> None:
    write_report(sample_result("train"), tmp_path / "train")
    write_report(sample_result("test_task"), tmp_path / "test_task")
    write_report(sample_result("all"), tmp_path / "all")

    splits, merged = find_reports(tmp_path)
    assert list(splits) == ["train", "test_task"]  # SPLITS order, not alphabetical
    assert merged is not None and merged["split"] == "all"


def test_find_reports_without_merged(tmp_path: Path) -> None:
    write_report(sample_result("test_task"), tmp_path / "test_task")
    splits, merged = find_reports(tmp_path)
    assert merged is None
    assert list(splits) == ["test_task"]


def test_find_reports_explains_a_missing_directory(tmp_path: Path) -> None:
    try:
        find_reports(tmp_path / "nope")
    except FileNotFoundError as error:
        assert "m2w-eda" in str(error)
    else:  # pragma: no cover - the call above must raise
        raise AssertionError("expected FileNotFoundError")


def test_find_reports_explains_an_empty_directory(tmp_path: Path) -> None:
    try:
        find_reports(tmp_path)
    except FileNotFoundError as error:
        assert "m2w-eda" in str(error)
    else:  # pragma: no cover
        raise AssertionError("expected FileNotFoundError")


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def test_render_tex_contains_every_section() -> None:
    document = render_tex(sample_result(), budget=5000)
    for heading in (
        r"\section{Summary}",
        r"\section{Corpus shape}",
        r"\section{Element types}",
        r"\section{The question-capable pool}",
        r"\section{Tree versus recorded targets}",
        r"\section{Question potential per category}",
        r"\section{A sampling budget that removes the bias}",
        r"\section{Method}",
        r"\section{Reproducing this report}",
    ):
        assert heading in document
    assert document.startswith(r"\documentclass")
    assert document.rstrip().endswith(r"\end{document}")
    assert document.count(r"\begin{document}") == 1


def test_render_tex_has_valid_environments() -> None:
    document = render_tex(sample_result())
    assert document.count(r"\begin{") == document.count(r"\end{")


def test_render_tex_embeds_a_chart_per_section() -> None:
    document = render_tex(sample_result())
    assert document.count(r"\begin{axis}") >= 4
    assert r"\label{fig:tree}" in document
    assert r"\label{fig:lift}" in document
    assert r"\label{tab:policy}" in document


def test_render_tex_with_several_splits_adds_the_drift_section() -> None:
    splits = {"train": sample_result("train"), "test_task": sample_result("test_task")}
    merged = sample_result("all")
    document = render_tex(merged, splits=splits, merged=merged)
    assert r"\section{Drift between splits}" in document
    assert r"\label{fig:splits}" in document
    # Small multiples: one axis per split, never several series in one axis
    assert document.count(r"\begin{axis}") >= 6


def test_render_tex_with_one_split_skips_the_drift_section() -> None:
    single = {"test_task": sample_result("test_task")}
    document = render_tex(sample_result("test_task"), splits=single, merged=None)
    assert r"\section{Drift between splits}" not in document


def test_render_tex_escapes_tag_names() -> None:
    document = render_tex(sample_result())
    assert r"\texttt{input}" in document
    assert r"\texttt{h1}" not in document  # not present in the sample tree


def test_write_tex_creates_the_directory(tmp_path: Path) -> None:
    path = write_tex("hello", tmp_path / "deep" / "report")
    assert path.name == "report.tex"
    assert path.read_text(encoding="utf-8") == "hello"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def test_main_writes_the_report(tmp_path: Path) -> None:
    write_report(sample_result("train"), tmp_path / "eda" / "train")
    write_report(sample_result("test_task"), tmp_path / "eda" / "test_task")
    write_report(sample_result("all"), tmp_path / "eda" / "all")

    code = main(["--root", str(tmp_path / "eda"), "--budget", "500"])
    assert code == 0
    report = (tmp_path / "eda" / "report" / "report.tex").read_text(encoding="utf-8")
    assert r"\section{Drift between splits}" in report
    assert "500" in report


def test_main_reports_an_unknown_split(tmp_path: Path) -> None:
    write_report(sample_result("test_task"), tmp_path / "eda" / "test_task")
    assert main(["--root", str(tmp_path / "eda"), "--split", "train"]) == 1


def test_main_without_data(tmp_path: Path) -> None:
    assert main(["--root", str(tmp_path / "missing")]) == 1
