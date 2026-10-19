"""Tests for the readable eval report (pure string formatting + tmp_path I/O)."""

from pathlib import Path
from typing import Any

from mind2web_vqg.metrics import compute_metrics
from mind2web_vqg.report import render_html, render_markdown, write_report


def records() -> list[dict[str, Any]]:
    return [
        {
            "id": "a:0::target_label",
            "target": "What text is shown on the link?",
            "answer": "Reservations",
            "image": "/data/images/a.jpg",
            "meta": {"category": "target_label", "task": "Rent a truck"},
            "prediction": "Where can I rent a <moving> truck?",
        },
        {
            "id": "a:0::target_region",
            "target": "In which part of the page is it?",
            "answer": "top-right",
            "image": "/data/images/a.jpg",
            "meta": {"category": "target_region", "task": "Rent a truck"},
            "prediction": "Where is the search button?",
        },
    ]


def test_markdown_has_summary_metrics_and_samples() -> None:
    markdown = render_markdown(records(), compute_metrics(records()), title="Eval report — x")
    assert markdown.startswith("# Eval report — x")
    assert "## Metrics by category" in markdown
    assert "`target_label`" in markdown
    assert "Reservations" in markdown  # the gold answer
    assert "Where is the search button?" in markdown  # the generated question


def test_markdown_respects_the_limit() -> None:
    markdown = render_markdown(records(), compute_metrics(records()), limit=1)
    assert "Reservations" in markdown
    assert "Where is the search button?" not in markdown


def test_markdown_escapes_table_pipes() -> None:
    rows = records()
    rows[0]["prediction"] = "a | b"
    markdown = render_markdown(rows, compute_metrics(rows))
    assert r"a \| b" in markdown


def test_html_has_images_and_escapes_markup() -> None:
    page = render_html(records(), compute_metrics(records()))
    assert 'src="/data/images/a.jpg"' in page
    assert "&lt;moving&gt;" in page  # the prediction was escaped
    assert "<table>" in page
    assert page.startswith("<!doctype html>")


def test_write_report_creates_both_files(tmp_path: Path) -> None:
    markdown, page = write_report(records(), tmp_path / "eval_test_task")
    assert markdown.name == "report.md" and markdown.exists()
    assert page.name == "report.html" and page.exists()
    assert "Reservations" in markdown.read_text(encoding="utf-8")


def test_reports_handle_an_empty_run(tmp_path: Path) -> None:
    empty: list[dict[str, Any]] = []
    metrics = compute_metrics(empty)
    assert "0 predictions" in render_markdown(empty, metrics)
    assert "<table>" in render_html(empty, metrics)
    markdown, page = write_report(empty, tmp_path)
    assert markdown.exists() and page.exists()
