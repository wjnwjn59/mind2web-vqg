"""Readable report for one eval run: summary tables + per-example cards.

``predictions.json`` is machine-oriented (nested ``meta``, long prompts). This turns it into
two files next to it:

* ``report.md`` — the metrics tables plus one row per example (reference vs generated question)
* ``report.html`` — a gallery: each screenshot with the reference question/answer and the
  generated question, so a run can be reviewed in a browser

Usage::

    uv run m2w-report --split test_task
    uv run m2w-report --split test_task --limit 50
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any

from mind2web_vqg.cli import ensure_utf8
from mind2web_vqg.metrics import compute_metrics, predictions_path

_STYLE = """<style>
  body { font: 14px/1.5 system-ui, sans-serif; margin: 24px; color: #1a1a1a; }
  h1 { font-size: 20px; }
  table { border-collapse: collapse; margin: 12px 0 24px; }
  th, td {
    border: 1px solid #ddd; padding: 4px 10px; text-align: left; vertical-align: top;
  }
  .grid {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
    gap: 16px;
  }
  .card {
    margin: 0; border: 1px solid #e2e2e2; border-radius: 8px;
    overflow: hidden; background: #fafafa;
  }
  .card img {
    display: block; width: 100%; height: 220px; object-fit: cover;
    object-position: top; background: #fff;
  }
  figcaption { padding: 10px 12px; }
  .badge {
    display: inline-block; font-size: 11px; background: #eef; color: #335;
    border-radius: 10px; padding: 1px 8px;
  }
  .task { color: #666; font-size: 12px; margin: 6px 0; }
  .pred {
    margin: 6px 0 0; background: #fff; border-left: 3px solid #4a7; padding: 4px 8px;
  }
</style>"""


def _flat(text: Any, limit: int = 200) -> str:
    """Collapse whitespace and truncate for a table cell."""
    flat = " ".join(str(text if text is not None else "").split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _md_cell(text: Any, limit: int = 200) -> str:
    return _flat(text, limit).replace("|", "\\|")


def _summary_line(metrics: dict[str, Any]) -> str:
    return (
        f"{metrics['n']} predictions — well-formed {metrics['well_formed_rate']:.2f}, "
        f"unique {metrics['unique_ratio']:.2f}, self-BLEU {metrics['self_bleu']:.2f}, "
        f"ROUGE-L {metrics['rouge_l']:.2f}, chrF {metrics['chrf']:.2f}"
    )


def _categories(metrics: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    return sorted(metrics.get("by_category", {}).items())


def render_markdown(
    records: list[dict[str, Any]],
    metrics: dict[str, Any],
    *,
    title: str = "Eval report",
    limit: int | None = None,
) -> str:
    """Markdown report: summary, per-category metrics, then one row per example."""
    lines = [f"# {title}", "", _summary_line(metrics), "", "## Metrics by category", ""]
    lines += [
        "| Category | n | well-formed | unique | ROUGE-L | chrF |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for category, block in _categories(metrics):
        lines.append(
            f"| `{category}` | {block['n']} | {block['well_formed_rate']:.2f} | "
            f"{block['unique_ratio']:.2f} | {block['rouge_l']:.2f} | {block['chrf']:.2f} |"
        )

    lines += ["", "## Samples", ""]
    lines += [
        "| # | Category | Gold answer | Reference question | Generated question |",
        "| ---: | --- | --- | --- | --- |",
    ]
    for index, record in enumerate(records[:limit] if limit else records, start=1):
        meta = record.get("meta") or {}
        lines.append(
            f"| {index} | `{meta.get('category', '?')}` | {_md_cell(record.get('answer'), 60)} "
            f"| {_md_cell(record.get('target'))} | {_md_cell(record.get('prediction'))} |"
        )
    return "\n".join(lines) + "\n"


def _metrics_table_html(metrics: dict[str, Any]) -> str:
    rows = [
        "<tr><th>Category</th><th>n</th><th>well-formed</th><th>unique</th>"
        "<th>ROUGE-L</th><th>chrF</th></tr>"
    ]
    for category, block in _categories(metrics):
        rows.append(
            f"<tr><td><code>{html.escape(category)}</code></td><td>{block['n']}</td>"
            f"<td>{block['well_formed_rate']:.2f}</td><td>{block['unique_ratio']:.2f}</td>"
            f"<td>{block['rouge_l']:.2f}</td><td>{block['chrf']:.2f}</td></tr>"
        )
    return "<table>" + "".join(rows) + "</table>"


def _card_html(record: dict[str, Any]) -> str:
    meta = record.get("meta") or {}
    escape = html.escape
    return (
        "<figure class=\"card\">"
        f'<img src="{escape(str(record.get("image") or ""))}" alt="screenshot" loading="lazy">'
        "<figcaption>"
        f'<span class="badge">{escape(str(meta.get("category", "?")))}</span>'
        f'<p class="task">{escape(_flat(meta.get("task"), 220))}</p>'
        f"<p><b>Q (reference):</b> {escape(_flat(record.get('target')))}</p>"
        f"<p><b>A (gold):</b> {escape(_flat(record.get('answer'), 80))}</p>"
        f'<p class="pred"><b>Q (generated):</b> {escape(_flat(record.get("prediction")))}</p>'
        "</figcaption></figure>"
    )


def render_html(
    records: list[dict[str, Any]],
    metrics: dict[str, Any],
    *,
    title: str = "Eval report",
    limit: int | None = None,
) -> str:
    """Standalone HTML gallery (screenshot + reference/generated question per card)."""
    cards = [_card_html(record) for record in (records[:limit] if limit else records)]
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            f"<title>{html.escape(title)}</title>",
            _STYLE,
            "</head>",
            "<body>",
            f"<h1>{html.escape(title)}</h1>",
            f"<p>{html.escape(_summary_line(metrics))}</p>",
            _metrics_table_html(metrics),
            "<h2>Examples</h2>",
            '<div class="grid">',
            *cards,
            "</div>",
            "</body>",
            "</html>",
        ]
    )


def write_report(
    records: list[dict[str, Any]],
    out_dir: str | Path,
    *,
    title: str = "Eval report",
    limit: int | None = None,
) -> tuple[Path, Path]:
    """Write ``report.md`` and ``report.html`` into ``out_dir``; return both paths."""
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    metrics = compute_metrics(records)

    markdown = directory / "report.md"
    markdown.write_text(
        render_markdown(records, metrics, title=title, limit=limit), encoding="utf-8"
    )
    page = directory / "report.html"
    page.write_text(render_html(records, metrics, title=title, limit=limit), encoding="utf-8")
    return markdown, page


def main(argv: list[str] | None = None) -> int:
    ensure_utf8()
    parser = argparse.ArgumentParser(description="Readable report for an eval run.")
    parser.add_argument("--split", default="test_task")
    parser.add_argument(
        "--predictions", type=Path, default=None, help="default: outputs/eval_<split>/..."
    )
    parser.add_argument("--out", type=Path, default=None, help="default: next to predictions")
    parser.add_argument("--limit", type=int, default=None, help="cap the examples in the report")
    parser.add_argument("--title", default=None, help="report title (default: the run dir name)")
    args = parser.parse_args(argv)

    path = args.predictions or predictions_path(args.split)
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    title = args.title or f"Eval report — {Path(path).parent.name}"
    out_dir = args.out or Path(path).parent
    markdown, page = write_report(records, out_dir, title=title, limit=args.limit)

    print(f"[report] {len(records)} predictions")
    print(f"  {markdown}")
    print(f"  {page}   # open in a browser")
    return 0


if __name__ == "__main__":
    sys.exit(main())
