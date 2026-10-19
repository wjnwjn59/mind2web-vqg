"""Render Mind2Web ``raw_html`` / ``cleaned_html`` to PNG, for a visual sanity check.

Mind2Web's JSON has **no images**: it stores the DOM as text. This script renders
that text with Chromium (Playwright) so you can see *why* it is not a screenshot:

- stylesheets, fonts and images are stripped, so the page is unstyled;
- ``<text>`` / ``backend_node_id`` attributes are not real HTML;
- there is no base URL, so no relative asset can load.

The goal is comparison, not a usable dataset. Real screenshots come from
``Multimodal-Mind2Web`` (see ``multimodal.py``); ``--compare-mm`` puts the real
capture next to the render so the difference is obvious.

Usage::

    # one-off: install the browser (per machine, ~150 MB)
    uv run playwright install chromium

    # render 3 tasks, raw + cleaned, with the real screenshot next to them
    uv run m2w-render-html --split test_task --limit 3 --compare-mm

    # just look at the output
    start outputs/render_check/test_task/index.html
"""

from __future__ import annotations

import argparse
import html as html_lib
import json
import sys
from pathlib import Path
from typing import Any

from mind2web_vqg.cli import ensure_utf8
from mind2web_vqg.data.load import load_split
from mind2web_vqg.paths import OUTPUT_DIR

VARIANTS = ("raw", "cleaned")


def wrap_html(fragment: str) -> str:
    """Make a DOM dump renderable: keep it as-is, but add a charset + reset.

    Deliberately does **not** inject CSS beyond a minimal reset: the point is to
    show the page exactly as stripped as the dataset is.
    """
    body = fragment.strip()
    if "<html" not in body.lower():
        body = f"<html><body>{body}</body></html>"
    return (
        "<!doctype html><meta charset='utf-8'>"
        "<style>body{margin:0;font-family:serif;background:#fff}"
        "img{max-width:100%}</style>" + body
    )


def render_html(
    page: Any,
    fragment: str,
    out_path: Path,
    *,
    width: int = 1280,
    wait_ms: int = 250,
) -> tuple[int, int]:
    """Render one HTML fragment into a full-page JPEG and return ``(w, h)``."""
    page.set_viewport_size({"width": width, "height": 900})
    page.set_content(wrap_html(fragment), wait_until="domcontentloaded", timeout=15_000)
    page.wait_for_timeout(wait_ms)
    page.screenshot(path=str(out_path), full_page=True, type="jpeg", quality=80)
    box = page.evaluate("() => [document.documentElement.scrollWidth, "
                        "document.documentElement.scrollHeight]")
    return int(box[0]), int(box[1])


def render_split(
    split: str,
    *,
    limit: int = 3,
    out_dir: Path | None = None,
    width: int = 1280,
    compare_mm: bool = False,
    root: str | Path | None = None,
    mm_root: str | Path | None = None,
) -> Path:
    """Render the first ``limit`` tasks of ``split`` and write an ``index.html``."""
    from playwright.sync_api import sync_playwright

    out_dir = out_dir or OUTPUT_DIR / "render_check" / split
    out_dir.mkdir(parents=True, exist_ok=True)

    instances = (load_split(split) if root is None else load_split(split, root))[:limit]
    rows: list[dict[str, Any]] = []

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        try:
            for inst in instances:
                annotation_id = str(inst.get("annotation_id") or "unknown")
                task = inst.get("confirmed_task") or ""
                actions = inst.get("actions") or []
                first = actions[0] if actions else {}
                entry: dict[str, Any] = {
                    "annotation_id": annotation_id,
                    "task": task,
                    "website": inst.get("website"),
                    "files": {},
                }

                for variant in VARIANTS:
                    fragment = first.get(f"{variant}_html")
                    if not fragment:
                        continue
                    name = f"{annotation_id}_{variant}.jpg"
                    size = render_html(page, fragment, out_dir / name, width=width)
                    entry["files"][variant] = {"name": name, "size": size}

                if compare_mm:
                    from mind2web_vqg.data.multimodal import load_screenshot

                    image = load_screenshot(
                        split, annotation_id, first.get("action_uid", ""), root=mm_root
                    )
                    if image is not None:
                        name = f"{annotation_id}_mm.jpg"
                        from mind2web_vqg.data.multimodal import fit_width

                        fit_width(image.convert("RGB"), width).save(
                            out_dir / name, format="JPEG", quality=82
                        )
                        entry["files"]["screenshot"] = {
                            "name": name,
                            "size": [image.width, image.height],
                        }
                rows.append(entry)
                print(f"[render] {annotation_id}: {', '.join(entry['files'])}")
        finally:
            browser.close()

    (out_dir / "index.html").write_text(_index_html(rows), encoding="utf-8")
    (out_dir / "summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[render] {len(rows)} task(s) -> {out_dir / 'index.html'}")
    return out_dir


def _index_html(rows: list[dict[str, Any]]) -> str:
    """Tiny static gallery so you can compare renders side by side."""
    blocks = []
    for row in rows:
        cells = []
        for label in ("screenshot", "raw", "cleaned"):
            info = row["files"].get(label)
            if not info:
                continue
            cells.append(
                f"<figure><figcaption>{html_lib.escape(label)} "
                f"({info['size'][0]}x{info['size'][1]})</figcaption>"
                f"<a href='{info['name']}'><img src='{info['name']}'></a></figure>"
            )
        blocks.append(
            "<section>"
            f"<h2>{html_lib.escape(row['annotation_id'])} - "
            f"{html_lib.escape(row.get('website') or '')}</h2>"
            f"<p>{html_lib.escape(row['task'])}</p>"
            f"<div class='grid'>{''.join(cells)}</div>"
            "</section>"
        )
    return (
        "<!doctype html><meta charset='utf-8'><title>render check</title>"
        "<style>body{font-family:sans-serif;margin:24px}"
        ".grid{display:flex;gap:16px;align-items:flex-start;flex-wrap:wrap}"
        "figure{margin:0;max-width:380px}img{width:100%;border:1px solid #ccc}"
        "figcaption{font-size:12px;color:#555}"
        "section{margin-bottom:48px;border-top:1px solid #eee;padding-top:16px}</style>"
        "<h1>Mind2Web render check</h1>"
        "<p>Left: the real screenshot (Multimodal-Mind2Web). Middle/right: the same page "
        "rendered from the stripped HTML in the JSON.</p>" + "".join(blocks)
    )


def main(argv: list[str] | None = None) -> int:
    ensure_utf8()
    parser = argparse.ArgumentParser(
        description="Render Mind2Web raw_html/cleaned_html with Playwright (visual check)."
    )
    parser.add_argument("--split", default="test_task")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument(
        "--root", type=Path, default=None, help="JSON data root (default data/raw/data)"
    )
    parser.add_argument(
        "--mm-root", type=Path, default=None, help="parquet root (default $MIND2WEB_MM_DIR)"
    )
    parser.add_argument(
        "--compare-mm",
        action="store_true",
        help="also export the real Multimodal-Mind2Web screenshot for the same task",
    )
    args = parser.parse_args(argv)

    try:
        render_split(
            args.split,
            limit=args.limit,
            out_dir=args.out,
            width=args.width,
            compare_mm=args.compare_mm,
            root=args.root,
            mm_root=args.mm_root,
        )
    except ImportError as exc:
        print(
            "playwright is not installed. Run:\n"
            "  uv sync --extra render\n"
            "  uv run playwright install chromium",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
    return 0


if __name__ == "__main__":
    sys.exit(main())
