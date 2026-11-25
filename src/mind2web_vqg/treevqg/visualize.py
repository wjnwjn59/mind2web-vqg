"""Render a TreeVQG-Bench sample as a self-contained HTML gallery.

The point of this tool is to make a generated question **checkable in five seconds**:
the screenshot is cropped around the referents, every node the question depends on is
drawn on the image, and the question/answer pair sits next to it. Without that, a
``container_membership`` row and a ``target_label`` row look equally plausible in JSONL.

Outputs (default ``outputs/bench_v1_demo``)::

    gallery.html      one card per sampled row
    summary.md        per-type counts, leakage, answer sources
    images/*.jpg      crops around the referents

Usage::

    uv run m2w-visualize-bench \
        --examples data/processed/bench/v1/test_task/examples.jsonl \
        --out-dir outputs/bench_v1_demo --per-type 3
"""

from __future__ import annotations

import argparse
import html
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from mind2web_vqg.cli import ensure_utf8
from mind2web_vqg.paths import OUTPUT_DIR
from mind2web_vqg.treevqg import io as tree_io

#: Draw order + colour per referent kind.
_KIND_STYLE: dict[str, tuple[str, str, str]] = {
    # kind: (colour, border style, human label)
    "anchor": ("#e11d48", "solid", "anchor"),
    "container": ("#16a34a", "dashed", "container"),
    "candidate": ("#f59e0b", "dashed", "same-label candidate"),
    "reference": ("#2563eb", "dotted", "reference"),
}

_MIN_WIDTH = 900
_MIN_HEIGHT = 620
_PAD = 70


def _slug(value: str, limit: int = 90) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    return cleaned[:limit] or "row"


def _boxes(row: dict[str, Any], width: int, height: int) -> list[dict[str, Any]]:
    """Referent bboxes in *image* pixel space (DOM boxes use ``source_size``)."""
    source = row.get("source_size") or [width, height]
    try:
        source_w, source_h = float(source[0]), float(source[1])
    except (TypeError, ValueError, IndexError):
        source_w, source_h = float(width), float(height)
    if source_w <= 0 or source_h <= 0:
        source_w, source_h = float(width), float(height)
    scale_x, scale_y = width / source_w, height / source_h

    out: list[dict[str, Any]] = []
    for referent in row.get("referents") or []:
        bbox = referent.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        try:
            x, y, w, h = (float(value) for value in bbox)
        except (TypeError, ValueError):
            continue
        if w <= 0 or h <= 0:
            continue
        out.append({**referent, "px": [x * scale_x, y * scale_y, w * scale_x, h * scale_y]})
    return out


def _crop_box(boxes: list[dict[str, Any]], width: int, height: int) -> tuple[int, int, int, int]:
    if not boxes:
        return 0, 0, width, min(height, _MIN_HEIGHT)
    left = min(box["px"][0] for box in boxes)
    top = min(box["px"][1] for box in boxes)
    right = max(box["px"][0] + box["px"][2] for box in boxes)
    bottom = max(box["px"][1] + box["px"][3] for box in boxes)

    left, top = left - _PAD, top - _PAD
    right, bottom = right + _PAD, bottom + _PAD

    # Grow to a readable minimum, keeping the referents centred.
    cx, cy = (left + right) / 2, (top + bottom) / 2
    box_w = max(right - left, _MIN_WIDTH)
    box_h = max(bottom - top, _MIN_HEIGHT)
    left, top = cx - box_w / 2, cy - box_h / 2

    left = max(0, int(left))
    top = max(0, int(top))
    right = min(width, int(left + box_w))
    bottom = min(height, int(top + box_h))
    # Re-expand after clamping so the kept region is never smaller than the minimum.
    left = max(0, right - int(box_w))
    top = max(0, bottom - int(box_h))
    return left, top, max(1, right - left), max(1, bottom - top)


def _crop_image(
    row: dict[str, Any], image_path: Path, out_path: Path
) -> list[dict[str, Any]] | None:
    from PIL import Image, ImageDraw, ImageFont

    try:
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            width, height = image.size
            boxes = _boxes(row, width, height)
            left, top, box_w, box_h = _crop_box(boxes, width, height)
            crop = image.crop((left, top, left + box_w, top + box_h))
            draw = ImageDraw.Draw(crop)
            font = ImageFont.load_default()
            # Containers first, the anchor last, so the most important box stays visible.
            order = {"container": 0, "candidate": 1, "reference": 2, "anchor": 3}
            for box in sorted(boxes, key=lambda item: order.get(str(item.get("kind")), -1)):
                colour, _, name = _KIND_STYLE.get(
                    str(box.get("kind")), ("#64748b", "solid", str(box.get("kind")))
                )
                x, y, w, h = box["px"]
                x0, y0 = x - left, y - top
                x1, y1 = x0 + w, y0 + h
                draw.rectangle([x0, y0, x1, y1], outline=colour, width=3)
                text = f'{name}: {str(box.get("label") or box.get("role") or "")[:38]}'
                text_w = draw.textlength(text, font=font)
                label_top = max(0, y0 - 13)
                draw.rectangle([x0, label_top, x0 + text_w + 6, label_top + 13], fill=colour)
                draw.text((x0 + 3, label_top + 1), text, fill="#ffffff", font=font)
            crop.save(out_path, format="JPEG", quality=85, optimize=True)
    except (OSError, ValueError):
        return None
    return boxes


def _chips(row: dict[str, Any]) -> str:
    meta = row.get("meta") or {}
    chips = [
        f'<span class="chip type">{html.escape(str(row.get("type")))}</span>',
        f'<span class="chip">{html.escape(str(meta.get("website") or "?"))}</span>',
        f'<span class="chip">{html.escape(str(meta.get("answer_source") or "?"))}</span>',
    ]
    if meta.get("requires_tree"):
        chips.append('<span class="chip tree">tree-dependent</span>')
    if meta.get("answer_in_input"):
        chips.append('<span class="chip leak">answer copyable from question+task</span>')
    if meta.get("answer_in_question"):
        chips.append('<span class="chip leak">answer inside question</span>')
    if meta.get("polarity"):
        chips.append(f'<span class="chip">polarity: {html.escape(str(meta["polarity"]))}</span>')
    if meta.get("consequence_class"):
        chips.append(
            f'<span class="chip">class: {html.escape(str(meta["consequence_class"]))}</span>'
        )
    if meta.get("n_candidates"):
        chips.append(f'<span class="chip">distractors: {int(meta["n_candidates"])}</span>')
    return "".join(chips)


def _card(row: dict[str, Any], image_rel: str, boxes: list[dict[str, Any]], index: int) -> str:
    meta = row.get("meta") or {}
    referents = "".join(
        f"<tr><td>{html.escape(str(ref.get('kind')))}</td>"
        f"<td>{html.escape(str(ref.get('role')))}</td>"
        f"<td>{html.escape(str(ref.get('label') or ''))}</td>"
        f"<td>{html.escape(str(ref.get('depth')))}</td></tr>"
        for ref in (row.get("referents") or [])
    )
    table_head = (
        "<thead><tr><th>kind</th><th>role</th><th>label</th><th>depth</th></tr></thead>"
    )
    return f"""
<section class="card" id="row-{index}">
  <header>
    <div class="chips">{_chips(row)}</div>
    <div class="id">{html.escape(str(row.get("id")))}</div>
  </header>
  <div class="shot"><img src="{image_rel}" alt="screenshot crop with referents drawn"></div>
  <div class="qa">
    <p class="task"><b>Task</b> {html.escape(str(meta.get("task") or ""))}</p>
    <p class="q"><b>Q</b> {html.escape(str(row.get("question")))}</p>
    <p class="a"><b>A</b> {html.escape(str(row.get("answer")))}</p>
  </div>
  <details>
    <summary>Generator condition &amp; referents</summary>
    <table class="refs">{table_head}<tbody>{referents}</tbody></table>
    <pre>{html.escape(str(row.get("condition") or ""))}</pre>
  </details>
</section>"""


_CSS = """
:root { color-scheme: light; }
* { box-sizing: border-box; }
body { margin: 0; font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
       background: #f8fafc; color: #0f172a; }
header.page { padding: 28px 32px 12px; background: #0f172a; color: #f8fafc; }
header.page h1 { margin: 0 0 6px; font-size: 22px; }
header.page p { margin: 0; color: #cbd5e1; max-width: 90ch; }
nav { padding: 12px 32px; background: #1e293b; display: flex; gap: 14px; flex-wrap: wrap; }
nav a { color: #e2e8f0; text-decoration: none; font-size: 13px; padding: 3px 9px;
        border: 1px solid #475569; border-radius: 999px; }
nav a:hover { background: #334155; }
main { padding: 24px 32px 60px; display: grid; gap: 22px;
       grid-template-columns: repeat(auto-fill, minmax(560px, 1fr)); }
.card { background: #fff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 16px;
        box-shadow: 0 1px 3px rgb(15 23 42 / .06); }
.card header { display: flex; justify-content: space-between; gap: 10px; align-items: start; }
.chips { display: flex; gap: 6px; flex-wrap: wrap; }
.chip { font-size: 11px; padding: 2px 8px; border-radius: 999px; background: #f1f5f9;
        color: #334155; border: 1px solid #e2e8f0; white-space: nowrap; }
.chip.type { background: #0f172a; color: #f8fafc; border-color: #0f172a; }
.chip.tree { background: #dcfce7; border-color: #86efac; color: #14532d; }
.chip.leak { background: #fee2e2; border-color: #fca5a5; color: #7f1d1d; }
.id { font: 11px/1.3 ui-monospace, SFMono-Regular, Consolas, monospace; color: #94a3b8;
      text-align: right; word-break: break-all; max-width: 40%; }
.shot { margin: 12px 0; border: 1px solid #e2e8f0; border-radius: 8px; overflow: auto;
        max-height: 460px; background: #f1f5f9; }
.shot-inner { position: relative; min-width: 100%; }
.shot img { display: block; width: 100%; }
.box { position: absolute; border-width: 2px; border-radius: 3px; pointer-events: none; }
.box span { position: absolute; left: -2px; top: -17px; font-size: 10px; line-height: 1.6;
            color: #fff; padding: 0 5px; border-radius: 3px 3px 3px 0;
            white-space: nowrap; max-width: 260px; overflow: hidden; text-overflow: ellipsis; }
.qa p { margin: 6px 0; }
.qa .task { color: #64748b; font-size: 13px; }
.qa .q { font-size: 15px; }
.qa .a { color: #065f46; background: #ecfdf5; border-left: 3px solid #10b981;
         padding: 6px 10px; border-radius: 0 6px 6px 0; }
details { margin-top: 10px; font-size: 13px; }
summary { cursor: pointer; color: #2563eb; }
pre { white-space: pre-wrap; word-break: break-word; background: #f8fafc; padding: 10px;
      border: 1px solid #e2e8f0; border-radius: 6px; max-height: 320px; overflow: auto;
      font-size: 12px; }
table.refs { border-collapse: collapse; margin: 10px 0; width: 100%; }
table.refs th, table.refs td { text-align: left; padding: 4px 8px; border-bottom: 1px solid #e2e8f0;
      font-size: 12px; vertical-align: top; }
.legend { display: flex; gap: 14px; flex-wrap: wrap; font-size: 12px; color: #475569;
          padding: 10px 32px 0; }
.legend i { display: inline-block; width: 16px; height: 10px; margin-right: 5px;
            border-width: 2px; border-style: solid; }
"""


def _pick(group: list[dict[str, Any]], per_type: int) -> list[dict[str, Any]]:
    """Take ``per_type`` rows, but make sure a leaky row is shown when one exists.

    The leak is the point of the `target_label` control: a gallery without it looks as
    if every question were equally hard.
    """
    chosen: list[dict[str, Any]] = []
    if per_type >= 3:
        leaky = next(
            (row for row in group if (row.get("meta") or {}).get("answer_in_input")),
            None,
        )
        if leaky is not None:
            chosen.append(leaky)
    seen = {id(row) for row in chosen}
    for row in group:
        if len(chosen) >= per_type:
            break
        if id(row) in seen:
            continue
        chosen.append(row)
    return chosen


def render_gallery(
    rows: list[dict[str, Any]],
    out_dir: Path,
    *,
    title: str,
    examples_path: Path,
) -> Path:
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    for stale in images_dir.glob("*.jpg"):
        stale.unlink(missing_ok=True)

    cards: list[str] = []
    nav: list[str] = []
    types: Counter[str] = Counter()
    missing = 0
    for row in rows:
        source_image = Path(str(row.get("image") or ""))
        if not source_image.is_file():
            missing += 1
            continue
        index = len(cards) + 1
        name = f"{index:03d}_{_slug(str(row.get('id')))}.jpg"
        rendered = _crop_image(row, source_image, images_dir / name)
        if rendered is None:
            missing += 1
            continue
        boxes = rendered
        cards.append(_card(row, f"images/{name}", boxes, index))
        nav.append(
            f'<a href="#row-{index}">{html.escape(str(row.get("type")))} · {index}</a>'
        )
        types[str(row.get("type"))] += 1

    legend = "".join(
        f'<span><i style="border-color:{colour};border-style:{style}"></i>{name}</span>'
        for colour, style, name in _KIND_STYLE.values()
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title><style>{_CSS}</style></head>
<body>
<header class="page">
  <h1>{html.escape(title)}</h1>
  <p>{len(cards)} question(s) sampled from <code>{html.escape(str(examples_path))}</code>.
  Screenshots are cropped around the referents; the question and its gold answer sit below.
  {("Missing images: " + str(missing) + ". ") if missing else ""}
  Open <code>summary.md</code> for per-type leakage numbers.</p>
</header>
<nav>{''.join(nav)}</nav>
<div class="legend">{legend}</div>
<main>{''.join(cards)}</main>
</body></html>"""

    path = out_dir / "gallery.html"
    path.write_text(document, encoding="utf-8")
    return path


def render_summary(
    rows: list[dict[str, Any]], out_dir: Path, *, examples_path: Path
) -> Path:
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_type[str(row.get("type"))].append(row)

    def _rate(group: list[dict[str, Any]], key: str) -> str:
        if not group:
            return "–"
        hits = sum(1 for row in group if (row.get("meta") or {}).get(key))
        return f"{hits / len(group):.2f}"

    lines = [
        "# TreeVQG-Bench v1 — demo summary",
        "",
        f"- Source: `{examples_path}`",
        f"- Rows shown: **{len(rows)}**",
        "",
        "| type | n | tree | answer source | leak: in input | leak: in question |",
        "| --- | ---: | :---: | --- | ---: | ---: |",
    ]
    for kind, group in by_type.items():
        meta = group[0].get("meta") or {}
        sources = Counter(str((row.get("meta") or {}).get("answer_source")) for row in group)
        lines.append(
            f"| `{kind}` | {len(group)} | {'yes' if meta.get('requires_tree') else 'no'} | "
            f"{', '.join(f'{name} ({count})' for name, count in sources.most_common())} | "
            f"{_rate(group, 'answer_in_input')} | {_rate(group, 'answer_in_question')} |"
        )

    lines += [
        "",
        "## Cách đọc",
        "",
        "- **tree = yes** → đáp án là một *referent cấu trúc* (section / container / hậu quả),",
        "  không phải chuỗi chữ nằm sẵn trên màn hình.",
        "- **leak: in input** → đáp án xuất hiện trong `Task + Question` mà người trả lời "
        "nhìn thấy.",
        "  Cao ở `target_label` là *dự kiến* (câu hỏi kiểu caption); thấp ở các type tree là",
        "  điều kiện để type đó có nghĩa.",
        "- **leak: in question** → câu hỏi tự chứa đáp án; phải bằng 0 với mọi type.",
        "- Ảnh đã crop quanh các referent; khung đỏ = anchor, xanh lá = container,",
        "  cam = candidate cùng label, xanh dương = reference.",
    ]
    path = out_dir / "summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def visualize(
    examples_path: str | Path,
    *,
    out_dir: str | Path | None = None,
    types: Iterable[str] | None = None,
    per_type: int = 3,
    max_total: int | None = None,
    title: str = "TreeVQG-Bench v1 — sample gallery",
) -> dict[str, Any]:
    source = Path(examples_path)
    rows = tree_io.read_jsonl(source)
    if types:
        wanted = set(types)
        rows = [row for row in rows if row.get("type") in wanted]

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("type")), []).append(row)
    sampled = [row for group in grouped.values() for row in _pick(group, per_type)]
    if max_total:
        sampled = sampled[:max_total]

    destination = Path(out_dir) if out_dir else OUTPUT_DIR / "bench_v1_demo"
    destination.mkdir(parents=True, exist_ok=True)
    gallery = render_gallery(sampled, destination, title=title, examples_path=source)
    summary = render_summary(sampled, destination, examples_path=source)
    return {"gallery": gallery, "summary": summary, "rows": len(sampled)}


def main(argv: list[str] | None = None) -> int:
    ensure_utf8()
    parser = argparse.ArgumentParser(description="Render a TreeVQG-Bench sample as HTML.")
    parser.add_argument("--examples", required=True, help="bench examples.jsonl")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--types", nargs="+", default=None)
    parser.add_argument("--per-type", type=int, default=3)
    parser.add_argument("--max-total", type=int, default=None)
    parser.add_argument("--title", default="TreeVQG-Bench v1 — sample gallery")
    args = parser.parse_args(argv)

    result = visualize(
        args.examples,
        out_dir=args.out_dir,
        types=args.types,
        per_type=args.per_type,
        max_total=args.max_total,
        title=args.title,
    )
    print(f"[viz] {result['rows']} row(s) -> {result['gallery']}")
    print(f"[viz] summary -> {result['summary']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
