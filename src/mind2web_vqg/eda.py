"""EDA over the Mind2Web HTML tree: which element types will dominate the data?

The VQG task asks about "the element the task needs". If questions are later synthesized
from *every* element of the page tree (instead of only the recorded target), the tag mix of
the tree becomes the tag mix of the training data. This module measures that mix first, so
the bias is a number instead of a surprise.

Per split it reports:

1. corpus shape — tasks, steps, HTML coverage, page size;
2. **element-type distribution** over the tree (count, share, share of steps containing it);
3. **tree vs reality** — the recorded action targets (``is_original_target``) with a *lift*
   column: a type rare in the tree but common among targets is one the agent is forced to use;
4. **VQG potential per type** — can the element carry a label, is the tag mapped to a human
   word, and why region questions cannot be synthesized from the tree (no coordinates);
5. other aspects — operations, input types, task/step lengths, sites, candidates, repeated
   page states.

Usage::

    uv run m2w-eda --split test_task --limit 200
    uv run m2w-eda --split train --limit 500 --html raw
    uv run m2w-eda --split test_task --limit 50 --out outputs/eda/quick
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import statistics
import sys
from collections import Counter
from collections.abc import Iterable
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from mind2web_vqg.cli import ensure_utf8
from mind2web_vqg.data.load import SPLITS, available_splits, iter_split
from mind2web_vqg.data.multimodal import as_json
from mind2web_vqg.data.vqg import ROLE_BY_TAG
from mind2web_vqg.paths import OUTPUT_DIR

#: ``--html`` -> the action field it reads.
HTML_FIELDS = {"cleaned": "cleaned_html", "raw": "raw_html"}

#: Tags that never contain children (so the parser must not push them).
VOID_TAGS = frozenset(
    {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }
)

#: Subtrees whose content is not page content (and whose children would flood the counts).
SKIP_TAGS = frozenset({"script", "style", "noscript", "template", "svg", "head"})

#: Attributes that give an element a human-readable name.
LABEL_ATTRS = ("aria-label", "title", "alt", "placeholder", "data-placeholder")


# --------------------------------------------------------------------------- #
# HTML parsing (stdlib only: the project keeps its dependency list short)
# --------------------------------------------------------------------------- #
class _TagCollector(HTMLParser):
    """Count tags, labelled elements and ``<input type=...>`` values in one page."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tags: Counter[str] = Counter()
        self.labelled: Counter[str] = Counter()
        self.input_types: Counter[str] = Counter()
        self.elements = 0
        self._open: list[list[Any]] = []  # [tag, has_label]
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if self._skip_depth:
            if tag in SKIP_TAGS:
                self._skip_depth += 1
            return
        if tag in SKIP_TAGS:
            self._skip_depth = 1
            return

        self.elements += 1
        self.tags[tag] += 1
        values = {key.lower(): value for key, value in attrs}
        labelled = self._has_label(tag, values)
        if tag == "input":
            self.input_types[(values.get("type") or "text").strip().lower()] += 1

        if tag in VOID_TAGS:
            if labelled:
                self.labelled[tag] += 1
            return
        self._open.append([tag, labelled])

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if self._skip_depth:
            if tag in SKIP_TAGS:
                self._skip_depth -= 1
            return
        for index in range(len(self._open) - 1, -1, -1):
            if self._open[index][0] == tag:
                self._close_from(index)
                return

    def handle_data(self, data: str) -> None:
        if self._skip_depth or not data.strip():
            return
        for entry in self._open:
            entry[1] = True

    def close(self) -> None:
        super().close()
        if self._open:
            self._close_from(0)

    def _close_from(self, index: int) -> None:
        for entry in self._open[index:]:
            if entry[1]:
                self.labelled[entry[0]] += 1
        del self._open[index:]

    @staticmethod
    def _has_label(tag: str, attrs: dict[str, str | None]) -> bool:
        if any((attrs.get(name) or "").strip() for name in LABEL_ATTRS):
            return True
        return tag in {"input", "button", "option"} and bool((attrs.get("value") or "").strip())


def parse_html(fragment: str) -> dict[str, Any]:
    """Count tags / labels / input types in one HTML string."""
    collector = _TagCollector()
    try:
        collector.feed(fragment or "")
        collector.close()
    except Exception:  # a malformed dump must not abort the whole EDA run
        collector.close()
    return {
        "tags": dict(collector.tags),
        "labelled": dict(collector.labelled),
        "input_types": dict(collector.input_types),
        "elements": collector.elements,
    }


# --------------------------------------------------------------------------- #
# Recorded targets
# --------------------------------------------------------------------------- #
def recorded_target(action: dict[str, Any]) -> dict[str, Any]:
    """The ``is_original_target`` candidate of one action (``{}`` when missing)."""
    candidates = [
        candidate
        for candidate in (as_json(raw, {}) for raw in action.get("pos_candidates") or [])
        if isinstance(candidate, dict)
    ]
    for candidate in candidates:
        if candidate.get("is_original_target"):
            return candidate
    return candidates[0] if candidates else {}


def _target_label(attrs: dict[str, Any]) -> str:
    return str(attrs.get("text") or attrs.get("aria_label") or attrs.get("title") or "").strip()


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def _share(part: int, whole: int) -> float:
    return round(part / whole, 4) if whole else 0.0


def _percentile(values: list[int], q: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1)))]


def summarize(
    instances: Iterable[dict[str, Any]],
    *,
    split: str = "?",
    html_field: str = "cleaned",
) -> dict[str, Any]:
    """Aggregate tree/target/operation statistics over an iterable of instances."""
    field = HTML_FIELDS.get(html_field, html_field)

    tags: Counter[str] = Counter()
    labelled: Counter[str] = Counter()
    input_types: Counter[str] = Counter()
    presence: Counter[str] = Counter()
    target_tags: Counter[str] = Counter()
    ops: Counter[str] = Counter()
    with_value: Counter[str] = Counter()
    websites: Counter[str] = Counter()
    domains: Counter[str] = Counter()
    subdomains: Counter[str] = Counter()

    tasks = steps = steps_with_html = parse_errors = tasks_without_html = 0
    elements = target_labels = target_bboxes = 0
    html_sizes: list[int] = []
    steps_per_task: list[int] = []
    task_words: list[int] = []
    pos_counts: list[int] = []
    neg_counts: list[int] = []
    seen_html: set[str] = set()
    repeated_steps = 0

    for instance in instances:
        tasks += 1
        actions = instance.get("actions") or []
        steps_per_task.append(len(actions))
        task_words.append(len(str(instance.get("confirmed_task") or "").split()))
        websites[str(instance.get("website") or "?")] += 1
        domains[str(instance.get("domain") or "?")] += 1
        subdomains[str(instance.get("subdomain") or "?")] += 1

        html_here = 0
        for action in actions:
            steps += 1
            operation = as_json(action.get("operation"), {}) or {}
            op = str(operation.get("op") or "?")
            ops[op] += 1
            if str(operation.get("value") or "").strip():
                with_value[op] += 1

            pos_counts.append(len(action.get("pos_candidates") or []))
            neg_counts.append(len(action.get("neg_candidates") or []))

            target = recorded_target(action)
            tag = str(target.get("tag") or "").strip().lower()
            attrs = as_json(target.get("attributes"), {}) or {}
            if tag:
                target_tags[tag] += 1
            if _target_label(attrs):
                target_labels += 1
            if attrs.get("bounding_box_rect"):
                target_bboxes += 1

            fragment = action.get(field) or ""
            if not fragment:
                continue
            html_here += 1
            html_sizes.append(len(fragment))
            digest = hashlib.md5(fragment.encode("utf-8", "ignore")).hexdigest()
            if digest in seen_html:
                repeated_steps += 1
            else:
                seen_html.add(digest)

            parsed = parse_html(fragment)
            elements += parsed["elements"]
            tags.update(parsed["tags"])
            labelled.update(parsed["labelled"])
            input_types.update(parsed["input_types"])
            presence.update(parsed["tags"].keys())

        steps_with_html += html_here
        if not html_here:
            tasks_without_html += 1

    questionable_tags = set(ROLE_BY_TAG) | {"input"}
    mapped_elements = sum(count for tag, count in tags.items() if tag in questionable_tags)
    label_elements = sum(labelled[tag] for tag in tags if tag in questionable_tags)

    tag_rows = [
        {
            "tag": tag,
            "count": count,
            "share": _share(count, elements),
            "steps": presence.get(tag, 0),
            "step_share": _share(presence.get(tag, 0), steps_with_html),
        }
        for tag, count in tags.most_common()
    ]

    questionable_rows = []
    for tag, count in tags.most_common():
        if tag not in questionable_tags:
            continue
        questionable_rows.append(
            {
                "tag": tag,
                "role": "input field (by type)" if tag == "input" else ROLE_BY_TAG[tag],
                "count": count,
                "share": _share(count, elements),
                "share_of_questionable": _share(count, mapped_elements),
                "steps": presence.get(tag, 0),
                "step_share": _share(presence.get(tag, 0), steps_with_html),
                "labelled": labelled.get(tag, 0),
                "label_rate": _share(labelled.get(tag, 0), count),
            }
        )

    target_total = sum(target_tags.values())
    target_rows = []
    for tag, count in target_tags.most_common():
        target_share = _share(count, target_total)
        tree_share = _share(tags.get(tag, 0), elements)
        target_rows.append(
            {
                "tag": tag,
                "count": count,
                "share": target_share,
                "tree_share": tree_share,
                "lift": round(target_share / tree_share, 2) if tree_share else None,
            }
        )

    return {
        "split": split,
        "html_field": field,
        "tasks": tasks,
        "steps": steps,
        "steps_with_html": steps_with_html,
        "tasks_without_html": tasks_without_html,
        "parse_errors": parse_errors,
        "elements": elements,
        "elements_per_step": round(elements / steps_with_html, 1) if steps_with_html else 0.0,
        "html_chars": {
            "mean": int(statistics.fmean(html_sizes)) if html_sizes else 0,
            "median": _percentile(html_sizes, 0.5),
            "p95": _percentile(html_sizes, 0.95),
            "max": max(html_sizes) if html_sizes else 0,
        },
        "mean_steps_per_task": (
            round(statistics.fmean(steps_per_task), 2) if steps_per_task else 0.0
        ),
        "mean_task_words": round(statistics.fmean(task_words), 1) if task_words else 0.0,
        "tags": tag_rows,
        "questionable": questionable_rows,
        "input_types": dict(input_types.most_common()),
        "targets": {
            "n": target_total,
            "label_n": target_labels,
            "bbox_n": target_bboxes,
            "label_rate": _share(target_labels, steps),
            "bbox_rate": _share(target_bboxes, steps),
            "tags": target_rows,
        },
        "ops": {
            op: {
                "count": count,
                "share": _share(count, steps),
                "with_value": with_value.get(op, 0),
            }
            for op, count in ops.most_common()
        },
        "sites": {
            "websites": dict(websites.most_common(15)),
            "domains": dict(domains.most_common()),
            "subdomains": dict(subdomains.most_common(15)),
        },
        "candidates": {
            "mean_pos": round(statistics.fmean(pos_counts), 1) if pos_counts else 0.0,
            "mean_neg": round(statistics.fmean(neg_counts), 1) if neg_counts else 0.0,
        },
        "html_repeated_steps": repeated_steps,
        "html_unique_steps": len(seen_html),
        "potential": {
            # One question per element in the tree is possible for these two categories.
            "target_label": label_elements,
            "element_type": mapped_elements,
            # The tree carries no coordinates: region questions need the recorded target.
            "target_region": target_bboxes,
        },
    }


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def render_markdown(result: dict[str, Any], *, top: int = 25) -> str:
    """A readable EDA report from :func:`summarize`."""
    questionable = result["questionable"]
    top_questionable = questionable[0] if questionable else None
    biggest = result["tags"][0] if result["tags"] else None
    lifts = [row for row in result["targets"]["tags"] if row["lift"] is not None]

    lines = [
        f"# EDA — {result['split']} ({result['html_field']})",
        "",
        "## Findings",
        "",
        f"- **{result['tasks']} tasks / {result['steps']} steps**, "
        f"{result['steps_with_html']} with HTML "
        f"({result['elements']} elements, {result['elements_per_step']} per step).",
    ]
    if biggest:
        lines.append(
            f"- The tree is dominated by `<{biggest['tag']}>`: {_pct(biggest['share'])} of all "
            f"elements, present in {_pct(biggest['step_share'])} of steps."
        )
    if top_questionable:
        lines.append(
            f"- Among elements that can carry a question, `<{top_questionable['tag']}>` is "
            f"{_pct(top_questionable['share_of_questionable'])} of the pool "
            f"({_pct(top_questionable['step_share'])} of steps contain one). Uncontrolled "
            "synthesis would be biased toward it."
        )
    if lifts:
        most = max(lifts, key=lambda row: row["lift"] or 0)
        least = min(lifts, key=lambda row: row["lift"] or 0)
        lines.append(
            f"- Recorded targets differ from the tree: `<{most['tag']}>` is "
            f"{most['lift']}× over-represented, `<{least['tag']}>` {least['lift']}×."
        )
    lines.append(
        f"- Label availability: {result['potential']['target_label']} of "
        f"{result['potential']['element_type']} question-capable elements have text or a "
        "label attribute."
    )
    lines.append(
        f"- Region questions **cannot** be synthesized from the tree (no coordinates): only "
        f"{result['targets']['n']} recorded targets exist, and "
        f"{_pct(result['targets']['bbox_rate'])} of steps carry a bounding box."
    )

    lines += [
        "",
        "## Corpus shape",
        "",
        _table(
            ["Metric", "Value"],
            [
                ["Tasks", result["tasks"]],
                ["Steps (actions)", result["steps"]],
                ["Steps with HTML", result["steps_with_html"]],
                ["Tasks without HTML", result["tasks_without_html"]],
                ["Elements (tree)", result["elements"]],
                ["Elements / step", result["elements_per_step"]],
                ["Steps / task", result["mean_steps_per_task"]],
                ["Task words (mean)", result["mean_task_words"]],
                ["HTML chars (mean / median / p95 / max)", "{} / {} / {} / {}".format(
                    result["html_chars"]["mean"],
                    result["html_chars"]["median"],
                    result["html_chars"]["p95"],
                    result["html_chars"]["max"],
                )],
                ["Repeated page states",
                 f"{result['html_repeated_steps']} "
                 f"({_pct(_share(result['html_repeated_steps'], result['steps_with_html']))})"],
                ["Candidates per step (pos / neg)",
                 f"{result['candidates']['mean_pos']} / {result['candidates']['mean_neg']}"],
            ],
        ),
        "",
        "## Element types — whole tree",
        "",
        _table(
            ["Tag", "Count", "Share", "Steps with it", "Step share"],
            [
                [f"`{row['tag']}`", row["count"], _pct(row["share"]), row["steps"],
                 _pct(row["step_share"])]
                for row in result["tags"][:top]
            ],
        ),
        "",
        "## Element types that can carry a question",
        "",
        _table(
            ["Tag", "Role", "Count", "Share of pool", "Steps with it", "Has label"],
            [
                [f"`{row['tag']}`", row["role"], row["count"],
                 _pct(row["share_of_questionable"]), _pct(row["step_share"]),
                 _pct(row["label_rate"])]
                for row in questionable
            ],
        ),
        "",
        "## Tree vs recorded targets",
        "",
        "`lift` = share among recorded targets / share in the tree. `> 1` means the agent is "
        "forced to use that type more often than the tree would suggest.",
        "",
        _table(
            ["Tag", "Targets", "Target share", "Tree share", "Lift"],
            [
                [f"`{row['tag']}`", row["count"], _pct(row["share"]),
                 _pct(row["tree_share"]), "—" if row["lift"] is None else f"{row['lift']}×"]
                for row in result["targets"]["tags"]
            ],
        ),
        "",
        "## VQG potential",
        "",
        _table(
            ["Category", "Questions available", "Why"],
            [
                ["`target_label`", result["potential"]["target_label"],
                 "elements with text or a label attribute"],
                ["`element_type`", result["potential"]["element_type"],
                 "elements whose tag maps to a human word"],
                ["`target_region`", result["potential"]["target_region"],
                 "recorded targets with a bounding box (the tree has no coordinates)"],
            ],
        ),
        "",
        "## Input types",
        "",
        _table(
            ["`<input type=...>`", "Count"],
            [[f"`{name}`", count] for name, count in result["input_types"].items()],
        ),
        "",
        "## Operations",
        "",
        _table(
            ["Op", "Count", "Share", "With a value"],
            [
                [f"`{op}`", block["count"], _pct(block["share"]), block["with_value"]]
                for op, block in result["ops"].items()
            ],
        ),
        "",
        "## Sites",
        "",
        _table(
            ["Website", "Tasks"],
            [[f"`{name}`", count] for name, count in result["sites"]["websites"].items()],
        ),
        "",
        _table(
            ["Domain", "Tasks"],
            [[f"`{name}`", count] for name, count in result["sites"]["domains"].items()],
        ),
        "",
        "## Method",
        "",
        "Counted with the stdlib `html.parser` over `actions[*]."
        f"{result['html_field']}`, skipping `script`/`style`/`noscript`/`template`/`svg`/"
        "`head`. An element counts as *labelled* when it has non-empty text content or one of "
        "`aria-label`/`title`/`alt`/`placeholder`/`data-placeholder` (plus `value` for "
        "input/button/option). Targets are the `is_original_target` candidate of each action.",
    ]
    return "\n".join(lines) + "\n"


def render_summary(results: list[dict[str, Any]]) -> str:
    """Cross-split comparison (one row per split plus a combined ``all`` row)."""
    rows: list[list[Any]] = []
    for result in results:
        top_tag = result["tags"][0]["tag"] if result["tags"] else "—"
        top_question = result["questionable"][0] if result["questionable"] else None
        rows.append(
            [
                f"`{result['split']}`",
                result["tasks"],
                result["steps"],
                result["elements"],
                result["elements_per_step"],
                f"`{top_tag}`",
                f"`{top_question['tag']}` "
                f"({_pct(top_question['share_of_questionable'])})"
                if top_question
                else "—",
                _pct(result["targets"]["label_rate"]),
                _pct(result["targets"]["bbox_rate"]),
                result["potential"]["target_label"],
                result["potential"]["element_type"],
            ]
        )

    steps = sum(result["steps"] for result in results)
    steps_with_html = sum(result["steps_with_html"] for result in results)
    elements = sum(result["elements"] for result in results)
    label_n = sum(result["targets"]["label_n"] for result in results)
    bbox_n = sum(result["targets"]["bbox_n"] for result in results)
    rows.append(
        [
            "**all**",
            sum(result["tasks"] for result in results),
            steps,
            elements,
            round(elements / steps_with_html, 1) if steps_with_html else 0.0,
            "—",
            "—",
            _pct(_share(label_n, steps)),
            _pct(_share(bbox_n, steps)),
            sum(result["potential"]["target_label"] for result in results),
            sum(result["potential"]["element_type"] for result in results),
        ]
    )

    lines = [
        "# EDA — all splits",
        "",
        f"Combined view of {len(results)} split(s). Per-split tables live in "
        "`eda/<split>/eda.md`.",
        "",
        _table(
            [
                "Split",
                "Tasks",
                "Steps",
                "Elements",
                "Elem/step",
                "Top tag",
                "Top question type",
                "Target has label",
                "Target has bbox",
                "Label questions",
                "Type questions",
            ],
            rows,
        ),
        "",
        "## Top question-capable types per split",
        "",
    ]
    for result in results:
        top = result["questionable"][:5]
        if not top:
            continue
        lines += [
            f"**{result['split']}** — " + ", ".join(
                f"`{row['tag']}` {_pct(row['share_of_questionable'])}" for row in top
            ),
            "",
        ]
    lines += [
        "## Reading this",
        "",
        "- `Target has label` / `Target has bbox` are shares of **recorded steps** whose "
        "target carries that attribute — the ceiling for `target_label` / `target_region`.",
        "- `Label questions` / `Type questions` are how many questions the **tree** could "
        "support if you synthesized from every element (not only the recorded target).",
        "- Region questions cannot be synthesized from the tree (it has no coordinates).",
    ]
    return "\n".join(lines) + "\n"


def write_report(result: dict[str, Any], out_dir: str | Path) -> tuple[Path, Path, Path]:
    """Write ``eda.json``, ``eda.md`` and ``tags.csv``; return the three paths."""
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)

    data_path = directory / "eda.json"
    data_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    markdown_path = directory / "eda.md"
    markdown_path.write_text(render_markdown(result), encoding="utf-8")

    csv_path = directory / "tags.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tag", "count", "share", "steps", "step_share", "labelled", "role"])
        roles = {row["tag"]: row["role"] for row in result["questionable"]}
        labelled = {row["tag"]: row["labelled"] for row in result["questionable"]}
        for row in result["tags"]:
            writer.writerow(
                [
                    row["tag"],
                    row["count"],
                    row["share"],
                    row["steps"],
                    row["step_share"],
                    labelled.get(row["tag"], ""),
                    roles.get(row["tag"], ""),
                ]
            )
    return data_path, markdown_path, csv_path


def main(argv: list[str] | None = None) -> int:
    ensure_utf8()
    parser = argparse.ArgumentParser(
        description="EDA over the Mind2Web HTML tree (element types, targets, yield)."
    )
    parser.add_argument("--split", default="test_task", choices=SPLITS)
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="max tasks per split; 0 = the whole split (default 200)",
    )
    parser.add_argument("--html", default="cleaned", choices=sorted(HTML_FIELDS),
                        help="which DOM dump to parse (default: cleaned)")
    parser.add_argument(
        "--all-splits",
        action="store_true",
        help="run every downloaded split and also write eda/summary.md",
    )
    parser.add_argument("--root", type=Path, default=None, help="JSON data root")
    parser.add_argument("--out", type=Path, default=None, help="default: outputs/eda/<split>")
    args = parser.parse_args(argv)

    def instances(split: str) -> Iterable[dict[str, Any]]:
        stream: Iterable[dict[str, Any]] = (
            iter_split(split, root=args.root) if args.root else iter_split(split)
        )
        return itertools.islice(stream, args.limit) if args.limit else stream

    out_root = args.out or OUTPUT_DIR / "eda"

    if args.all_splits:
        splits = available_splits(args.root) if args.root else available_splits()
        if not splits:
            print("[eda] no downloaded split found — run `uv run m2w-download` first")
            return 1
        results = []
        for index, split in enumerate(splits, start=1):
            print(f"[eda] ({index}/{len(splits)}) {split}")
            result = summarize(instances(split), split=split, html_field=args.html)
            write_report(result, out_root / split)
            print(
                f"  {result['tasks']} task(s), {result['steps']} step(s), "
                f"{result['elements']} element(s)"
            )
            results.append(result)

        summary_md = out_root / "summary.md"
        summary_md.write_text(render_summary(results), encoding="utf-8")
        summary_json = out_root / "summary.json"
        summary_json.write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[eda] {len(results)} split(s) -> {summary_md}")
        return 0

    result = summarize(instances(args.split), split=args.split, html_field=args.html)
    data_path, markdown_path, csv_path = write_report(result, out_root / args.split)

    print(f"[eda] {result['tasks']} task(s), {result['steps']} step(s), "
          f"{result['elements']} element(s)")
    print(f"  {markdown_path}")
    print(f"  {data_path}")
    print(f"  {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
