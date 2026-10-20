"""A LaTeX report for the EDA: tables, findings and pgfplots charts in one ``.tex``.

Reads what ``m2w-eda`` already wrote (``outputs/eda/<split>/eda.json`` plus the merged
``outputs/eda/all/eda.json``), derives the numbers a sampling policy needs, and emits a
self-contained ``report.tex``:

- no extra dependency — charts are ``pgfplots``, not bitmaps, so the file is the only artefact;
- every number comes from ``eda.json``, so the report can be regenerated from a finished run;
- ``--compile`` runs ``latexmk``/``pdflatex`` when a TeX distribution is installed.

Usage::

    uv run m2w-eda --all-splits --limit 0        # first: produce the JSON
    uv run m2w-eda-report --compile              # -> outputs/eda/report/report.{tex,pdf}

    uv run m2w-eda-report --budget 50000         # size the sampling table differently
    uv run m2w-eda-report --split test_task      # one split only
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from mind2web_vqg.cli import ensure_utf8
from mind2web_vqg.data.load import SPLITS
from mind2web_vqg.paths import OUTPUT_DIR

#: Characters that must be escaped in LaTeX text mode (single pass, so ``\\`` is safe).
_TEX_TABLE = str.maketrans(
    {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
)

#: Tags highlighted by name in the scatter plots (the rest stay unlabelled).
_LABELLED_POINTS = 6


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #
def tex(text: Any) -> str:
    """Escape a value for LaTeX text mode."""
    return str(text).translate(_TEX_TABLE)


def num(value: float | int) -> str:
    """Thousands-separated integer, or two decimals for a non-integer."""
    if isinstance(value, int) or float(value).is_integer():
        return f"{int(value):,}"
    return f"{value:,.2f}"


def whole(value: float) -> str:
    """Rounded, thousands-separated value (for question budgets)."""
    return f"{round(value):,}"


def pct(value: float, digits: int = 1) -> str:
    """Percent with an escaped ``%`` sign."""
    return f"{value * 100:.{digits}f}\\,\\%"


def ratio(value: float) -> str:
    """Two-decimal multiplier."""
    return f"{value:.2f}"


def tag(value: str) -> str:
    """A tag name in monospace, escaped."""
    return rf"\texttt{{{tex(value)}}}"


# --------------------------------------------------------------------------- #
# Reading the EDA output
# --------------------------------------------------------------------------- #
def find_reports(root: str | Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any] | None]:
    """Load ``<root>/<name>/eda.json``; return ``(per-split reports, merged report)``."""
    base = Path(root)
    if not base.is_dir():
        raise FileNotFoundError(
            f"{base} not found. Run `uv run m2w-eda --all-splits --limit 0` first."
        )

    found: dict[str, dict[str, Any]] = {}
    for path in sorted(base.glob("*/eda.json")):
        found[path.parent.name] = json.loads(path.read_text(encoding="utf-8"))
    if not found:
        raise FileNotFoundError(
            f"no eda.json under {base}/*. Run `uv run m2w-eda --all-splits --limit 0` first."
        )

    merged = found.pop("all", None)
    splits = {name: found[name] for name in SPLITS if name in found}
    splits.update({name: value for name, value in found.items() if name not in splits})
    return splits, merged


def concentration(shares: Iterable[float]) -> dict[str, Any]:
    """Herfindahl index and how many types it takes to cover 50% / 80% of the pool."""
    ordered = sorted((share for share in shares if share > 0), reverse=True)
    total = sum(ordered) or 1.0
    hhi = sum((share / total) ** 2 for share in ordered)

    def coverage(target: float) -> int:
        running = 0.0
        for index, share in enumerate(ordered, start=1):
            running += share / total
            if running >= target:
                return index
        return len(ordered)

    return {
        "n": len(ordered),
        "hhi": hhi,
        "effective": 1 / hhi if hhi else 0.0,
        "tags_for_50": coverage(0.5),
        "tags_for_80": coverage(0.8),
    }


def policy_rows(result: dict[str, Any], budget: int) -> dict[str, Any]:
    """Compare uncontrolled sampling with an equal budget per type.

    ``uncontrolled`` = the number of questions a type gets if you sample questions from the
    tree in proportion to how often the type occurs. ``cap`` = ``min(labelled, budget/n_types)``,
    i.e. an equal share per type, limited by what the corpus actually holds.
    """
    rows = result["questionable"]
    n_types = len(rows) or 1
    even = budget / n_types

    out = []
    for row in rows:
        share = row["share_of_questionable"]
        available = row["labelled"]
        uncontrolled = share * budget
        cap = min(available, even)
        out.append(
            {
                "tag": row["tag"],
                "role": row["role"],
                "count": row["count"],
                "share": share,
                "available": available,
                "uncontrolled": uncontrolled,
                "cap": cap,
                "factor": (cap / uncontrolled) if uncontrolled else 0.0,
            }
        )

    shortfall = sum(max(0.0, even - row["cap"]) for row in out)
    return {
        "rows": out,
        "budget": budget,
        "even": even,
        "shortfall": shortfall,
        "shortfall_share": (shortfall / budget) if budget else 0.0,
    }


# --------------------------------------------------------------------------- #
# LaTeX building blocks
# --------------------------------------------------------------------------- #
def _table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    align: str,
    caption: str,
    label: str,
) -> str:
    """A booktabs table in a float."""
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    lines += [" & ".join(str(cell) for cell in row) + r" \\" for row in rows]
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def _figure(body: str, *, caption: str, label: str) -> str:
    return "\n".join(
        [
            r"\begin{figure}[H]",
            r"\centering",
            body,
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            r"\end{figure}",
        ]
    )


def _yticklabels(names: Sequence[str]) -> str:
    return ", ".join("{" + tag(name) + "}" for name in names)


def _tree_figure(result: dict[str, Any], top: int = 12) -> str:
    rows = result["tags"][:top]
    if not rows:
        return ""
    coords = " ".join(f"({row['count']},{index})" for index, row in enumerate(rows, start=1))
    body = "\n".join(
        [
            r"\begin{tikzpicture}",
            r"\begin{axis}[",
            r"  width=0.95\linewidth, height=7.2cm,",
            r"  xbar, bar width=5pt, y dir=reverse,",
            rf"  ytick={{1,...,{len(rows)}}},",
            rf"  yticklabels={{{_yticklabels([row['tag'] for row in rows])}}},",
            r"  yticklabel style={font=\scriptsize}, xticklabel style={font=\scriptsize},",
            rf"  ymin=0.4, ymax={len(rows) + 0.6},",
            r"  xmin=0,",
            r"  xlabel={elements in the tree},",
            r"  nodes near coords,",
            r"  nodes near coords style={font=\tiny, anchor=south},",
            r"  every axis plot/.append style={fill=accent!55, draw=none},",
            r"]",
            rf"\addplot coordinates {{{coords}}};",
            r"\end{axis}",
            r"\end{tikzpicture}",
        ]
    )
    return _figure(
        body,
        caption=(
            f"The {len(rows)} most frequent element types in "
            f"{tag(result['html_field'])}. A linear axis keeps the head of the distribution "
            "honest: an uncontrolled sampler would reproduce exactly this shape."
        ),
        label="fig:tree",
    )


def _pool_figure(result: dict[str, Any], top: int = 10) -> str:
    rows = result["questionable"][:top]
    if not rows:
        return ""
    coords = " ".join(
        f"({row['share_of_questionable'] * 100:.3f},{index})"
        for index, row in enumerate(rows, start=1)
    )
    body = "\n".join(
        [
            r"\begin{tikzpicture}",
            r"\begin{axis}[",
            r"  width=0.95\linewidth, height=6.4cm,",
            r"  xbar, bar width=5pt, y dir=reverse,",
            rf"  ytick={{1,...,{len(rows)}}},",
            rf"  yticklabels={{{_yticklabels([row['tag'] for row in rows])}}},",
            r"  yticklabel style={font=\scriptsize}, xticklabel style={font=\scriptsize},",
            rf"  ymin=0.4, ymax={len(rows) + 0.6},",
            r"  xlabel={share of the question-capable pool (\%)},",
            r"  nodes near coords,",
            r"  nodes near coords={\pgfmathprintnumber[fixed,precision=1]{\pgfplotspointmeta}},",
            r"  nodes near coords style={font=\tiny, anchor=south},",
            r"  every axis plot/.append style={fill=accent!55, draw=none},",
            r"]",
            rf"\addplot coordinates {{{coords}}};",
            r"\end{axis}",
            r"\end{tikzpicture}",
        ]
    )
    return _figure(
        body,
        caption=(
            "Composition of the question-capable pool. Every element of these types can carry "
            "an \\texttt{element\\_type} question; the leader is the type a naive sampler "
            "over-produces."
        ),
        label="fig:pool",
    )


def _label_figure(result: dict[str, Any]) -> str:
    rows = [row for row in result["questionable"] if row["count"] > 0]
    if not rows:
        return ""
    coords = " ".join(
        f"({row['share_of_questionable'] * 100:.3f},{row['label_rate'] * 100:.3f})"
        for row in rows
    )
    nodes = "\n".join(
        rf"\node[font=\tiny, anchor=south west, text=black!70] at "
        rf"(axis cs:{row['share_of_questionable'] * 100:.3f},"
        rf"{row['label_rate'] * 100:.3f}) {{{tag(row['tag'])}}};"
        for row in rows[:_LABELLED_POINTS]
    )
    # Fit the x axis to the data: a fixed 0-100 range squeezes every point into the corner.
    xmax = max(row["share_of_questionable"] for row in rows) * 100 * 1.15
    body = "\n".join(
        [
            r"\begin{tikzpicture}",
            r"\begin{axis}[",
            r"  width=0.95\linewidth, height=7.0cm,",
            r"  only marks, mark=*, mark size=1.6pt, color=accent,",
            r"  xlabel={share of the question-capable pool (\%)},",
            r"  ylabel={elements with a label (\%)},",
            rf"  xmin=0, xmax={xmax:.2f}, ymin=0, ymax=105,",
            r"  grid=major, grid style={black!12},",
            r"  ticklabel style={font=\scriptsize}, label style={font=\small},",
            r"]",
            rf"\addplot coordinates {{{coords}}};",
            nodes,
            r"\end{axis}",
            r"\end{tikzpicture}",
        ]
    )
    return _figure(
        body,
        caption=(
            "Pool share against label availability. Points high on the y axis are cheap "
            "\\texttt{target\\_label} questions; points on the right are the types that would "
            "dominate without a cap."
        ),
        label="fig:label",
    )


def _lift_figure(result: dict[str, Any]) -> str:
    rows = [row for row in result["targets"]["tags"] if row["tree_share"] > 0]
    if not rows:
        return ""
    coords = " ".join(f"({row['tree_share']:.5f},{row['share']:.5f})" for row in rows)
    ranked = sorted(rows, key=lambda row: row["lift"] or 0, reverse=True)
    nodes = "\n".join(
        rf"\node[font=\tiny, anchor=south west, text=black!70] at "
        rf"(axis cs:{row['tree_share']:.5f},{row['share']:.5f}) {{{tag(row['tag'])}}};"
        for row in ranked[:_LABELLED_POINTS]
    )
    # One shared range for both axes keeps the dashed line a true y = x; derive it from the
    # data so the points are not squeezed into a 1e-4 .. 1e0 corner.
    values = [value for row in rows for value in (row["tree_share"], row["share"])]
    low = max(min(values) / 3, 1e-6)
    high = max(values) * 3
    body = "\n".join(
        [
            r"\begin{tikzpicture}",
            r"\begin{axis}[",
            r"  width=0.95\linewidth, height=7.4cm,",
            r"  xmode=log, ymode=log,",
            r"  xlabel={share of the tree (log)}, ylabel={share of recorded targets (log)},",
            rf"  xmin={low:.6f}, xmax={high:.6f}, ymin={low:.6f}, ymax={high:.6f},",
            r"  grid=major, grid style={black!12},",
            r"  ticklabel style={font=\scriptsize}, label style={font=\small},",
            r"  legend pos=north west, legend style={font=\scriptsize},",
            r"]",
            rf"\addplot[dashed, draw=black!55, domain={low:.6f}:{high:.6f}] {{x}};",
            r"\addlegendentry{no bias}",
            r"\addplot[only marks, mark=*, mark size=1.6pt, color=accent] "
            rf"coordinates {{{coords}}};",
            r"\addlegendentry{recorded targets}",
            nodes,
            r"\end{axis}",
            r"\end{tikzpicture}",
        ]
    )
    return _figure(
        body,
        caption=(
            "Recorded targets against the tree. Above the dashed line the agent uses a type "
            "more often than the tree suggests (the task forces it); below the line the tree "
            "is full of it but it rarely matters."
        ),
        label="fig:lift",
    )


def _potential_figure(result: dict[str, Any]) -> str:
    potential = result["potential"]
    coords = " ".join(
        [
            f"(label,{potential['target_label']})",
            f"(type,{potential['element_type']})",
            f"(region,{potential['target_region']})",
        ]
    )
    body = "\n".join(
        [
            r"\begin{tikzpicture}",
            r"\begin{axis}[",
            r"  width=0.8\linewidth, height=5.6cm,",
            r"  ybar, bar width=18pt,",
            r"  symbolic x coords={label,type,region}, xtick=data,",
            r"  xticklabels={{label},{type},{region}},",
            r"  xticklabel style={font=\small}, yticklabel style={font=\scriptsize},",
            r"  ylabel={questions available}, ymin=0,",
            r"  nodes near coords,",
            r"  nodes near coords={\pgfmathprintnumber[fixed,precision=0]{\pgfplotspointmeta}},",
            r"  nodes near coords style={font=\scriptsize},",
            r"  every axis plot/.append style={fill=accent!55, draw=none},",
            r"]",
            rf"\addplot coordinates {{{coords}}};",
            r"\end{axis}",
            r"\end{tikzpicture}",
        ]
    )
    return _figure(
        body,
        caption=(
            "Questions the tree can support, by category. \\texttt{region} is bounded by the "
            "recorded targets (the tree carries no coordinates); the other two by the elements "
            "themselves."
        ),
        label="fig:potential",
    )


def _split_figure(splits: dict[str, dict[str, Any]]) -> str:
    """One small single-series panel per split.

    Deliberately *not* a grouped bar chart: several ``\\addplot``s inside one ``ybar`` axis
    send pgfplots 1.18 into an infinite loop with ``symbolic x coords`` (verified: one plot
    compiles, two do not). Small multiples avoid the construct entirely.
    """
    if len(splits) < 2:
        return ""

    panels = []
    for name, result in splits.items():
        scale = 1000 / result["steps"] if result["steps"] else 0.0
        coords = " ".join(
            [
                f"(label,{result['potential']['target_label'] * scale:.2f})",
                f"(type,{result['potential']['element_type'] * scale:.2f})",
                f"(region,{result['potential']['target_region'] * scale:.2f})",
            ]
        )
        panels.append(
            "\n".join(
                [
                    r"\begin{tikzpicture}",
                    r"\begin{axis}[",
                    r"  width=\linewidth, height=4.4cm,",
                    r"  ybar, bar width=13pt,",
                    r"  symbolic x coords={label,type,region}, xtick=data,",
                    r"  xticklabels={{label},{type},{region}},",
                    rf"  title={{{tag(name)}}},",
                    r"  title style={font=\small},",
                    r"  xticklabel style={font=\scriptsize},",
                    r"  yticklabel style={font=\tiny},",
                    r"  ylabel={per 1000 steps}, ylabel style={font=\scriptsize}, ymin=0,",
                    r"  scaled y ticks=false,",
                    r"  yticklabel={\pgfmathprintnumber[fixed,precision=0]{\tick}},",
                    r"  every axis plot/.append style={fill=accent!55, draw=none},",
                    r"]",
                    rf"\addplot coordinates {{{coords}}};",
                    r"\end{axis}",
                    r"\end{tikzpicture}",
                ]
            )
        )

    rows = []
    for index in range(0, len(panels), 2):
        pair = panels[index : index + 2]
        cells = [
            r"\begin{minipage}[t]{0.48\linewidth}",
            r"\centering",
            pair[0],
            r"\end{minipage}",
        ]
        if len(pair) > 1:
            cells += [
                r"\hfill",
                r"\begin{minipage}[t]{0.48\linewidth}",
                r"\centering",
                pair[1],
                r"\end{minipage}",
            ]
        rows.append("\n".join(cells))

    return _figure(
        "\n\n".join(rows),
        caption=(
            "Question yield per 1000 steps, split by split. Normalising by steps makes splits "
            "of different sizes comparable: an uneven panel means the same sampler would "
            "behave differently on each test set."
        ),
        label="fig:splits",
    )


# --------------------------------------------------------------------------- #
# The document
# --------------------------------------------------------------------------- #
def _preamble(title: str) -> str:
    return "\n".join(
        [
            r"\documentclass[11pt,a4paper]{article}",
            r"\usepackage[margin=2.2cm]{geometry}",
            r"\usepackage[T1]{fontenc}",
            r"\usepackage{lmodern}",
            r"\usepackage{booktabs,array}",
            r"\usepackage{amsmath}",
            r"\usepackage{pgfplots}",
            r"\pgfplotsset{compat=1.18}",
            r"\usepackage{xcolor}",
            r"\definecolor{accent}{HTML}{1F4E79}",
            r"\usepackage{caption}",
            r"\captionsetup{font=small,labelfont=bf}",
            r"\usepackage{float}",
            r"\usepackage[hidelinks]{hyperref}",
            r"\setlength{\parskip}{0.35em}",
            r"\setlength{\parindent}{0pt}",
            rf"\title{{{tex(title)}}}",
            r"\author{Mind2Web-VQG --- automatic EDA}",
            r"\date{\today}",
            r"\begin{document}",
            r"\maketitle",
        ]
    )


def _findings(result: dict[str, Any]) -> str:
    biggest = result["tags"][0] if result["tags"] else None
    pool = result["questionable"]
    pooled = pool[0] if pool else None
    conc = concentration(row["share_of_questionable"] for row in pool)
    labels = result["potential"]["target_label"]
    types = result["potential"]["element_type"]
    lifts = [row for row in result["targets"]["tags"] if row["lift"] is not None]
    most = max(lifts, key=lambda row: row["lift"] or 0) if lifts else None
    least = min(lifts, key=lambda row: row["lift"] or 0) if lifts else None

    bullets = [
        (
            f"The corpus holds {num(result['tasks'])} tasks and {num(result['steps'])} steps; "
            f"{num(result['elements'])} elements were parsed from "
            f"{tag(result['html_field'])}, {num(result['elements_per_step'])} per step."
        )
    ]
    if biggest:
        bullets.append(
            f"The tree is dominated by {tag(biggest['tag'])}: {pct(biggest['share'])} of all "
            f"elements, present in {pct(biggest['step_share'])} of steps."
        )
    if pooled:
        bullets.append(
            f"Inside the question-capable pool the leader is {tag(pooled['tag'])} with "
            f"{pct(pooled['share_of_questionable'])}; the pool behaves like "
            f"{conc['effective']:.1f} equally likely types (HHI {conc['hhi']:.3f}), and "
            f"{conc['tags_for_50']} types cover half of it."
        )
    if most and least:
        bullets.append(
            f"Recorded targets are not tree-typical: {tag(most['tag'])} is "
            f"{ratio(most['lift'] or 0)}$\\times$ over-represented among targets and "
            f"{tag(least['tag'])} {ratio(least['lift'] or 0)}$\\times$."
        )
    bullets.append(
        f"Label availability caps \\texttt{{target\\_label}}: {num(labels)} of "
        f"{num(types)} question-capable elements carry text or a label attribute "
        f"({pct(labels / types if types else 0.0)})."
    )
    bullets.append(
        f"\\texttt{{target\\_region}} cannot be synthesised from the tree (no coordinates): "
        f"only the {num(result['targets']['n'])} recorded targets know where they are, and "
        f"{pct(result['targets']['bbox_rate'])} of steps carry a bounding box."
    )
    return "\n".join([r"\section{Summary}", r"\begin{itemize}"] + [
        r"  \item " + bullet for bullet in bullets
    ] + [r"\end{itemize}"])


def _corpus_section(result: dict[str, Any]) -> str:
    chars = result["html_chars"]
    steps_with_html = result["steps_with_html"]
    repeated = (result["html_repeated_steps"] / steps_with_html) if steps_with_html else 0.0
    rows = [
        ["Tasks", num(result["tasks"])],
        ["Steps (actions)", num(result["steps"])],
        ["Steps with HTML", num(result["steps_with_html"])],
        ["Tasks without HTML", num(result["tasks_without_html"])],
        ["Elements (tree)", num(result["elements"])],
        ["Elements per step", num(result["elements_per_step"])],
        ["Steps per task (mean)", num(result["mean_steps_per_task"])],
        ["Task words (mean)", num(result["mean_task_words"])],
        [
            "HTML chars (mean / median / p95 / max)",
            f"{num(chars['mean'])} / {num(chars['median'])} / "
            f"{num(chars['p95'])} / {num(chars['max'])}",
        ],
        [
            "Repeated page states",
            f"{num(result['html_repeated_steps'])} "
            f"({pct(repeated)})",
        ],
        [
            "Candidates per step (pos / neg)",
            f"{result['candidates']['mean_pos']} / {result['candidates']['mean_neg']}",
        ],
    ]
    return "\n".join(
        [
            r"\section{Corpus shape}",
            _table(["Metric", "Value"], rows, align="lr", caption=(
                "Size and shape of the corpus under study."
            ), label="tab:corpus"),
        ]
    )


def _types_section(result: dict[str, Any], top: int = 20) -> str:
    rows = [
        [tag(row["tag"]), num(row["count"]), pct(row["share"]), num(row["steps"]),
         pct(row["step_share"])]
        for row in result["tags"][:top]
    ]
    note = ""
    if len(result["tags"]) > top:
        rest = result["tags"][top:]
        note = (
            f"\n\nThe remaining {num(len(rest))} types hold "
            f"{pct(sum(row['share'] for row in rest))} of the elements."
        )
    return "\n".join(
        [
            r"\section{Element types}",
            _table(
                ["Tag", "Count", "Share", "Steps with it", "Step share"],
                rows,
                align="lrrrr",
                caption=(
                    f"The {min(top, len(result['tags']))} most frequent element types, counted "
                    "over every parsed tree."
                ),
                label="tab:types",
            ),
            _tree_figure(result),
            note,
        ]
    )


def _pool_section(result: dict[str, Any], top: int = 14) -> str:
    rows = result["questionable"]
    if not rows:
        return r"\section{Question-capable pool}" + "\n\nNo question-capable elements found."

    conc = concentration(row["share_of_questionable"] for row in rows)
    table_rows = [
        [
            tag(row["tag"]),
            tex(row["role"]),
            num(row["count"]),
            pct(row["share_of_questionable"]),
            num(row["labelled"]),
            pct(row["label_rate"]),
        ]
        for row in rows[:top]
    ]
    return "\n".join(
        [
            r"\section{The question-capable pool}",
            (
                f"{num(len(rows))} element types can carry a question "
                f"({num(sum(row['count'] for row in rows))} elements). "
                f"Concentration: HHI {conc['hhi']:.3f}, i.e. "
                f"{conc['effective']:.1f} effective types; "
                f"{conc['tags_for_50']} types cover 50\\,\\% of the pool and "
                f"{conc['tags_for_80']} cover 80\\,\\%."
            ),
            _table(
                ["Tag", "Role", "In tree", "Pool share", "Has label", "Label rate"],
                table_rows,
                align="llrrrr",
                caption=(
                    "The question-capable types, ordered by how much of the pool they own. "
                    "\\emph{Has label} is the hard upper bound for \\texttt{target\\_label} "
                    "questions of that type."
                ),
                label="tab:pool",
            ),
            _pool_figure(result),
            _label_figure(result),
        ]
    )


def _lift_section(result: dict[str, Any], top: int = 12) -> str:
    rows = result["targets"]["tags"]
    if not rows:
        return r"\section{Tree versus recorded targets}" + "\n\nNo recorded targets found."
    table_rows = [
        [
            tag(row["tag"]),
            num(row["count"]),
            pct(row["share"]),
            pct(row["tree_share"]),
            "--" if row["lift"] is None else f"{ratio(row['lift'])}$\\times$",
        ]
        for row in rows[:top]
    ]
    return "\n".join(
        [
            r"\section{Tree versus recorded targets}",
            (
                "A type that is rare in the tree but common among the recorded targets is one "
                "the agent is forced to use; a type that fills the page but is rarely clicked "
                "is noise for question generation. The ratio of the two shares is the lift."
            ),
            _table(
                ["Tag", "Targets", "Target share", "Tree share", "Lift"],
                table_rows,
                align="lrrrr",
                caption=(
                    "Recorded action targets next to their frequency in the tree. "
                    "Lift $>1$ means over-used relative to availability."
                ),
                label="tab:lift",
            ),
            _lift_figure(result),
        ]
    )


def _potential_section(result: dict[str, Any]) -> str:
    potential = result["potential"]
    rows = [
        [
            r"\texttt{target\_label}",
            num(potential["target_label"]),
            "elements with text or a label attribute",
        ],
        [
            r"\texttt{element\_type}",
            num(potential["element_type"]),
            "elements whose tag maps to a human word",
        ],
        [
            r"\texttt{target\_region}",
            num(potential["target_region"]),
            "recorded targets with a bounding box",
        ],
    ]
    return "\n".join(
        [
            r"\section{Question potential per category}",
            _table(
                ["Category", "Questions available", "Why"],
                rows,
                align="lrl",
                caption="Upper bound of questions of each kind, given this corpus.",
                label="tab:potential",
            ),
            _potential_figure(result),
        ]
    )


def _split_section(splits: dict[str, dict[str, Any]], merged: dict[str, Any]) -> str:
    if len(splits) < 2:
        return ""
    rows = []
    for name, result in splits.items():
        pool = result["questionable"]
        top = pool[0] if pool else None
        rows.append(
            [
                tag(name),
                num(result["tasks"]),
                num(result["steps"]),
                num(result["elements_per_step"]),
                "--" if top is None else tag(top["tag"]),
                "--" if top is None else pct(top["share_of_questionable"]),
                pct(result["targets"]["label_rate"]),
                pct(result["targets"]["bbox_rate"]),
            ]
        )
    rows.append(
        [
            r"\textbf{all}",
            num(merged["tasks"]),
            num(merged["steps"]),
            num(merged["elements_per_step"]),
            "--",
            "--",
            pct(merged["targets"]["label_rate"]),
            pct(merged["targets"]["bbox_rate"]),
        ]
    )
    return "\n".join(
        [
            r"\section{Drift between splits}",
            (
                "The same sampler is applied to every test set, so a type that dominates one "
                "split but not another is a bias that only shows up at test time."
            ),
            _table(
                [
                    "Split",
                    "Tasks",
                    "Steps",
                    "Elem/step",
                    "Top type",
                    "Pool share",
                    "Has label",
                    "Has bbox",
                ],
                rows,
                align="lrrrlrrr",
                caption="Per-split shape. Rates are shares of recorded steps.",
                label="tab:splits",
            ),
            _split_figure(splits),
        ]
    )


def _policy_section(result: dict[str, Any], budget: int, top: int = 14) -> str:
    policy = policy_rows(result, budget)
    rows = [
        [
            tag(row["tag"]),
            num(row["count"]),
            pct(row["share"]),
            num(row["available"]),
            whole(row["uncontrolled"]),
            whole(row["cap"]),
            f"{ratio(row['factor'])}$\\times$",
        ]
        for row in policy["rows"][:top]
    ]
    return "\n".join(
        [
            r"\section{A sampling budget that removes the bias}",
            (
                f"Sampling questions in proportion to the tree gives each type "
                f"$\\text{{share}} \\times N$ questions for a budget of $N={num(budget)}$, so "
                f"the leader keeps its advantage. Capping each type at "
                f"$\\min(\\text{{labelled}}, N / n)$ with $n={len(policy['rows'])}$ types "
                f"removes it: the cap is {whole(policy['even'])} questions per type."
            ),
            _table(
                ["Tag", "In tree", "Pool share", "Labelled", "Uncontrolled", "Capped",
                 "Correction"],
                rows,
                align="lrrrrrr",
                caption=(
                    f"What each type would receive from a budget of {num(budget)} questions, "
                    "sampled in proportion to the pool (\\emph{Uncontrolled}) versus capped at "
                    "an equal share per type (\\emph{Capped})."
                ),
                label="tab:policy",
            ),
            (
                f"{whole(policy['shortfall'])} of the {num(budget)} questions "
                f"({pct(policy['shortfall_share'])}) cannot be filled at the equal cap because "
                "those types simply do not have that many labelled elements. Two options: "
                "lower the budget so the cap is reachable, or reallocate the surplus to the "
                "types that still have material. Reporting the shortfall is the point --- a "
                "sampler that silently redistributes it reintroduces the bias it was meant to "
                "remove."
            ),
        ]
    )


def _method_section(result: dict[str, Any]) -> str:
    return "\n".join(
        [
            r"\section{Method}",
            (
                f"Counted with the Python standard-library \\texttt{{html.parser}} over "
                f"\\texttt{{actions[*].{tex(result['html_field'])}}}, skipping "
                r"\texttt{script}, \texttt{style}, \texttt{noscript}, \texttt{template}, "
                r"\texttt{svg} and \texttt{head}. An element counts as \emph{labelled} when it "
                r"has non-empty text content or one of \texttt{aria-label}, \texttt{title}, "
                r"\texttt{alt}, \texttt{placeholder}, \texttt{data-placeholder} (plus "
                r"\texttt{value} for input, button and option). Targets are the "
                r"\texttt{is\_original\_target} candidate of each action. No page was rendered: "
                r"the numbers describe the DOM the model would read, not the pixels."
            ),
        ]
    )


def _reproduce_section(root: str | Path) -> str:
    # `verbatim` is literal: escaping here would print the backslashes.
    commands = [
        "uv run m2w-download",
        "uv run m2w-eda --all-splits --limit 0",
        f"uv run m2w-eda-report --budget 20000   # reads {root}",
    ]
    return "\n".join(
        [
            r"\section{Reproducing this report}",
            r"\begin{verbatim}",
            *commands,
            r"\end{verbatim}",
            (
                "Every figure and table above is derived from the JSON written by "
                "\\texttt{m2w-eda}; regenerating the report never touches the dataset."
            ),
        ]
    )


def render_tex(
    subject: dict[str, Any],
    *,
    splits: dict[str, dict[str, Any]] | None = None,
    merged: dict[str, Any] | None = None,
    budget: int = 20_000,
    root: str | Path = OUTPUT_DIR / "eda",
    title: str | None = None,
) -> str:
    """Render the whole report as one LaTeX document."""
    splits = splits or {}
    corpus = merged or subject
    heading = title or f"Mind2Web: what the HTML tree contains ({corpus['split']})"

    parts = [
        _preamble(heading),
        _findings(corpus),
        _corpus_section(corpus),
        _types_section(corpus),
        _pool_section(corpus),
        _lift_section(corpus),
        _potential_section(corpus),
        _split_section(splits, corpus),
        _policy_section(corpus, budget),
        _method_section(corpus),
        _reproduce_section(root),
        r"\end{document}",
    ]
    return "\n\n".join(part for part in parts if part) + "\n"


def write_report(document: str, out_dir: str | Path) -> Path:
    """Write ``report.tex`` and return its path."""
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "report.tex"
    path.write_text(document, encoding="utf-8")
    return path


def compile_pdf(tex_path: str | Path, *, timeout: int = 300) -> Path | None:
    """Run ``latexmk`` (or ``pdflatex``) next to the ``.tex``; return the PDF if it appears.

    Stale ``.aux``/``.log`` files are removed first: a half-written ``.aux`` from an earlier
    interrupted run makes TeX fail with a misleading ``runaway argument`` error. Output goes to
    a file rather than a pipe, and ``stdin`` is closed, so a prompt-happy engine cannot block.
    """
    tex_path = Path(tex_path)
    engine = shutil.which("latexmk")
    if engine:
        command = [engine, "-pdf", "-interaction=nonstopmode", "-halt-on-error", tex_path.name]
    else:
        engine = shutil.which("pdflatex")
        if not engine:
            print("[eda-report] no latexmk/pdflatex on PATH; leaving only the .tex file")
            return None
        command = [engine, "-interaction=nonstopmode", "-halt-on-error", tex_path.name]

    for stale in ("aux", "log", "out", "pdf"):
        tex_path.with_suffix(f".{stale}").unlink(missing_ok=True)

    print(f"[eda-report] compiling with {Path(engine).name} ...")
    transcript = tex_path.with_suffix(".compile.txt")
    try:
        with open(transcript, "w", encoding="utf-8", errors="replace") as handle:
            result = subprocess.run(
                command,
                cwd=tex_path.parent,
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=False,
            )
    except subprocess.TimeoutExpired:
        print(f"[eda-report] compile timed out after {timeout}s (see {transcript.name})")
        return None

    pdf = tex_path.with_suffix(".pdf")
    if result.returncode != 0 or not pdf.exists():
        transcript_text = transcript.read_text(encoding="utf-8", errors="replace")
        tail = "\n".join(transcript_text.splitlines()[-15:])
        print(f"[eda-report] compile failed:\n{tail}")
        return None

    # Keep only the deliverables: the transcript stays, the run intermediates do not.
    for intermediate in ("aux", "log", "out", "fdb_latexmk", "fls"):
        tex_path.with_suffix(f".{intermediate}").unlink(missing_ok=True)
    return pdf


def main(argv: list[str] | None = None) -> int:
    ensure_utf8()
    parser = argparse.ArgumentParser(
        description="Turn the EDA JSON into a LaTeX report with pgfplots charts."
    )
    parser.add_argument(
        "--root", type=Path, default=OUTPUT_DIR / "eda", help="where m2w-eda wrote its output"
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="default: <root>/report"
    )
    parser.add_argument(
        "--split", default=None, help="report one split instead of the merged corpus"
    )
    parser.add_argument(
        "--budget", type=int, default=20_000, help="questions to spread in the policy table"
    )
    parser.add_argument("--title", default=None, help="document title")
    parser.add_argument("--compile", action="store_true", help="run latexmk/pdflatex afterwards")
    args = parser.parse_args(argv)

    try:
        splits, merged = find_reports(args.root)
    except FileNotFoundError as error:
        print(f"[eda-report] {error}")
        return 1

    if args.split:
        if args.split not in splits:
            available = ", ".join(splits) or "none"
            print(f"[eda-report] {args.split!r} not found; available: {available}")
            return 1
        subject = splits[args.split]
        merged = None
    else:
        subject = merged or next(iter(splits.values()), None)
        if subject is None:
            print(f"[eda-report] no report under {args.root}")
            return 1

    document = render_tex(
        subject,
        splits=splits,
        merged=merged,
        budget=args.budget,
        root=args.root,
        title=args.title,
    )
    out_dir = args.out or Path(args.root) / "report"
    tex_path = write_report(document, out_dir)
    print(f"[eda-report] {tex_path}")

    if args.compile:
        pdf = compile_pdf(tex_path)
        if pdf:
            print(f"[eda-report] {pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
