"""A LaTeX report for the EDA: tables, findings and PNG charts.

Reads what ``m2w-eda`` already wrote (``outputs/eda/<split>/eda.json`` plus the merged
``outputs/eda/all/eda.json``), derives the numbers a sampling policy needs, and emits:

- ``report.tex`` — the document (tables + ``\\includegraphics``);
- ``figures/fig_*.png`` — the charts, drawn with matplotlib (``uv sync --extra report``).

The PNGs are written before the document, so the charts can be opened without a TeX
distribution; ``--compile`` additionally runs ``latexmk``/``pdflatex`` when one exists.

Usage::

    uv run m2w-eda --all-splits --limit 0        # first: produce the JSON
    uv run m2w-eda-report --compile              # -> outputs/eda/report/{report.tex,pdf,figures}

    uv run m2w-eda-report --budget 50000         # size the sampling table differently
    uv run m2w-eda-report --split test_task      # one split only
    uv run m2w-eda-report --no-figures           # tables only, no matplotlib needed
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from collections.abc import Iterable, Mapping, Sequence
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


def _figure(path: str | None, *, caption: str, label: str) -> str:
    """A float around a PNG; returns ``""`` when the chart was not produced."""
    if not path:
        return ""
    return "\n".join(
        [
            r"\begin{figure}[H]",
            r"\centering",
            rf"\includegraphics[width=\linewidth]{{{path}}}",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            r"\end{figure}",
        ]
    )


#: Colours shared by every chart: one accent for the data, one grey for reference lines.
ACCENT = "#1F4E79"
MUTED = "#6B6B6B"

#: The three question categories, as the EDA JSON keys them and as a chart should label them.
CATEGORY_KEYS = ("target_label", "element_type", "target_region")
CATEGORY_LABELS = ("label", "type", "region")

#: How many scatter points get a text label, and where those labels go.
_LABELLED_POINTS = 7
_LABEL_OFFSETS = ((5, 4), (5, -12), (-6, 5), (5, 12), (-6, -12), (5, -5), (-6, 12))


def _horizontal_bars(
    ax, names: Sequence[str], values: Sequence[float], labels: Sequence[str], *, log: bool
) -> None:
    """One bar per category: the shape every counted chart in the report shares."""
    bars = ax.barh(names, values, color=ACCENT, height=0.72)
    if log:
        ax.set_xscale("log")
    ax.bar_label(bars, labels=list(labels), padding=2, fontsize=7)
    ax.margins(x=0.22)
    ax.grid(axis="y", visible=False)


def _annotate(ax, points: Sequence[tuple[float, float, str]]) -> None:
    """Label points, cycling the offset so neighbouring labels do not overprint."""
    for index, (x, y, text) in enumerate(points):
        ax.annotate(
            text,
            (x, y),
            textcoords="offset points",
            xytext=_LABEL_OFFSETS[index % len(_LABEL_OFFSETS)],
            fontsize=7,
            color=MUTED,
            zorder=4,
        )


def _pyplot():
    """Import matplotlib lazily, so the CLI still runs without the optional extra."""
    try:
        import matplotlib
    except ModuleNotFoundError as error:  # pragma: no cover - depends on the install
        raise RuntimeError(
            "matplotlib is required to draw the report figures: run "
            "`uv sync --extra report`, or pass --no-figures for tables only."
        ) from error

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 200,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "font.size": 9,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.frameon": False,
            "legend.fontsize": 8,
        }
    )
    return plt


def _chart_tree(plt, result: dict[str, Any], top: int = 12):
    """Horizontal bars on a log x axis: the tree is heavy-tailed, so linear hides the tail."""
    rows = result["tags"][:top]
    if not rows:
        return None
    names = [row["tag"] for row in rows][::-1]
    counts = [row["count"] for row in rows][::-1]

    figure, ax = plt.subplots(figsize=(6.6, 3.3))
    _horizontal_bars(ax, names, counts, [f"{count:,}" for count in counts], log=True)
    ax.set_xlabel("elements in the tree (log scale)")
    return figure


def _chart_pool(plt, result: dict[str, Any], top: int = 10):
    rows = result["questionable"][:top]
    if not rows:
        return None
    names = [row["tag"] for row in rows][::-1]
    shares = [row["share_of_questionable"] * 100 for row in rows][::-1]

    figure, ax = plt.subplots(figsize=(6.6, 2.9))
    _horizontal_bars(ax, names, shares, [f"{share:.1f}" for share in shares], log=False)
    ax.set_xlabel("share of the question-capable pool (%)")
    return figure


def _chart_label(plt, result: dict[str, Any]):
    """Pool share against label availability: cheap questions sit high, biased types right."""
    points = [
        (row["share_of_questionable"] * 100, row["label_rate"] * 100, row["tag"])
        for row in result["questionable"]
        if row["count"] > 0
    ]
    if not points:
        return None

    figure, ax = plt.subplots(figsize=(6.6, 3.2))
    ax.scatter([x for x, _, _ in points], [y for _, y, _ in points], s=26, color=ACCENT, zorder=3)
    _annotate(ax, points[:_LABELLED_POINTS])
    ax.set_xlabel("share of the question-capable pool (%)")
    ax.set_ylabel("elements with a label (%)")
    ax.set_xlim(0, max(x for x, _, _ in points) * 1.18)
    ax.set_ylim(0, 105)
    return figure


def _chart_lift(plt, result: dict[str, Any]):
    """Log-log scatter against the no-bias diagonal (both axes share one range)."""
    rows = [row for row in result["targets"]["tags"] if row["tree_share"] > 0]
    if not rows:
        return None
    shares = [row["tree_share"] for row in rows] + [row["share"] for row in rows]
    low, high = max(min(shares) / 1.5, 1e-6), max(shares) * 1.5
    ranked = sorted(rows, key=lambda row: row["lift"] or 0, reverse=True)

    figure, ax = plt.subplots(figsize=(6.6, 3.4))
    ax.plot([low, high], [low, high], linestyle="--", color=MUTED, linewidth=0.9, label="no bias")
    ax.scatter(
        [row["tree_share"] for row in rows],
        [row["share"] for row in rows],
        s=26,
        color=ACCENT,
        zorder=3,
        label="recorded targets",
    )
    _annotate(
        ax,
        [(row["tree_share"], row["share"], row["tag"]) for row in ranked[:_LABELLED_POINTS]],
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_xlabel("share of the tree (log)")
    ax.set_ylabel("share of recorded targets (log)")
    ax.legend(loc="upper left")
    return figure


def _chart_potential(plt, result: dict[str, Any]):
    """Three counts that differ by orders of magnitude: a log axis is the only honest one."""
    values = [result["potential"][key] for key in CATEGORY_KEYS]
    if not any(values):
        return None

    figure, ax = plt.subplots(figsize=(6.2, 2.3))
    _horizontal_bars(
        ax, list(CATEGORY_LABELS)[::-1], values[::-1], [f"{value:,}" for value in values[::-1]],
        log=True,
    )
    ax.set_xlabel("questions the corpus can support (log scale)")
    return figure


def _per_1000_steps(result: dict[str, Any]) -> list[float]:
    """Questions per 1000 steps per category: splits differ in size, so rates compare."""
    scale = 1000 / result["steps"] if result["steps"] else 0.0
    return [result["potential"][key] * scale for key in CATEGORY_KEYS]


def _chart_splits(plt, splits: dict[str, dict[str, Any]]):
    """One panel per split (never several series in one axis: pgfplots used to loop on that)."""
    series = {name: _per_1000_steps(result) for name, result in splits.items()}
    positive = [value for values in series.values() for value in values if value > 0]
    if len(splits) < 2 or not positive:
        return None
    columns = 2
    rows = (len(splits) + 1) // 2
    figure, axes = plt.subplots(
        rows, columns, figsize=(6.6, 2.3 * rows), squeeze=False, sharey=True
    )
    # On a log axis a zero bar has nowhere to start, so it is drawn as a stub at the floor
    # and the annotation carries the real number.
    floor = min(positive) / 3

    for index, (name, values) in enumerate(series.items()):
        ax = axes[index // columns][index % columns]
        bars = ax.bar(
            CATEGORY_LABELS, [value or floor for value in values], color=ACCENT, width=0.55
        )
        ax.bar_label(bars, labels=[f"{value:,.0f}" for value in values], padding=2, fontsize=7)
        ax.set_yscale("log")
        ax.set_title(name, fontsize=8.5)
        ax.grid(axis="x", visible=False)
        if index % columns == 0:
            ax.set_ylabel("per 1000 steps (log)", fontsize=8)

    axes[0][0].set_ylim(floor / 3, max(positive) * 3)
    for index in range(len(splits), rows * columns):
        axes[index // columns][index % columns].set_visible(False)

    figure.tight_layout()
    return figure


def build_figures(
    result: dict[str, Any],
    splits: dict[str, dict[str, Any]],
    directory: str | Path,
) -> dict[str, str]:
    """Render every chart to ``<directory>/fig_<name>.png``; return ``name -> relative path``."""
    plt = _pyplot()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    charts = {
        "tree": _chart_tree(plt, result),
        "pool": _chart_pool(plt, result),
        "label": _chart_label(plt, result),
        "lift": _chart_lift(plt, result),
        "potential": _chart_potential(plt, result),
        "splits": _chart_splits(plt, splits),
    }

    figures: dict[str, str] = {}
    for name, chart in charts.items():
        if chart is None:
            continue
        path = directory / f"fig_{name}.png"
        chart.savefig(path)
        plt.close(chart)
        figures[name] = f"figures/{path.name}"
    return figures


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
            r"\usepackage{graphicx}",
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


def _types_section(
    result: dict[str, Any], figures: Mapping[str, str] | None = None, top: int = 20
) -> str:
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
            _figure(
                (figures or {}).get("tree"),
                caption=(
                    f"The {min(top, len(result['tags']))} most frequent element types in "
                    f"{tag(result['html_field'])}. The log axis keeps the tail readable: an "
                    "uncontrolled sampler would reproduce exactly this shape."
                ),
                label="fig:tree",
            ),
            note,
        ]
    )


def _pool_section(
    result: dict[str, Any], figures: Mapping[str, str] | None = None, top: int = 14
) -> str:
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
            _figure(
                (figures or {}).get("pool"),
                caption=(
                    "Composition of the question-capable pool. Every element of these types can "
                    "carry an \\texttt{element\\_type} question; the leader is the type a naive "
                    "sampler over-produces."
                ),
                label="fig:pool",
            ),
            _figure(
                (figures or {}).get("label"),
                caption=(
                    "Pool share against label availability. Points high on the y axis are cheap "
                    "\\texttt{target\\_label} questions; points on the right are the types that "
                    "would dominate without a cap."
                ),
                label="fig:label",
            ),
        ]
    )


def _lift_section(
    result: dict[str, Any], figures: Mapping[str, str] | None = None, top: int = 12
) -> str:
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
            _figure(
                (figures or {}).get("lift"),
                caption=(
                    "Recorded targets against the tree. Above the dashed line the agent uses a "
                    "type more often than the tree suggests (the task forces it); below the line "
                    "the tree is full of it but it rarely matters."
                ),
                label="fig:lift",
            ),
        ]
    )


def _potential_section(
    result: dict[str, Any], figures: Mapping[str, str] | None = None
) -> str:
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
            _figure(
                (figures or {}).get("potential"),
                caption=(
                    "Questions the tree can support, by category, on a log axis because the "
                    "three differ by orders of magnitude. \\texttt{region} is bounded by "
                    "the recorded targets (the tree carries no coordinates); the other two by the "
                    "elements themselves."
                ),
                label="fig:potential",
            ),
        ]
    )


def _split_section(
    splits: dict[str, dict[str, Any]],
    merged: dict[str, Any],
    figures: Mapping[str, str] | None = None,
) -> str:
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
            _figure(
                (figures or {}).get("splits"),
                caption=(
                    "Question yield per 1000 steps, split by split. Normalising by steps makes "
                    "splits of different sizes comparable: an uneven panel means the same "
                    "sampler would behave differently on each test set."
                ),
                label="fig:splits",
            ),
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
    figures: Mapping[str, str] | None = None,
) -> str:
    """Render the whole report as one LaTeX document.

    ``figures`` maps a chart name (``tree``, ``pool``, ``label``, ``lift``, ``potential``,
    ``splits``) to the image path *relative to the report directory*; a missing name simply
    leaves that chart out.
    """
    splits = splits or {}
    corpus = merged or subject
    heading = title or f"Mind2Web: what the HTML tree contains ({corpus['split']})"

    parts = [
        _preamble(heading),
        _findings(corpus),
        _corpus_section(corpus),
        _types_section(corpus, figures),
        _pool_section(corpus, figures),
        _lift_section(corpus, figures),
        _potential_section(corpus, figures),
        _split_section(splits, corpus, figures),
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
        description="Turn the EDA JSON into a LaTeX report with PNG charts."
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
    parser.add_argument(
        "--no-figures", action="store_true", help="tables only: skip matplotlib and the charts"
    )
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

    out_dir = args.out or Path(args.root) / "report"
    figures: dict[str, str] = {}
    if not args.no_figures:
        try:
            figures = build_figures(subject, splits, out_dir / "figures")
        except RuntimeError as error:
            print(f"[eda-report] {error}")
            return 1
        for path in figures.values():
            print(f"[eda-report] {path}")

    document = render_tex(
        subject,
        splits=splits,
        merged=merged,
        budget=args.budget,
        root=args.root,
        title=args.title,
        figures=figures,
    )
    tex_path = write_report(document, out_dir)
    print(f"[eda-report] {tex_path}")

    if args.no_figures:
        print("[eda-report] --no-figures: the report has tables and text, no charts")
    if args.compile:
        pdf = compile_pdf(tex_path)
        if pdf:
            print(f"[eda-report] {pdf}")
    elif not args.no_figures:
        print(
            "[eda-report] charts are PNG files in figures/ (open them directly); "
            "add --compile, or upload the folder, to get the PDF"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
