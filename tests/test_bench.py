"""Tests for TreeVQG-Bench v1 (question types + stratified sampling).

The builders run on hand-written HTML: no screenshots, no network, no dataset.
"""

from __future__ import annotations

import json
from typing import Any

from mind2web_vqg.treevqg.bench import (
    _make_row,
    q_absence,
    q_consequence,
    q_container_membership,
    q_sibling_disambiguation,
    q_target_label,
    q_target_region,
    stratified_sample,
)
from mind2web_vqg.treevqg.parse import (
    apply_candidate_metadata,
    parse_html,
    target_from_candidates,
    tree_stats,
)
from mind2web_vqg.treevqg.templates import LevelContext

TWO_CARDS = """
<html><body>
  <main>
    <div aria-label="Recommended">
      <h2>Recommended</h2>
      <button backend_node_id="1">Add</button>
    </div>
    <div aria-label="Recently viewed">
      <h2>Recently viewed</h2>
      <button backend_node_id="2">Add</button>
    </div>
  </main>
</body></html>
"""

DELETE_BUTTON = """
<html><body>
  <div aria-label="Saved items">
    <h2>Saved items</h2>
    <button backend_node_id="3">Delete</button>
  </div>
</body></html>
"""

FORM = """
<html><body>
  <form aria-label="Checkout">
    <h2>Checkout</h2>
    <label>Card number</label>
    <input backend_node_id="4" aria-label="Card number">
    <button backend_node_id="5">Submit</button>
  </form>
</body></html>
"""


def build(
    html: str, candidates: dict[str, str], *, target: str | None = None
) -> tuple[Any, dict[str, Any]]:
    """Parse ``html`` and attach bbox metadata to the given backend ids."""
    action = {
        "action_uid": "uid1",
        "cleaned_html": html,
        "pos_candidates": [
            {
                "tag": "button",
                "backend_node_id": node_id,
                "is_original_target": target is None or node_id == target,
                "attributes": json.dumps(
                    {"backend_node_id": node_id, "bounding_box_rect": rect}
                ),
            }
            for node_id, rect in candidates.items()
        ],
        "neg_candidates": [],
        "target_action_reprs": "",
    }
    tree = parse_html(html)
    apply_candidate_metadata(tree, action)
    return tree, action


def context(**over: Any) -> LevelContext:
    base = LevelContext(
        task="Add the cheapest item to the cart",
        image_size=[1280, 2000],
        op="CLICK",
    )
    for key, value in over.items():
        setattr(base, key, value)
    return base


def test_container_membership_names_the_card() -> None:
    tree, action = build(
        TWO_CARDS, {"1": "100,100,80,30", "2": "100,400,80,30"}, target="2"
    )
    target = target_from_candidates(tree, action)
    question = q_container_membership(tree, target, context(), image_size=[1280, 2000])
    assert question is not None
    assert question.answer == "Recently viewed"
    assert "section" in question.question
    assert [kind for _, kind in question.referents] == ["anchor", "container"]


def test_sibling_disambiguation_prefers_the_distinguishing_card() -> None:
    tree, action = build(
        TWO_CARDS, {"1": "100,100,80,30", "2": "100,400,80,30"}, target="2"
    )
    target = target_from_candidates(tree, action)
    question = q_sibling_disambiguation(tree, target, context(), image_size=[1280, 2000])
    assert question is not None
    assert question.extra["n_same"] == 2
    assert "Recently viewed" in question.answer


def test_sibling_disambiguation_needs_more_than_one_candidate() -> None:
    tree, action = build(DELETE_BUTTON, {"3": "100,100,80,30"})
    target = target_from_candidates(tree, action)
    assert q_sibling_disambiguation(tree, target, context()) is None


def test_absence_is_balanced_and_named() -> None:
    html = """<html><body><main>
      <div aria-label="Recommended"><h2>Recommended</h2>
        <button backend_node_id="1">Add</button></div>
      <div aria-label="Recently viewed"><h2>Recently viewed</h2>
        <button backend_node_id="2">Save</button></div>
    </main></body></html>"""
    tree, action = build(html, {"1": "100,100,80,30", "2": "100,400,80,30"})
    target = target_from_candidates(tree, action)
    question = q_absence(tree, target, context())
    assert question is not None
    assert question.answer in {"Yes", "No"}
    assert question.extra["section"]
    assert question.extra["polarity"] in {"yes", "no"}


def test_consequence_reads_the_next_state() -> None:
    tree, action = build(DELETE_BUTTON, {"3": "100,100,80,30"})
    target = target_from_candidates(tree, action)
    question = q_consequence(
        tree,
        target,
        context(op="CLICK"),
        current_html="<div>Saved items Delete</div>",
        next_html="<div>Your saved items list is empty.</div>",
        op="CLICK",
    )
    assert question is not None
    assert question.extra["consequence_class"] == "deletion"
    assert question.answer_source == "trajectory:t+1"


def test_consequence_transaction_requires_new_confirmation_text() -> None:
    tree, action = build(FORM, {"4": "100,100,200,30", "5": "100,300,100,30"}, target="5")
    target = target_from_candidates(tree, action)
    question = q_consequence(
        tree,
        target,
        context(op="CLICK"),
        current_html="<div>Checkout</div>",
        next_html="<div>Thank you for your order. Confirmation number 42.</div>",
        op="CLICK",
    )
    assert question is not None
    assert question.extra["consequence_class"] == "transaction"


def test_consequence_ignores_non_click_actions() -> None:
    tree, action = build(FORM, {"4": "100,100,200,30", "5": "100,300,100,30"}, target="5")
    target = target_from_candidates(tree, action)
    assert (
        q_consequence(
            tree,
            target,
            context(op="TYPE"),
            current_html="<div>Checkout</div>",
            next_html="<div>Thank you for your order.</div>",
            op="TYPE",
        )
        is None
    )


def test_target_label_and_region() -> None:
    tree, action = build(FORM, {"4": "100,100,200,30", "5": "100,300,100,30"}, target="5")
    target = target_from_candidates(tree, action)
    label = q_target_label(tree, target, context())
    assert label is not None and label.answer == "Submit"
    region = q_target_region(tree, target, context(), image_size=[1280, 2000])
    assert region is not None and region.answer in {"top-left", "top-center", "top-right"}


def test_make_row_exposes_two_views_and_leak_flags() -> None:
    tree, action = build(DELETE_BUTTON, {"3": "100,100,80,30"})
    target = target_from_candidates(tree, action)
    ctx = context()
    question = q_target_label(tree, target, ctx)
    source = {
        "id": "task1:0",
        "image": "images/task1_00.jpg",
        "source_size": [1280, 2000],
        "image_size": [1280, 2000],
        "task": ctx.task,
        "website": "shop",
        "op": "CLICK",
    }
    row = _make_row(
        source=source,
        instance={"annotation_id": "task1"},
        split="test_task",
        question=question,
        tree=tree,
        context=ctx,
        image_size=[1280, 2000],
        stats=tree_stats(tree),
        max_lines=40,
    )
    assert row["input"].startswith("Task:")
    assert "UI tree" in row["condition"]
    assert row["target"] == row["question"]
    assert row["meta"]["answer_in_question"] is False
    assert row["meta"]["requires_tree"] is False
    assert row["referents"][0]["kind"] == "anchor"


def test_stratified_sample_respects_website_cap_then_relaxes() -> None:
    items = [
        {"meta": {"website": "a"}, "id": f"a{i}"} for i in range(5)
    ] + [{"meta": {"website": "b"}, "id": f"b{i}"} for i in range(5)]
    chosen = stratified_sample(items, per_type=6, max_per_website=2, seed=1)
    assert len(chosen) == 6
    websites = [item["meta"]["website"] for item in chosen]
    # The first pass honours the cap; the second pass relaxes it rather than returning
    # fewer rows than asked for.
    assert websites.count("a") >= 2 and websites.count("b") >= 2
