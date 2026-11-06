"""Tests for the TreeVQG MVP: parsing, 3-level anchors, templates, builder, metrics."""

from __future__ import annotations

from typing import Any

from mind2web_vqg.metrics import compute_metrics
from mind2web_vqg.treevqg import io as tree_io
from mind2web_vqg.treevqg.build import build_rows_for_action
from mind2web_vqg.treevqg.levels import select_anchors, target_node
from mind2web_vqg.treevqg.parse import parse_bbox, parse_html, tree_stats
from mind2web_vqg.treevqg.schema import UiTree
from mind2web_vqg.treevqg.templates import LevelContext, build_input, region_label

HTML = """
<html>
  <body>
    <nav aria-label="Main navigation">
      <a href="/home">Home</a>
      <a href="/cart">Cart</a>
    </nav>
    <main>
      <form aria-label="Flight Search">
        <h2>Flight Search</h2>
        <label>Departure date</label>
        <input type="text" aria-label="Departure date" name="from">
        <input type="text" aria-label="Return date" name="to">
        <button>Search flights</button>
      </form>
    </main>
    <script>var ignored = "<button>fake</button>";</script>
  </body>
</html>
"""


def source_row(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "task1:0",
        "annotation_id": "task1",
        "action_uid": "uid1",
        "split": "test_task",
        "task": "Book a flight from Hanoi to Paris",
        "website": "airline",
        "domain": "travel",
        "action_history": ["CLICK [Flights]"],
        "target_action": "CLICK [Search flights]",
        "target_tag": "button",
        "target_text": "Search flights",
        "op": "CLICK",
        "value": "",
        "image": "images/task1_00.jpg",
        "image_size": [1280, 1000],
    }
    base.update(over)
    return base


def action(html: str = HTML) -> dict[str, Any]:
    return {"action_uid": "uid1", "cleaned_html": html}


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def test_parse_bbox_accepts_common_encodings() -> None:
    assert parse_bbox("1,2,3,4") == [1.0, 2.0, 3.0, 4.0]
    assert parse_bbox("[1, 2, 3, 4]") == [1.0, 2.0, 3.0, 4.0]
    assert parse_bbox("1 2 3 4") == [1.0, 2.0, 3.0, 4.0]
    assert parse_bbox("bad") is None
    assert parse_bbox(None) is None


def test_parse_html_builds_a_readable_tree() -> None:
    tree = parse_html(HTML)
    assert len(tree) > 8
    assert tree.nodes[0].tag == "document"
    # script content is skipped: its fake button must not exist
    assert not any(node.tag == "button" and node.label == "fake" for node in tree.nodes)
    buttons = [node for node in tree.nodes if node.tag == "button"]
    assert len(buttons) == 1
    assert buttons[0].clickable
    assert tree.label(buttons[0].id) == "Search flights"
    inputs = [node for node in tree.nodes if node.tag == "input"]
    assert {tree.label(node.id) for node in inputs} == {"Departure date", "Return date"}
    for node in tree.nodes[1:]:
        if node.parent is not None:
            assert node.depth == tree.nodes[node.parent].depth + 1


def test_aria_role_becomes_the_human_word() -> None:
    tree = parse_html('<html><body><div role="button" tabindex="0">Save</div></body></html>')
    node = next(node for node in tree.nodes if node.tag == "div")
    assert node.role == "button"
    assert node.clickable
    assert tree.label(node.id) == "Save"


def test_tree_roundtrip_preserves_structure() -> None:
    tree = parse_html(HTML)
    rebuilt = UiTree.from_dict(tree.to_dict())
    assert len(rebuilt) == len(tree)
    assert [node.tag for node in rebuilt.nodes] == [node.tag for node in tree.nodes]
    assert rebuilt.label(tree.root) == tree.label(tree.root)


def test_pruning_keeps_the_anchor_path() -> None:
    tree = parse_html(HTML)
    target = target_node(tree, tag="button", text="Search flights")
    assert target is not None
    view, remap = tree.pruned_with_map(6, anchor=target)
    new_anchor = remap[target]
    assert new_anchor < len(view)
    assert view.path(new_anchor)[0] == view.root
    assert view.node(new_anchor).tag == "button"
    assert all(node.parent is None or node.parent < node.id for node in view.nodes[1:])


def test_tree_stats_and_region() -> None:
    stats = tree_stats(parse_html(HTML))
    assert stats["nodes"] > 0 and stats["clickable"] >= 1
    assert region_label([0, 0, 10, 10], [100, 100]) == "top-left"
    assert region_label([95, 95, 5, 5], [100, 100]) == "bottom-right"
    assert region_label([0, 0, 10, 10], None) is None


# --------------------------------------------------------------------------- #
# Levels
# --------------------------------------------------------------------------- #
def test_target_and_three_anchors() -> None:
    tree = parse_html(HTML)
    target = target_node(tree, tag="button", text="Search flights")
    assert target is not None
    anchors = select_anchors(tree, target=target)
    assert set(anchors) == {"L1", "L2", "L3"}
    assert anchors["L1"] == target
    assert tree.node(anchors["L2"]).tag == "form"
    assert anchors["L3"] == tree.root


def test_anchor_selection_survives_a_short_tree() -> None:
    tree = parse_html("<html><body><button>Only</button></body></html>")
    anchors = select_anchors(tree)
    assert "L1" in anchors
    assert tree.node(anchors["L1"]).tag == "button"


# --------------------------------------------------------------------------- #
# Templates + builder
# --------------------------------------------------------------------------- #
def test_build_input_marks_the_anchor() -> None:
    tree = parse_html(HTML)
    target = target_node(tree, tag="button", text="Search flights")
    assert target is not None
    prompt = build_input(
        tree,
        target,
        "L1",
        LevelContext(task="Book a flight", action_history=["CLICK [Flights]"]),
    )
    assert "marked with *" in prompt
    assert "Book a flight" in prompt
    assert "CLICK [Flights]" in prompt
    assert "Search flights" in prompt


def test_build_rows_for_action_emits_three_levels() -> None:
    rows, skipped = build_rows_for_action(source_row(), action(), split="test_task")
    assert skipped == []
    assert [row["level"] for row in rows] == ["L1", "L2", "L3"]
    for row in rows:
        assert row["target"].endswith("?")
        assert row["answer"]
        assert row["tree"]["anchor"] < len(row["tree"]["nodes"])
        assert row["meta"]["category"] == row["level"]
        assert row["meta"]["tree_nodes"] <= row["meta"]["full_tree_nodes"]
    assert rows[0]["answer"] == "Search flights"
    assert rows[2]["answer"] == "1"  # one completed step


def test_build_rows_skips_when_html_is_missing() -> None:
    rows, skipped = build_rows_for_action(source_row(), {"action_uid": "uid1"}, split="test_task")
    assert rows == []
    assert skipped == ["no_html"]


def test_build_split_end_to_end(tmp_path: Any, monkeypatch: Any) -> None:
    import mind2web_vqg.treevqg.build as build_mod

    image = tmp_path / "shot.jpg"
    image.write_bytes(b"not-a-real-jpeg")
    instance = {
        "annotation_id": "task1",
        "confirmed_task": "Book a flight from Hanoi to Paris",
        "website": "airline",
        "domain": "travel",
        "actions": [{"action_uid": "uid1", "cleaned_html": HTML}],
    }
    monkeypatch.setattr(build_mod, "load_exported", lambda *a, **k: [source_row(image=str(image))])
    monkeypatch.setattr(build_mod, "iter_split", lambda split: iter([instance]))
    monkeypatch.setattr(build_mod, "PROCESSED_DIR", tmp_path)

    result = build_mod.build_split("test_task", limit=1, levels=("L1", "L2", "L3"))
    stats = result["stats"]
    assert result["examples"].is_file()
    assert stats["matched_actions"] == 1
    assert stats["built_examples"] == 3
    assert stats["by_level"] == {"L1": 1, "L2": 1, "L3": 1}
    rows = tree_io.read_jsonl(result["examples"])
    assert [row["level"] for row in rows] == ["L1", "L2", "L3"]


def test_metrics_group_treevqg_levels() -> None:
    records = [
        {
            "prediction": "What text is shown on the button?",
            "target": "What text is shown on the button?",
            "answer": "Search",
            "meta": {"category": "L1"},
        },
        {
            "prediction": "Which container holds it?",
            "target": "Which container holds it?",
            "answer": "form",
            "meta": {"category": "L2"},
        },
    ]
    metrics = compute_metrics(records)
    assert set(metrics["by_category"]) == {"L1", "L2"}
    assert metrics["bleu4"] == 1.0
