"""Tests for the ms-swift export + the training command builders (no GPU, no ms-swift)."""

import json
from pathlib import Path
from typing import Any

from mind2web_vqg.config import Config
from mind2web_vqg.data.swift import (
    IMAGE_PLACEHOLDER,
    dataset_stats,
    filter_missing_images,
    group_key,
    prepare_split,
    resolve_image_path,
    sample_per_category,
    split_examples,
    to_swift_example,
    write_jsonl,
)
from mind2web_vqg.train import (
    build_export_command,
    build_sft_command,
    find_last_checkpoint,
    format_command,
)


def vqg_example(**over: Any) -> dict[str, Any]:
    """A ``build_examples``-shaped row with sane defaults."""
    base: dict[str, Any] = {
        "id": "task1:0::target_label",
        "input": "You are shown a screenshot. Ask exactly one question...",
        "target": "What text is shown on the button the user needs?",
        "answer": "Search",
        "image": "images/task1_00.jpg",
        "meta": {
            "category": "target_label",
            "annotation_id": "task1",
            "source_id": "task1:0",
            "answer_source": "screenshot",
        },
    }
    base.update(over)
    return base


def write_vqg(rows: list[dict[str, Any]], path: Path) -> Path:
    return write_jsonl(rows, path)


# --------------------------------------------------------------------------- #
# Conversion
# --------------------------------------------------------------------------- #
def test_to_swift_example_shape(tmp_path: Path) -> None:
    image = tmp_path / "images" / "task1_00.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"jpeg")

    row = to_swift_example(vqg_example(image=str(image)))

    assert [m["role"] for m in row["messages"]] == ["user", "assistant"]
    assert row["messages"][0]["content"].startswith(IMAGE_PLACEHOLDER)
    assert row["messages"][0]["content"].endswith("Ask exactly one question...")
    assert row["messages"][1]["content"].startswith("What text is shown")
    assert row["images"] == [str(image.resolve())]


def test_with_answer_adds_a_second_turn() -> None:
    row = to_swift_example(vqg_example(), with_answer=True)

    assert [m["role"] for m in row["messages"]] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert row["messages"][-1]["content"] == "Search"


def test_with_answer_keeps_one_turn_when_answer_missing() -> None:
    row = to_swift_example(vqg_example(answer=""), with_answer=True)
    assert len(row["messages"]) == 2


def test_image_path_is_resolved_against_the_jsonl(tmp_path: Path) -> None:
    examples = tmp_path / "vqg" / "examples.jsonl"
    resolved = resolve_image_path("images/a.jpg", examples_path=examples)
    assert resolved == (tmp_path / "vqg" / "images" / "a.jpg").resolve()


def test_image_root_remaps_by_basename(tmp_path: Path) -> None:
    resolved = resolve_image_path(
        "/old/machine/processed/images/a.jpg", image_root=tmp_path / "screenshots"
    )
    assert resolved == (tmp_path / "screenshots" / "a.jpg").resolve()


def test_filter_missing_images_keeps_only_existing(tmp_path: Path) -> None:
    present = tmp_path / "a.jpg"
    present.write_bytes(b"jpeg")

    kept, missing = filter_missing_images(
        [vqg_example(image=str(present)), vqg_example(image=str(tmp_path / "gone.jpg"))]
    )

    assert len(kept) == 1
    assert missing == [str(tmp_path / "gone.jpg")]


# --------------------------------------------------------------------------- #
# Splitting / sampling
# --------------------------------------------------------------------------- #
def test_group_key_prefers_the_task() -> None:
    assert group_key(vqg_example()) == "task1"
    assert group_key(vqg_example(meta={"source_id": "task9:3", "category": "element_type"})) == (
        "task9:3"
    )
    assert group_key(vqg_example(meta={}, id="row-7")) == "row-7"


def test_split_is_grouped_by_task() -> None:
    rows = [
        vqg_example(id=f"task{task}:{step}::target_label", meta={"annotation_id": f"task{task}"})
        for task in range(6)
        for step in range(2)
    ]

    train, val = split_examples(rows, val_fraction=0.5, seed=0)

    train_tasks = {group_key(ex) for ex in train}
    val_tasks = {group_key(ex) for ex in val}
    assert train_tasks and val_tasks
    assert train_tasks.isdisjoint(val_tasks)
    assert len(train) + len(val) == len(rows)


def test_split_is_deterministic() -> None:
    rows = [vqg_example(id=f"task{i}", meta={"annotation_id": f"task{i}"}) for i in range(10)]
    first = split_examples(rows, val_fraction=0.3, seed=7)
    second = split_examples(rows, val_fraction=0.3, seed=7)
    assert [ex["id"] for ex in first[1]] == [ex["id"] for ex in second[1]]


def test_single_group_has_no_validation_split() -> None:
    rows = [vqg_example(id=f"task1:{step}") for step in range(3)]
    train, val = split_examples(rows, val_fraction=0.5)
    assert len(train) == 3
    assert val == []


def test_val_fraction_zero_keeps_everything_in_train() -> None:
    rows = [vqg_example(id=f"task{i}", meta={"annotation_id": f"task{i}"}) for i in range(4)]
    train, val = split_examples(rows, val_fraction=0.0)
    assert len(train) == 4
    assert val == []


def test_sample_per_category_caps_and_is_deterministic() -> None:
    rows = [
        vqg_example(id=f"task{i}", meta={"annotation_id": f"task{i}", "category": category})
        for category in ("target_label", "element_type")
        for i in range(10)
    ]

    first = sample_per_category(rows, 3, seed=1)
    second = sample_per_category(rows, 3, seed=1)

    assert len(first) == 6
    assert [ex["id"] for ex in first] == [ex["id"] for ex in second]
    assert {ex["meta"]["category"] for ex in first} == {"target_label", "element_type"}


def test_sample_per_category_none_keeps_everything() -> None:
    rows = [vqg_example(id=f"t{i}") for i in range(5)]
    assert len(sample_per_category(rows, None)) == 5


# --------------------------------------------------------------------------- #
# prepare_split (I/O)
# --------------------------------------------------------------------------- #
def test_prepare_split_writes_swift_files(tmp_path: Path) -> None:
    image = tmp_path / "images" / "a.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"jpeg")

    rows = [
        vqg_example(
            id=f"task{task}:{step}",
            image=str(image),
            meta={
                "annotation_id": f"task{task}",
                "category": "target_label" if step == 0 else "element_type",
            },
        )
        for task in range(4)
        for step in range(2)
    ]
    examples = write_vqg(rows, tmp_path / "vqg" / "examples.jsonl")

    result = prepare_split(
        "train", out_dir=tmp_path / "swift", examples_path=examples, val_fraction=0.5, seed=3
    )

    train_rows = [json.loads(line) for line in result["train"].read_text("utf-8").splitlines()]
    val_rows = [json.loads(line) for line in result["val"].read_text("utf-8").splitlines()]
    assert train_rows and val_rows
    assert len(train_rows) + len(val_rows) == len(rows)
    assert all(row["messages"][0]["content"].startswith(IMAGE_PLACEHOLDER) for row in train_rows)
    assert all(Path(row["images"][0]).is_absolute() for row in val_rows)

    stats = json.loads(result["stats_path"].read_text("utf-8"))
    assert stats["val"]["examples"] == len(val_rows)
    assert stats["train"]["tasks"] + stats["val"]["tasks"] == 4
    assert stats["missing_images"] == 0


def test_prepare_split_drops_and_counts_missing_images(tmp_path: Path) -> None:
    present = tmp_path / "a.jpg"
    present.write_bytes(b"jpeg")

    rows = [
        vqg_example(id="task1:0", image=str(present), meta={"annotation_id": "task1"}),
        vqg_example(
            id="task2:0", image=str(tmp_path / "gone.jpg"), meta={"annotation_id": "task2"}
        ),
    ]
    examples = write_vqg(rows, tmp_path / "vqg" / "examples.jsonl")

    result = prepare_split("train", out_dir=tmp_path / "swift", examples_path=examples)

    assert result["stats"]["missing_images"] == 1
    assert result["stats"]["train"]["examples"] + result["stats"]["val"]["examples"] == 1
    assert result["val"] is None  # only one group survives -> all train


def test_dataset_stats_reports_categories() -> None:
    train = [vqg_example(), vqg_example(meta={"category": "element_type", "annotation_id": "t"})]
    stats = dataset_stats(train, [], missing=["x.jpg"])
    assert stats["train"]["examples"] == 2
    assert stats["train"]["categories"]["target_label"] == 1
    assert stats["train"]["categories"]["target_region"] == 0
    assert stats["missing_images"] == 1


# --------------------------------------------------------------------------- #
# swift commands
# --------------------------------------------------------------------------- #
def config(**train: Any) -> Config:
    return Config(raw={"train": train, "merge": {}})


def test_build_sft_command_lora_defaults() -> None:
    cmd = build_sft_command(
        config(),
        model="Qwen/Qwen3.5-9B",
        dataset=Path("train.jsonl"),
        val_dataset=Path("val.jsonl"),
        output_dir=Path("outputs/train/run"),
    )

    assert cmd[:2] == ["swift", "sft"]
    assert cmd[cmd.index("--tuner_type") + 1] == "lora"
    assert "--lora_rank" in cmd
    assert "--quant_method" not in cmd
    assert cmd[cmd.index("--val_dataset") + 1] == "val.jsonl"
    assert "--gradient_checkpointing" in cmd
    assert "--use_hf" in cmd


def test_build_sft_command_qlora_adds_quantization() -> None:
    cmd = build_sft_command(
        config(method="qlora", batch_size=2, lr=2e-5),
        model="m",
        dataset=Path("t.jsonl"),
        val_dataset=None,
        output_dir=Path("o"),
    )

    assert cmd[cmd.index("--quant_method") + 1] == "bnb"
    assert cmd[cmd.index("--quant_bits") + 1] == "4"
    assert cmd[cmd.index("--learning_rate") + 1] == "2e-05"
    assert cmd[cmd.index("--per_device_train_batch_size") + 1] == "2"
    assert "--eval_steps" not in cmd  # no val set -> no eval schedule


def test_build_sft_command_full_skips_lora_flags() -> None:
    cmd = build_sft_command(
        config(method="full"),
        model="m",
        dataset=Path("t.jsonl"),
        val_dataset=None,
        output_dir=Path("o"),
    )
    assert cmd[cmd.index("--tuner_type") + 1] == "full"
    assert "--lora_rank" not in cmd
    assert "--target_modules" not in cmd


def test_build_sft_command_rejects_unknown_method() -> None:
    try:
        build_sft_command(
            config(method="dpo"),
            model="m",
            dataset=Path("t.jsonl"),
            val_dataset=None,
            output_dir=Path("o"),
        )
    except ValueError as exc:
        assert "dpo" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_build_sft_command_passes_extra_args_last() -> None:
    cmd = build_sft_command(
        config(extra_args=["--packing", "true"]),
        model="m",
        dataset=Path("t.jsonl"),
        val_dataset=None,
        output_dir=Path("o"),
    )
    assert cmd[-2:] == ["--packing", "true"]


def test_build_export_command_merges() -> None:
    cmd = build_export_command(config(), adapters=Path("ckpt"), output_dir=Path("merged"))
    assert cmd[:2] == ["swift", "export"]
    assert cmd[cmd.index("--merge_lora") + 1] == "true"
    assert cmd[cmd.index("--adapters") + 1] == "ckpt"


def test_format_command_quotes_spaces() -> None:
    assert format_command(["swift", "sft", "--model", "a b/c"]) == "swift sft --model 'a b/c'"


# --------------------------------------------------------------------------- #
# Checkpoints
# --------------------------------------------------------------------------- #
def test_find_last_checkpoint_picks_the_highest_step(tmp_path: Path) -> None:
    for step in (5, 100, 50):
        (tmp_path / "v1-run" / f"checkpoint-{step}").mkdir(parents=True)
    (tmp_path / "not-a-checkpoint").mkdir()

    found = find_last_checkpoint(tmp_path)

    assert found is not None
    assert found.name == "checkpoint-100"


def test_find_last_checkpoint_none_when_empty(tmp_path: Path) -> None:
    assert find_last_checkpoint(tmp_path) is None
