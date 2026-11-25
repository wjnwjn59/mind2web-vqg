"""Fine-tune the VQG model with ms-swift (LoRA / QLoRA), then merge the adapter.

The trainer is ms-swift, called as a **subprocess**. This project's environment stays
inference-only (``uv sync``), so the heavy training stack can live in its own venv or
conda env on the lab server — see ``docs/train-framework.md`` for why that split exists.

Pipeline::

    data/processed/multimodal/<split>/examples.jsonl     screenshots + action metadata
      -> data/processed/vqg/<split>/examples.jsonl       question/answer references
      -> data/processed/vqg/<split>/swift/train.jsonl    ms-swift format (see data/swift.py)
      -> swift sft                                       -> <output_dir>/vX/checkpoint-N
      -> swift export --merge_lora true                  -> <merged_dir>
      -> m2w-eval --model vlm/<merged_dir>               -> outputs/eval_<split>/metrics.json

Usage::

    uv run m2w-train --config configs/train_swift.yaml --dry-run   # print the commands
    uv run m2w-train --config configs/train_swift.yaml             # train + merge
    uv run m2w-train --config configs/train_swift.yaml --eval      # ... then evaluate

    # data only, no GPU needed (checks the export before renting one)
    uv run m2w-train --config configs/train_swift.yaml --build-only
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from mind2web_vqg.cli import ensure_utf8
from mind2web_vqg.config import Config, load_config
from mind2web_vqg.data.swift import prepare_split, print_summary
from mind2web_vqg.data.vqg import SCREENSHOT_GROUNDED
from mind2web_vqg.models.hf import resolve_model_path

_CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)$")


# --------------------------------------------------------------------------- #
# Command builders (pure -> unit-tested without ms-swift installed)
# --------------------------------------------------------------------------- #
def _num(value: Any) -> str:
    """Format a number the way CLI flags expect it (``1e-4`` stays ``0.0001``)."""
    return f"{value:g}" if isinstance(value, float) else str(value)


def _add(cmd: list[str], flag: str, value: Any) -> None:
    """Append ``flag value`` unless the value is unset."""
    if value is not None:
        cmd.extend([flag, _num(value)])


def build_sft_command(
    cfg: Config,
    *,
    model: str,
    dataset: Path,
    val_dataset: Path | None,
    output_dir: Path,
) -> list[str]:
    """Build the ``swift sft`` argv from the ``train`` group of a config."""
    method = str(cfg.get("train.method", "lora")).lower()
    if method not in {"lora", "qlora", "full"}:
        raise ValueError(f"train.method must be lora|qlora|full, got {method!r}")
    tuner = "full" if method == "full" else "lora"

    cmd: list[str] = [
        str(cfg.get("train.swift_bin", "swift")),
        "sft",
        "--model",
        model,
        "--dataset",
        str(dataset),
        "--tuner_type",
        tuner,
        "--output_dir",
        str(output_dir),
    ]

    _add(cmd, "--val_dataset", str(val_dataset) if val_dataset else None)
    _add(cmd, "--eval_steps", cfg.get("train.eval_steps") if val_dataset else None)

    _add(cmd, "--torch_dtype", cfg.get("train.dtype", "bfloat16"))
    _add(cmd, "--num_train_epochs", cfg.get("train.epochs", 2))
    _add(cmd, "--per_device_train_batch_size", cfg.get("train.batch_size", 1))
    _add(cmd, "--gradient_accumulation_steps", cfg.get("train.grad_accum", 16))
    _add(cmd, "--learning_rate", cfg.get("train.lr", 1e-4))
    _add(cmd, "--max_length", cfg.get("train.max_length", 4096))
    _add(cmd, "--logging_steps", cfg.get("train.logging_steps", 5))
    _add(cmd, "--save_steps", cfg.get("train.save_steps", 200))
    _add(cmd, "--save_total_limit", cfg.get("train.save_total_limit", 2))
    _add(cmd, "--warmup_ratio", cfg.get("train.warmup_ratio", 0.03))
    _add(cmd, "--dataloader_num_workers", cfg.get("train.num_workers", 4))
    _add(cmd, "--seed", cfg.get("train.seed", 42))
    _add(cmd, "--resume_from_checkpoint", cfg.get("train.resume_from_checkpoint"))

    # Pixel budget: the single biggest knob for full-page screenshots (see README).
    _add(cmd, "--max_pixels", cfg.get("train.max_pixels"))

    if tuner == "lora":
        _add(cmd, "--lora_rank", cfg.get("train.lora_rank", 16))
        _add(cmd, "--lora_alpha", cfg.get("train.lora_alpha", 32))
        _add(cmd, "--lora_dropout", cfg.get("train.lora_dropout", 0.05))
        _add(cmd, "--target_modules", cfg.get("train.target_modules", "all-linear"))
    if method == "qlora":
        _add(cmd, "--quant_method", cfg.get("train.quant_method", "bnb"))
        _add(cmd, "--quant_bits", cfg.get("train.quant_bits", 4))
    if cfg.get("train.gradient_checkpointing", True):
        _add(cmd, "--gradient_checkpointing", "true")
    if cfg.get("train.packing", False):
        _add(cmd, "--packing", "true")
    _add(cmd, "--deepspeed", cfg.get("train.deepspeed"))

    report_to = str(cfg.get("train.report_to", "none")).lower()
    if report_to not in {"none", ""}:
        _add(cmd, "--report_to", report_to)
    if cfg.get("train.use_hf", True):
        _add(cmd, "--use_hf", "true")

    cmd.extend(str(arg) for arg in (cfg.get("train.extra_args") or []))
    return cmd


def build_export_command(cfg: Config, *, adapters: Path, output_dir: Path) -> list[str]:
    """Build the ``swift export --merge_lora`` argv (HF-format model for ``m2w-eval``)."""
    cmd: list[str] = [
        str(cfg.get("train.swift_bin", "swift")),
        "export",
        "--adapters",
        str(adapters),
        "--output_dir",
        str(output_dir),
        "--merge_lora",
        "true",
    ]
    if cfg.get("train.use_hf", True):
        _add(cmd, "--use_hf", "true")
    cmd.extend(str(arg) for arg in (cfg.get("merge.extra_args") or []))
    return cmd


def find_last_checkpoint(root: str | Path) -> Path | None:
    """Newest ``checkpoint-<step>`` under ``root`` (ms-swift nests it in a run folder)."""
    best: tuple[int, float, Path] | None = None
    for path in Path(root).rglob("checkpoint-*"):
        if not path.is_dir():
            continue
        match = _CHECKPOINT_RE.search(path.name)
        key = (int(match.group(1)) if match else -1, path.stat().st_mtime, path)
        if best is None or key[:2] > best[:2]:
            best = key
    return best[2] if best else None


def format_command(cmd: list[str]) -> str:
    """Shell-quoted command, good enough to paste back into a Linux shell."""
    return shlex.join(str(part) for part in cmd)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _require_swift(swift_bin: str) -> None:
    """Fail early (not after a 10-minute dataset build) if ms-swift is missing."""
    looks_like_path = any(sep in swift_bin for sep in ("/", "\\"))
    found = Path(swift_bin).is_file() if looks_like_path else shutil.which(swift_bin)
    if found:
        return
    raise SystemExit(
        f"[train] ms-swift not found: {swift_bin!r}\n"
        "  Install it in its own environment (it pins its own torch/transformers):\n"
        "      uv venv .venv-swift --python 3.11\n"
        "      . .venv-swift/bin/activate          # Windows: .venv-swift\\Scripts\\activate\n"
        "      uv pip install ms-swift -U\n"
        "  Then activate that env, or pass --swift-bin .venv-swift/bin/swift\n"
        "  See docs/train-framework.md."
    )


def _data_settings(cfg: Config) -> dict[str, Any]:
    """Map the ``data`` config group onto ``prepare_split`` keyword arguments."""
    splits = cfg.get("data.splits") or []
    split = cfg.get("data.split") or (splits[0] if splits else "train")
    return {
        "split": split,
        "out_dir": cfg.get("data.swift_dir"),
        "categories": cfg.get("data.categories") or list(SCREENSHOT_GROUNDED),
        "with_task": cfg.get("data.with_task", True),
        "with_answer": cfg.get("data.with_answer", False),
        "val_fraction": cfg.get("data.val_fraction", 0.05),
        "max_per_category": cfg.get("data.max_per_category"),
        "seed": cfg.get("data.seed", 42),
        "image_root": cfg.get("data.image_root"),
        "examples_path": cfg.get("data.vqg_examples"),
        "limit": cfg.get("data.limit"),
    }


def run_train(
    config_path: str | Path | None = None,
    *,
    model: str | None = None,
    output_dir: str | Path | None = None,
    swift_bin: str | None = None,
    dry_run: bool = False,
    build_only: bool = False,
    merge_only: bool = False,
    no_merge: bool = False,
    eval_after: bool = False,
    extra_args: list[str] | None = None,
) -> int:
    cfg = load_config(config_path)
    if swift_bin:
        cfg.raw.setdefault("train", {})["swift_bin"] = swift_bin
    if extra_args:
        cfg.raw.setdefault("train", {})["extra_args"] = extra_args

    out_dir = Path(output_dir or cfg.get("train.output_dir", "outputs/train"))
    merged_dir = Path(cfg.get("merge.output_dir") or f"{out_dir}-merged")
    do_merge = bool(cfg.get("merge.enabled", True)) and not no_merge

    # ------------------------------------------------------------------ data
    dataset: Path | None = None
    val_dataset: Path | None = None
    if not merge_only:
        try:
            result = prepare_split(**_data_settings(cfg))
        except FileNotFoundError as exc:
            raise SystemExit(
                "[train] the screenshots are not on this machine yet.\n"
                f"  {exc}\n"
                "  Download/export them first (see README section 5):\n"
                "      uv run m2w-download-mm --splits train\n"
                "      uv run m2w-mm-export --split train --limit 0\n"
                "  ...or point --config at a VQG jsonl that already exists "
                "(data.vqg_examples)."
            ) from exc
        print_summary(result)
        dataset, val_dataset = result["train"], result["val"]

    if build_only:
        print("[train] --build-only: data is ready, nothing was trained.")
        return 0

    model_name = model or cfg.get("train.base_model")
    if not model_name:
        raise SystemExit("[train] train.base_model is not set in the config.")
    model_path = resolve_model_path(str(model_name))
    if not Path(model_path).exists() and "/" not in model_path:
        raise SystemExit(f"[train] cannot find the base model: {model_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[train] base_model = {model_path}")
    print(f"[train] output_dir = {out_dir}")

    # ------------------------------------------------------------- train cmd
    adapters: Path | None = None
    if not merge_only:
        if dataset is None:
            raise SystemExit("[train] internal error: no dataset was built.")
        cmd = build_sft_command(
            cfg, model=model_path, dataset=dataset, val_dataset=val_dataset, output_dir=out_dir
        )

        command_file = out_dir / "train_command.txt"
        command_file.write_text(format_command(cmd) + "\n", encoding="utf-8")
        meta_file = out_dir / "train_meta.json"
        meta_file.write_text(
            json.dumps(
                {
                    "model": model_path,
                    "dataset": str(dataset),
                    "val_dataset": str(val_dataset) if val_dataset else None,
                    "command": format_command(cmd),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        print("[train] " + format_command(cmd))
        if dry_run:
            print(f"[train] dry run: command saved to {command_file}")
            return 0

        _require_swift(str(cfg.get("train.swift_bin", "swift")))
        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            print(
                "[train] note: CUDA_VISIBLE_DEVICES is unset — on the lab server use card 1 "
                "(CUDA_VISIBLE_DEVICES=1)."
            )
        subprocess.run(cmd, check=True)

        found = find_last_checkpoint(out_dir)
        if found is None:
            raise SystemExit(f"[train] training finished but no checkpoint-* under {out_dir}")
        adapters = found
        print(f"[train] last checkpoint: {adapters}")
        print(
            "[train] smoke-test it with:\n"
            f"    swift infer --adapters {adapters} --stream true --max_new_tokens 64"
        )

    # ------------------------------------------------------------- merge
    if do_merge:
        if adapters is None:
            adapters = find_last_checkpoint(out_dir)
        if adapters is None:
            raise SystemExit(f"[train] --merge-only found no checkpoint under {out_dir}")
        merge_cmd = build_export_command(cfg, adapters=adapters, output_dir=merged_dir)
        print("[train] " + format_command(merge_cmd))
        if dry_run:
            print("[train] dry run: merge step not executed")
            return 0
        _require_swift(str(cfg.get("train.swift_bin", "swift")))
        subprocess.run(merge_cmd, check=True)
        print(f"[train] merged model -> {merged_dir}")
    elif eval_after:
        print("[train] --eval needs a merged model; --no-merge was set, so nothing to evaluate.")
        return 0

    # ------------------------------------------------------------- next step
    if do_merge:
        eval_config = str(cfg.get("run.eval_config", "configs/default.yaml"))
        print(
            "[train] evaluate it with (the split comes from data.splits in the eval config):\n"
            f"    uv run m2w-eval --model vlm/{merged_dir} --config {eval_config}"
        )
        if eval_after:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "mind2web_vqg.evaluate",
                    "--model",
                    f"vlm/{merged_dir}",
                    "--config",
                    eval_config,
                ],
                check=True,
            )
    return 0


def main(argv: list[str] | None = None) -> int:
    ensure_utf8()
    parser = argparse.ArgumentParser(
        description="Fine-tune the VQG model with ms-swift (LoRA/QLoRA) and merge the adapter."
    )
    parser.add_argument("--config", default=None, help="YAML file (default: configs/default.yaml)")
    parser.add_argument("--model", default=None, help="override train.base_model")
    parser.add_argument("--output-dir", default=None, help="override train.output_dir")
    parser.add_argument(
        "--swift-bin",
        default=None,
        help="path to the swift executable when ms-swift lives in another env",
    )
    parser.add_argument("--dry-run", action="store_true", help="build the data + print commands")
    parser.add_argument("--build-only", action="store_true", help="only prepare the JSONL files")
    parser.add_argument("--merge-only", action="store_true", help="merge the last checkpoint")
    parser.add_argument("--no-merge", action="store_true", help="keep the LoRA adapter only")
    parser.add_argument("--eval", dest="eval_after", action="store_true", help="run m2w-eval after")
    parser.add_argument(
        "--extra-args",
        nargs=argparse.REMAINDER,
        default=None,
        help="everything after this is passed to `swift sft` verbatim",
    )
    args = parser.parse_args(argv)

    return run_train(
        args.config,
        model=args.model,
        output_dir=args.output_dir,
        swift_bin=args.swift_bin,
        dry_run=args.dry_run,
        build_only=args.build_only,
        merge_only=args.merge_only,
        no_merge=args.no_merge,
        eval_after=args.eval_after,
        extra_args=args.extra_args,
    )


if __name__ == "__main__":
    sys.exit(main())
