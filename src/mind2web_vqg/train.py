"""Fine-tune a question generation model (run on the cloud: Colab/Kaggle/rented GPU).

The skeleton is ready; you need to:
  1. Install the train extra: ``uv sync --extra train``
  2. Define the training data in ``data/prepare.py::build_examples``
  3. Adjust the ``train`` section of ``configs/default.yaml``.

Usage::

    uv run m2w-train --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mind2web_vqg.cli import ensure_utf8
from mind2web_vqg.config import load_config


def run_train(config_path: str | Path | None = None) -> None:
    cfg = load_config(config_path)
    base_model = cfg.get("train.base_model")
    output_dir = cfg.get("train.output_dir")
    print(f"[train] base_model={base_model} -> output_dir={output_dir}")

    # TODO: finish the fine-tuning pipeline:
    #   1) build the dataset (build_examples + tokenize)
    #   2) transformers.Seq2SeqTrainer / Trainer + TrainingArguments
    #   3) save checkpoints to output_dir
    raise NotImplementedError("Training pipeline not implemented yet — see the notes in this file.")


def main(argv: list[str] | None = None) -> int:
    ensure_utf8()
    parser = argparse.ArgumentParser(description="Fine-tune a question generation model.")
    parser.add_argument("--config", default=None, help="path to a YAML file")
    args = parser.parse_args(argv)
    run_train(args.config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
