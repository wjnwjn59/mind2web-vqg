# `configs/` — Run configuration

This folder holds YAML files describing **one run** (eval or train): which split to use,
which model, which prompt, where to write results.

## Current files

| File | Used by |
| --- | --- |
| `default.yaml` | Default config for `m2w-eval` and `m2w-train` |
| `smoke.yaml` | `m2w-vlm-smoke` — split/limit/model/probes for the screenshot pipeline test |
| `eval_vlm.yaml` | `m2w-eval` with the local VLM on `test_task`, 6 examples (needs `uv sync --extra vlm`) |

## How the code reads a config

`src/mind2web_vqg/config.py` loads the YAML into a `Config` object, accessed with dotted keys:

```python
from mind2web_vqg.config import load_config

cfg = load_config("configs/default.yaml")   # empty -> falls back to configs/default.yaml
split = cfg.get("data.splits")              # ["test_task"]
name  = cfg.get("model.name")               # "api/gpt-4o-mini"
```

`cfg.get("a.b.c", default)` returns `default` when a key is missing, so you don't have to
declare every field.

## Field groups in `default.yaml`

| Group | Field | Read by | Meaning |
| --- | --- | --- | --- |
| `data` | `splits` | `evaluate.py` | List of splits; **only the first element** is used |
| `data` | `examples` | `evaluate.py` | Override the input `examples.jsonl`; `null` = `$MIND2WEB_DATA_DIR/processed/multimodal/<split>/examples.jsonl` |
| `data` | `max_samples` | `evaluate.py` | Cap the number of examples for a quick test; `null` = all (the `--max-samples` flag overrides it) |
| `model` | `name` | `evaluate.py` | `api/`, `hf/` or `vlm/<repo_id>` (see `src/mind2web_vqg/models/README.md`); the `--model` flag overrides it |
| `model` | `temperature`, `max_new_tokens` | *(not read yet)* | Declared but `evaluate.py` doesn't pass them to the model |
| `prompt` | `file`, `template` | *(not read yet)* | The real prompt lives in `src/mind2web_vqg/prompts/` |
| `output` | `dir` | *(not read yet)* | `evaluate.py` hard-codes `outputs/eval_<split>/` |
| `train` | `base_model`, `output_dir`, `epochs`, ... | `train.py` (only prints them) | Reserved for the fine-tuning pipeline (still TODO) |

> ⚠️ `model.temperature`, `model.max_new_tokens`, `prompt.*` and `output.dir` are **not wired
> into the code** yet. To make them take effect you must change `evaluate.py` (and/or `build_model`).

For one-off runs you don't need a second file: `m2w-eval --model vlm/Qwen/Qwen3.5-9B
--max-samples 6` overrides `model.name` and `data.max_samples` from the config.

## `smoke.yaml` fields

Read by `smoke_vlm.py` via `cfg.get(...)`:

| Field | Meaning |
| --- | --- |
| `data.split` | Which split to export from (`test_task` is smallest) |
| `data.limit` | How many screenshots to run the probes on |
| `data.max_width` | *Documented only* — export downscaling is a CLI flag (`--max-width`) |
| `model.name` | `vlm/<repo>` (local) or `api/<model>` (served endpoint) |
| `model.max_new_tokens` | Answer length cap |
| `model.enable_thinking` | Qwen3.5: `false` = direct answer, no ` thinking` block |
| `probes` | Subset of `what_heading`, `descriptive`, `yes_no`, `count`, `how_next`, `where_target`; `null` = all |

## Creating your own config

```bash
# copy, then edit
cp configs/default.yaml configs/my_run.yaml

# run with it
uv run m2w-eval --config configs/my_run.yaml
```

Name configs after their purpose, e.g. `api_gpt4omini_test_task.yaml` or
`flan_t5_base_test_website.yaml`, so you can tell runs apart later.
