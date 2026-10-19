# `configs/` — Run configuration

This folder holds YAML files describing **one run** (eval or train): which split to use,
which model, which prompt, where to write results.

## Current files

| File | Used by |
| --- | --- |
| `default.yaml` | Default config for `m2w-eval` and `m2w-train` |

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
| `data` | `root` | `evaluate.py` | Directory containing `train/`, `test_task/`, ... `null` = `data/raw/data` |
| `data` | `splits` | `evaluate.py` | List of splits; **only the first element** is used |
| `data` | `max_samples` | `evaluate.py` | Cap the number of examples for a quick test; `null` = all |
| `model` | `name` | `evaluate.py` | `api/<model>` or `hf/<repo_id>` (see `src/mind2web_vqg/models/README.md`) |
| `model` | `temperature`, `max_new_tokens` | *(not read yet)* | Declared but `evaluate.py` doesn't pass them to the model |
| `prompt` | `file`, `template` | *(not read yet)* | The real prompt lives in `src/mind2web_vqg/prompts/` |
| `output` | `dir` | *(not read yet)* | `evaluate.py` hard-codes `outputs/eval_<split>/` |
| `train` | `base_model`, `output_dir`, `epochs`, ... | `train.py` (only prints them) | Reserved for the fine-tuning pipeline (still TODO) |

> ⚠️ `model.temperature`, `model.max_new_tokens`, `prompt.*` and `output.dir` are **not wired
> into the code** yet. To make them take effect you must change `evaluate.py` (and/or `build_model`).

## Creating your own config

```bash
# copy, then edit
cp configs/default.yaml configs/my_run.yaml

# run with it
uv run m2w-eval --config configs/my_run.yaml
```

Name configs after their purpose, e.g. `api_gpt4omini_test_task.yaml` or
`flan_t5_base_test_website.yaml`, so you can tell runs apart later.
