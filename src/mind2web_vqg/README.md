# `src/mind2web_vqg/` — Main package

The package holds all project logic: download data → load data → build examples →
generate questions with a model → evaluate / fine-tune.

## File map

```
mind2web_vqg/
├── __init__.py        # __version__
├── paths.py           # canonical project paths (data, hf-cache, outputs, configs, prompts)
├── config.py          # YAML -> Config, accessed via cfg.get("model.name")
├── cli.py             # ensure_utf8(): fixes UnicodeEncodeError on Windows consoles
├── evaluate.py        # eval loop: load -> build_examples -> model.generate -> predictions.json
├── train.py           # fine-tuning skeleton (not implemented)
│
├── data/              # download & prepare data        -> data/README.md
├── models/            # model backends                  -> models/README.md
└── prompts/           # prompt templates                -> prompts/README.md
```

## How `m2w-eval` flows

```
configs/default.yaml
        │  load_config
        ▼
evaluate.run_eval
        │
        ├─ data.load.load_split(split)        # read JSON under data/raw/data/<split>/
        ├─ data.prepare.build_examples(...)   # -> [{id, input, target, meta}, ...]
        ├─ models.build_model("api/..." | "hf/...")
        └─ for each example: model.generate(input)
                 │
                 ▼
        outputs/eval_<split>/predictions.json
```

## Utility files

### `paths.py`

All paths are centralized here — **never hard-code paths** elsewhere.

| Variable | Default | Env override |
| --- | --- | --- |
| `PROJECT_ROOT` | repo root | — |
| `DATA_DIR` | `./data` | `MIND2WEB_DATA_DIR` |
| `RAW_DIR` | `./data/raw` | (follows `DATA_DIR`) |
| `MULTIMODAL_DIR` | `./data/multimodal` | `MIND2WEB_MM_DIR` |
| `PROCESSED_DIR` | `./data/processed` | (follows `DATA_DIR`) |
| `HF_CACHE_DIR` | `./hf-cache` | `MIND2WEB_HF_CACHE` |
| `CONFIG_DIR` | `./configs` | — |
| `OUTPUT_DIR` | `./outputs` | — |
| `PROMPT_DIR` | `src/mind2web_vqg/prompts` | — |

### `config.py`

```python
cfg = load_config()                  # configs/default.yaml
cfg = load_config("configs/x.yaml")  # another file
cfg.get("data.splits")               # ["test_task"]
cfg.get("does.not.exist", 42)        # 42
```

No Hydra, just `yaml.safe_load` — simple enough for the current needs.

### `cli.py`

`ensure_utf8()` is called at the start of every `main()` so printing Vietnamese text or
special characters doesn't break on Windows consoles (cp1252).

## Which file should you edit?

| Goal | Edit |
| --- | --- |
| Define the VQG task (what's the input, what's the target) | `data/prepare.py::build_examples` 👈 **most important** |
| Add evaluation metrics | `evaluate.py` |
| Add a new model backend | `models/` + `models/__init__.py::build_model` |
| Change the prompt | `prompts/question_gen.json` / `prompts/__init__.py` |
| Change the data location | env var, not code |
| Implement fine-tuning | `train.py` |

## Quick checks

```bash
uv run pytest                 # smoke tests
uv run ruff check src tests   # lint
```
