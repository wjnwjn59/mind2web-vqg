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
├── metrics.py         # question-quality metrics (offline) + round-trip answerability
├── smoke_vlm.py       # screenshot -> VLM smoke test (6 probes) -> outputs/smoke_vlm/
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
        ├─ data.vqg.load_exported(split)      # read data/processed/multimodal/<split>/examples.jsonl
        ├─ data.vqg.build_examples(...)       # -> [{id, input, target, answer, image, meta}, ...]
        ├─ models.build_model("api/..." | "hf/..." | "vlm/...")
        └─ for each example: model.generate(input, image=...)
                 │
                 ▼
        outputs/eval_<split>/predictions.json
        outputs/eval_<split>/metrics.json      # offline metrics (see metrics.py)
```

## How `m2w-vlm-smoke` flows

```
data/multimodal.py::export_samples   # parquet -> images/*.jpg + examples.jsonl
        │
        ▼
smoke_vlm.run_smoke                  # 6 probes x N screenshots
        │
        ├─ models.build_model("vlm/Qwen/Qwen3.5-9B")
        └─ model.generate(probe, image=path)
                 │
                 ▼
outputs/smoke_vlm/<split>_<timestamp>/{results.jsonl, report.md}
```

This is a **pipeline** check, not the research task: it only answers "can the model see this
screenshot, and is the latency acceptable?". The real VQG task is defined in `data/vqg.py`.

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

Env vars are read **once, at import** (so `.env` alone has no effect — export them in the
shell). The download scripts and the readers import these same constants, which is what keeps
a download destination and its read path in sync. Print what the current shell resolves to:

```bash
uv run python -m mind2web_vqg.paths
```

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

### `metrics.py`

Quality metrics for one eval run, in two families:

| Family | What | Cost |
| --- | --- | --- |
| Offline | format validity, diversity (unique ratio, Distinct-1/2, self-BLEU), ROUGE-L / chrF vs the reference question | free, runs inside `m2w-eval` |
| Round-trip | ask the model the *generated* question against the same screenshot, compare its answer with the ground truth — plus the *ceiling* (same loop with the reference question) | needs weights |

`m2w-eval` writes the offline part to `metrics.json` automatically; recompute or add the
round-trip with `uv run m2w-metrics --split <split> [--round-trip]`.

## Which file should you edit?

| Goal | Edit |
| --- | --- |
| Define the VQG task (what's the input, what's the target) | `data/vqg.py` 👈 **most important** (re-exported by `data/prepare.py`) |
| Change / add evaluation metrics | `metrics.py` |
| Add a new model backend | `models/` + `models/__init__.py::build_model` |
| Change the prompt | `prompts/question_gen.json` / `prompts/__init__.py` |
| Change the smoke-test probes | `smoke_vlm.py::PROBES` |
| Change the data location | env var, not code |
| Implement fine-tuning | `train.py` |

## Quick checks

```bash
uv run pytest                 # smoke tests
uv run ruff check src tests   # lint
```
