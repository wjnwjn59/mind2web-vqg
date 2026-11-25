# `src/` — Source code

There is exactly **one** Python package here: `mind2web_vqg/`. See
[`mind2web_vqg/README.md`](mind2web_vqg/README.md) for details.

## Why the `src/` layout?

So that `import mind2web_vqg` only works once the package is installed (`uv sync`), which
prevents accidentally importing the current directory instead of the real package.
Packaging is configured in `pyproject.toml`:

```toml
[tool.hatch.build.targets.wheel]
packages = ["src/mind2web_vqg"]
```

`uv sync` installs the package in editable mode, so edits under `src/` take effect
immediately — no reinstall needed.

## Declared CLI entry points

| Command | Points to | Status |
| --- | --- | --- |
| `m2w-download` | `data/download.py:main` | ✅ works |
| `m2w-download-mm` | `data/download_multimodal.py:main` | ✅ works (Multimodal-Mind2Web screenshots) |
| `m2w-download-model` | `models/download.py:main` | ✅ works |
| `m2w-mm-export` | `data/multimodal.py:main` | ✅ works (parquet -> JPEG + `examples.jsonl`) |
| `m2w-build-vqg` | `data/vqg.py:main` | ✅ works (`examples.jsonl` -> VQG examples) |
| `m2w-build-treevqg` | `treevqg/build.py:main` | ✅ works (HTML + screenshots -> 3-level TreeVQG JSONL) |
| `m2w-eval-treevqg` | `treevqg/evaluate.py:main` | ✅ works (TreeVQG JSONL -> predictions + per-level metrics) |
| `m2w-audit-treevqg` | `treevqg/audit.py:main` | ✅ works (leakage / template collapse / level balance -> `audit.md`) |
| `m2w-build-clarify` | `treevqg/clarify.py:main` | ✅ works (H2 clarification questions -> JSONL + stats) |
| `m2w-audit-clarify` | `treevqg/clarify_audit.py:main` | ✅ works (option coverage / referent visibility -> `audit.md`) |
| `m2w-eval-clarify` | `treevqg/clarify_metrics.py:main` | ✅ works (Ask-F1 + information gain from a predictions file) |
| `m2w-build-bench` | `treevqg/bench.py:main` | ✅ works (6 question types, stratified by website -> JSONL + stats) |
| `m2w-eval-bench` | `treevqg/bench_eval.py:main` | ✅ works (local model generates bench questions -> predictions + metrics + report) |
| `m2w-visualize-bench` | `treevqg/visualize.py:main` | ✅ works (crop + box + Q/A gallery as HTML) |
| `m2w-swift-data` | `data/swift.py:main` | ✅ works (VQG -> ms-swift JSONL + train/val split) |
| `m2w-render-html` | `data/render_html.py:main` | ✅ works (Playwright sanity check) |
| `m2w-vlm-smoke` | `smoke_vlm.py:main` | ✅ works (screenshot -> VLM, 6 probes) |
| `m2w-eval` | `evaluate.py:main` | ✅ works (writes `predictions.json` + `metrics.json`) |
| `m2w-metrics` | `metrics.py:main` | ✅ works (offline + `--round-trip`) |
| `m2w-report` | `report.py:main` | ✅ works (`report.md` + `report.html` gallery) |
| `m2w-eda` | `eda.py:main` | ✅ works (HTML-tree EDA, `--all-splits`) |
| `m2w-eda-report` | `eda_report.py:main` | ✅ works (LaTeX report + matplotlib PNG charts) |
| `m2w-train` | `train.py:main` | ✅ works (builds data, runs `swift sft`, merges; needs ms-swift in its own env) |

Run them with `uv run <command>` (no need to activate the venv).
