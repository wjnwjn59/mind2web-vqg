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
| `m2w-render-html` | `data/render_html.py:main` | ✅ works (Playwright sanity check) |
| `m2w-vlm-smoke` | `smoke_vlm.py:main` | ✅ works (screenshot -> VLM, 6 probes) |
| `m2w-eval` | `evaluate.py:main` | ⚠️ runs, but the task is still a placeholder |
| `m2w-train` | `train.py:main` | ❌ `NotImplementedError` |

Run them with `uv run <command>` (no need to activate the venv).
