# `tests/` — Testing

Uses `pytest` (a dev dependency, installed by `uv sync`).

## Running

```bash
uv run pytest                 # everything
uv run pytest -v              # verbose
uv run pytest tests/test_smoke.py::test_config_loads
```

## Current files

`test_smoke.py` — minimal checks that the project is still healthy:

| Test | Checks |
| --- | --- |
| `test_version` | `mind2web_vqg.__version__` is non-empty |
| `test_config_loads` | `configs/default.yaml` loads and has `model.name` |
| `test_project_root` | `PROJECT_ROOT/pyproject.toml` exists |

These are smoke tests: fast, no data, no API calls.

## What's worth adding

- `load_split` against a fake directory (pytest `tmp_path`) with sample JSON → verify file
  concatenation, dict-wrapped lists, and the error for an unknown split.
- `build_examples` → every example has `id`, `input`, `target`; `len(examples)` is correct.
- `models.build_model` → unknown prefix raises `ValueError`; an `api/` prefix returns the
  right class (mocked, no API key needed).
- `resolve_model_path` → a repo id maps to `<ns>_<repo>` when that folder exists.

## Conventions

- No test may hit the network, need an API key, or require the real dataset.
- Use pytest's `tmp_path` fixture instead of creating files inside the repo.
- File names `test_*.py`, function names `test_*`.
- Lint alongside: `uv run ruff check src tests`.
