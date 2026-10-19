# `src/mind2web_vqg/models/` — Model backends

Every model implements the same interface, so `evaluate.py` doesn't care whether it's
talking to an API or a local model.

## Interface

```python
class QuestionGenerator(ABC):
    def generate(self, prompt: str) -> str: ...
    def generate_batch(self, prompts: list[str]) -> list[str]:  # defaults to sequential calls
```

## Factory: `build_model(name)`

The prefix selects the backend:

| `name` | Class | Requires |
| --- | --- | --- |
| `api/gpt-4o-mini` | `APIModel` | `uv sync --extra api` |
| `hf/google/flan-t5-base` | `HFModel` | `uv sync --extra local` |

```python
from mind2web_vqg.models import build_model

model = build_model("api/gpt-4o-mini")
model = build_model("hf/google/flan-t5-base", max_new_tokens=128)
```

Any other prefix raises `ValueError`.

## `base.py`

Only the `QuestionGenerator` abstract class. `generate_batch` loops over `generate` by
default; subclasses can override it to batch/async for speed.

## `api.py` — API baseline

- Uses the OpenAI Chat Completions API (the `openai` library).
- Reads `OPENAI_API_KEY` and `OPENAI_BASE_URL` from the environment → you need a `.env`
  file (copied from `.env.example`) and to load it (e.g. `python-dotenv`, or export manually).
- Parameters: `model`, `temperature=0.0`, `max_tokens=512`, `system_prompt=None`.
- `openai` is imported **lazily** inside `__init__`, so the module imports fine without the
  `api` extra installed.
- ⚠️ `evaluate.py` doesn't pass `temperature`/`max_new_tokens` from the config yet, so those
  fields in `configs/default.yaml` currently have no effect.

## `hf.py` — local HuggingFace models

- `resolve_model_path(name)`: if `name` is an existing path, use it; else if
  `<HF_CACHE_DIR>/<name with '/' -> '_'>` exists, use that folder; otherwise return `name`
  so transformers downloads it from the hub.
- `HFModel` loads with `AutoTokenizer` + `AutoModelForSeq2SeqLM` (good for T5/BART;
  causal models like Llama need `AutoModelForCausalLM` instead).
- Picks `cuda` if available, otherwise `cpu`.
- `load_in_8bit=True` requires `bitsandbytes` (not declared in the dependencies).
- ⚠️ `generate()` only passes `truncation=True` with the tokenizer's default length — long
  HTML gets cut without control; set `max_length` explicitly for the real task.

## `download.py` — download weights using the lab convention

Folder name = repo id with `/` → `_`:

```
google/flan-t5-base  ->  $MIND2WEB_HF_CACHE/google_flan-t5-base
```

```bash
uv run m2w-download-model google/flan-t5-base
uv run m2w-download-model google/flan-t5-base --cache-dir D:\hf-cache
uv run m2w-download-model google/flan-t5-base --force
```

Equivalent HuggingFace CLI command:

```bash
hf download google/flan-t5-base --local-dir /mnt/data/hf-cache/google_flan-t5-base
```

Thanks to this convention, the config only needs `hf/google/flan-t5-base` and
`resolve_model_path` finds the downloaded folder — no absolute path required.

## `__init__.py`

Exports `QuestionGenerator` and the `build_model` factory.

## Adding a new backend

1. Create `my_backend.py` with a class extending `QuestionGenerator` that implements `generate`.
2. Add a prefix branch in `build_model` (`"my/" -> MyModel`).
3. (Optional) add a matching extra in `pyproject.toml`.
