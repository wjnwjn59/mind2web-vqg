# `src/mind2web_vqg/models/` — Model backends

Every model implements the same interface, so `evaluate.py` doesn't care whether it's
talking to an API or a local model.

## Interface

```python
class QuestionGenerator(ABC):
    def generate(self, prompt: str, image: str | Path | None = None) -> str: ...
    def generate_batch(self, prompts, images=None) -> list[str]:  # defaults to sequential calls
```

`image` is optional: text-only backends ignore it (or raise, like `HFModel`), vision
backends must use it.

## Factory: `build_model(name)`

The prefix selects the backend:

| `name` | Class | Requires |
| --- | --- | --- |
| `api/gpt-4o-mini` | `APIModel` | `uv sync --extra api` |
| `hf/google/flan-t5-base` | `HFModel` | `uv sync --extra local` |
| `vlm/Qwen/Qwen3.5-9B` | `VLMModel` | `uv sync --extra vlm` |

```python
from mind2web_vqg.models import build_model

model = build_model("api/gpt-4o-mini")
model = build_model("hf/google/flan-t5-base", max_new_tokens=128)
model = build_model("vlm/Qwen/Qwen3.5-9B", max_new_tokens=256, enable_thinking=False)

model.generate("What is the main heading?", image="data/processed/.../x.jpg")
```

Any other prefix raises `ValueError`.

## `base.py`

Only the `QuestionGenerator` abstract class. `generate_batch` loops over `generate` by
default; subclasses can override it to batch/async for speed.

## `api.py` — API baseline

- Uses the OpenAI Chat Completions API (the `openai` library).
- Reads `OPENAI_API_KEY` and `OPENAI_BASE_URL` from the environment → you need a `.env`
  file (copied from `.env.example`) and to load it (e.g. `python-dotenv`, or export manually).
- Supports images: a local path is base64-encoded into a `data:image/...` URL, so the same
  class works with `gpt-4o-mini` **and** with a local vLLM/SGLang server
  (`OPENAI_BASE_URL=http://localhost:8000/v1`, key can be anything).
- Parameters: `model`, `temperature=0.0`, `max_tokens=512`, `system_prompt=None`,
  `extra_body=None` (e.g. `{"chat_template_kwargs": {"enable_thinking": False}}`).
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
- `generate(image=...)` raises: this backend is text-only, use `vlm/...` for screenshots.
- ⚠️ `generate()` only passes `truncation=True` with the tokenizer's default length — long
  HTML gets cut without control; set `max_length` explicitly for the real task.

## `vlm.py` — local vision-language models

For screenshots. Requires `uv sync --extra vlm` (torch + transformers>=5.20 + torchvision).

```python
model = build_model("vlm/Qwen/Qwen3.5-9B")
model.generate("What is the main heading?", image="screenshot.jpg")
```

- `resolve_model_path` is reused, so a model downloaded with `m2w-download-model` is found
  automatically under `$MIND2WEB_HF_CACHE/<namespace>_<repo>`.
- Loads with `AutoProcessor` + `AutoModelForImageTextToText`, `dtype=bfloat16` on GPU.
- `enable_thinking=False` (default) asks Qwen3.5 for a direct answer instead of a
  ` thinking...<｜end▁of▁thinking｜>` block. Any leftover block is stripped by `strip_thinking()`.
- `load_in_4bit=True` uses bitsandbytes NF4 (handy for 9B on a 16 GB card).
- **No `flash-attn`** (lab rule); default SDPA attention is used.
- ⚠️ Full-page screenshots are 1280 px wide and up to ~12k px tall. The processor's default
  pixel budget will downscale them hard; for text-heavy pages, tile the page or pass
  `min_pixels`/`max_pixels` (Qwen2.5-VL) and accept the token cost.

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

> ⚠️ `m2w-download-model` writes to `$MIND2WEB_HF_CACHE`, and **falls back to `<repo>/hf-cache`
> inside the repository when that variable is unset** — so on the server, export it first:
>
> ```bash
> export MIND2WEB_HF_CACHE=/mnt/data/hf-cache   # lab convention
> ```
>
> The command prints the destination before downloading and warns (`hf_cache_hint`) when the
> env var is missing. `--cache-dir` overrides it for a single run.

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

## `smoke_vlm.py` (in the package root, not here)

Not a backend — the pipeline test that uses `build_model` + `data/multimodal.py`. See the
project README section 5.
