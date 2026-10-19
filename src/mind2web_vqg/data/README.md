# `src/mind2web_vqg/data/` — Downloading and preparing data

Two chains: the **screenshot chain** (the real VQG task) and the task-level JSON chain.

```
download.py             -->  data/raw/         (task-level JSON, no images)
download_multimodal.py  -->  data/multimodal/  (action-level parquet, screenshots)
        │
        ├-->  load.py       -->  list[dict]                     (JSON chain)
        └-->  multimodal.py -->  images/*.jpg + examples.jsonl
                     │
                     └-->  vqg.py (via prepare.py) -->  [{input, target, answer, image}]
                     └-->  smoke_vlm.py            -->  VLM probe report
```

## `download.py` — download Mind2Web

Downloads from HuggingFace `osunlp/Mind2Web` via `snapshot_download`:

- `data/train/*.json` — the train split
- `test.zip` — extracted in place with password `mind2web`, producing
  `test_task/`, `test_website/`, `test_domain/`

```bash
uv run m2w-download              # both
uv run m2w-download --skip-test  # train only
uv run m2w-download --skip-train # test only
uv run m2w-download --force      # re-download
uv run m2w-download --dest D:\datasets\Mind2Web\raw
```

Default destination: `RAW_DIR` = `$MIND2WEB_DATA_DIR/raw` (default `./data/raw`).

Public function: `download(dest=None, *, skip_train=False, skip_test=False, force=False) -> Path`.

## `download_multimodal.py` — screenshots (Multimodal-Mind2Web)

The JSON above has no images. `osunlp/Multimodal-Mind2Web` is the action-level companion
dataset where every row carries the page screenshot that matches its `raw_html`/`cleaned_html`.

```bash
uv run m2w-download-mm --dry-run                     # size/space report only
uv run m2w-download-mm --splits test_task            # ~1.3 GB
uv run m2w-download-mm --splits test_website test_domain
uv run m2w-download-mm --splits all                  # ~13.6 GB
uv run m2w-download-mm --dest /mnt/data/datasets/Multimodal-Mind2Web
```

- Destination: `MULTIMODAL_DIR` = `$MIND2WEB_MM_DIR`, else `$MIND2WEB_DATA_DIR/multimodal`
  (local default `./data/multimodal`). On the lab server `MIND2WEB_DATA_DIR` is set to
  `/mnt/data/datasets/Mind2Web`, so leaving `MIND2WEB_MM_DIR` unset would nest the
  screenshots inside the Mind2Web folder — the script warns via `nested_dest_hint()`.
- `--dry-run` uses the Hub's dry-run API: reports file count and total size, writes nothing.
- Before downloading it prints free space on the destination and warns if the margin is thin.
- Re-running resumes (existing files are verified and skipped); `--force` re-downloads.
- After downloading, `verify()` reads only parquet metadata and prints rows per split plus a
  `screenshot=yes/no` flag, so you can confirm images are really there without decoding any.
- `--token` / `HF_TOKEN` is optional (public dataset) but lifts rate limits.

Sizes per split (from the Hub): `test_task` 5 files / 1.3 GB, `test_website` 4 / 1.1 GB,
`test_domain` 11 / 3.6 GB, `train` 27 / 7.6 GB.

## `multimodal.py` — parquet -> JPEGs + `examples.jsonl`

Turns parquet rows into something a model can consume. Each row is an **action**, not a task.

```bash
uv run m2w-mm-export --split test_task --limit 5
uv run m2w-mm-export --split test_task --limit 5 --max-width 1600 --force
```

```
data/processed/multimodal/test_task/
├── images/test_task_<annotation_id>_<row>.jpg
└── examples.jsonl
```

Each JSON line has: `id`, `annotation_id`, `action_uid`, `task`, `action_history`,
`target_action`, `op`, `value`, `target_tag`, `target_type`, `target_role`, `target_text`,
`target_bbox`, `source_size`, `image_size`, `image`. Screenshots are full-page (1280 px wide,
up to ~12k tall) and are downscaled to `--max-width` before encoding. `source_size` records
the raw screenshot size so `target_bbox` (DOM pixels) can be mapped onto the JPEG.

Key helpers (pure, unit-tested in `tests/test_multimodal.py`):

| Function | Purpose |
| --- | --- |
| `parquet_files(split, root=None)` | Sorted parquet files for a split |
| `as_json(value, default=None)` | Decode the JSON-encoded string columns (`operation`, `pos_candidates`, ...) |
| `parse_bbox("x,y,w,h")` | `[x, y, w, h]` floats |
| `target_element(pos_candidates)` | The `is_original_target` candidate |
| `from_row(row, index, split)` | Row -> example dict (no image decoding) |
| `iter_rows(...)` | Batched streaming iterator (memory-flat for 13 GB) |
| `load_screenshot(split, annotation_id, action_uid)` | Look up one screenshot with predicate pushdown |
| `fit_width(image, max_width)` | Downscale keeping the aspect ratio |

## `render_html.py` — Playwright sanity check

Renders `raw_html` / `cleaned_html` from the JSON to JPEG, next to the real screenshot.

```bash
uv sync --extra render
uv run playwright install chromium
uv run m2w-render-html --split test_task --limit 3 --compare-mm
# -> outputs/render_check/test_task/index.html
```

This exists to **prove the HTML cannot replace the screenshots**: the DOM was cleaned with
`Cleaner(style=True, inline_style=True, links=True, ...)`, so stylesheets, fonts and assets
are gone, there is no base URL, and `<text>` nodes are not real HTML. Expect an unstyled page.

## `load.py` — read the data

```python
from mind2web_vqg.data.load import SPLITS, load_split, load_all_splits

instances = load_split("test_task")            # list[dict]
instances = load_split("test_task", root=...)  # different root directory
all_data  = load_all_splits()                  # skips splits that weren't downloaded
```

- `SPLITS = {"train", "test_task", "test_website", "test_domain"}` → subdirectory names.
- `DEFAULT_ROOT = RAW_DIR / "data"`.
- Each JSON file can be a list, or a dict wrapping a list under `data` / `instances`.
- A missing split raises `FileNotFoundError` with a hint to run `m2w-download`.

Run it directly to see the counts:

```bash
uv run python -m mind2web_vqg.data.load
```

## `vqg.py` — the VQG task 👈

**This is where the task is defined.** It is *self-supervised Visual Question Generation*:
given a screenshot (and the user's task), the model asks a question whose answer can be read
off the screen. The answer is already on the screen, so the reference question is derived by
rule from the action metadata — no human or LLM labelling.

| Category | Question is about | Answer from | Default |
| --- | --- | --- | --- |
| `target_label` | the text on the element the task needs | `target_text` (screen) | yes |
| `element_type` | what kind of UI element it is | tag + input `type` (screen) | yes |
| `target_region` | where on the page it is | `target_bbox` (screen) | yes |
| `field_value` | the value to type or select | the task + operation | opt-in |

```bash
uv run m2w-mm-export --split test_task --limit 5
uv run m2w-build-vqg  --split test_task --limit 5
# -> data/processed/vqg/test_task/examples.jsonl
```

Each line is `{id, input, target, answer, image, meta}`: `input` is the generation prompt
(instruction + the user's task), `target` the reference question and `answer` its ground
truth. A category whose answer is missing for a row (no label, unmapped tag, off-page box)
is skipped, so every emitted example is answerable.

| Function | Purpose |
| --- | --- |
| `element_role(example)` | Human word for the element (button, link, search field, ...); `None` if unknown |
| `relative_region(bbox, source_size=, image_size=)` | Coarse region, e.g. `"top-right"` (rescales DOM -> JPEG) |
| `generate_questions(example, categories=None)` | Reference questions + answers for one screenshot |
| `build_examples(examples, categories=None, with_task=True)` | Full VQG set (one row per applicable category) |
| `load_exported(split, ...)` / `write_examples(...)` | Read `m2w-mm-export` output / write the VQG set |
| `category_stats(examples)` | Per-category counts |

Key helpers are pure and unit-tested in `tests/test_vqg.py`.

To change the task, edit the templates/categories in `vqg.py`, then rebuild with
`uv run m2w-build-vqg`.

## `prepare.py` — the shared entry point

`prepare.py` re-exports `build_examples` (and the other `vqg.py` helpers) so `evaluate.py` /
`train.py` keep a stable import path:

```python
from mind2web_vqg.data.prepare import build_examples, load_exported
```

Both `evaluate.py` and `train.py` call `build_examples`, so one edit applies to both.

## `__init__.py`

Re-exports `SPLITS`, `load_split`, `load_all_splits` for convenience.
