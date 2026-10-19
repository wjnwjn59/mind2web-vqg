# `src/mind2web_vqg/data/` — Downloading and preparing data

Three steps: **download** raw data → **load** it into Python objects → **transform** it into
VQG examples.

```
download.py             -->  data/raw/         (task-level JSON, no images)
download_multimodal.py  -->  data/multimodal/  (action-level parquet, screenshots)
        │
        └-->  load.py  -->  list[dict]  -->  prepare.py  -->  [{input, target}]
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

- Destination: `MULTIMODAL_DIR` = `$MIND2WEB_MM_DIR` (default `./data/multimodal`).
- `--dry-run` uses the Hub's dry-run API: reports file count and total size, writes nothing.
- Before downloading it prints free space on the destination and warns if the margin is thin.
- Re-running resumes (existing files are verified and skipped); `--force` re-downloads.
- After downloading, `verify()` reads only parquet metadata and prints rows per split plus a
  `screenshot=yes/no` flag, so you can confirm images are really there without decoding any.
- `--token` / `HF_TOKEN` is optional (public dataset) but lifts rate limits.

Sizes per split (from the Hub): `test_task` 5 files / 1.3 GB, `test_website` 4 / 1.1 GB,
`test_domain` 11 / 3.6 GB, `train` 27 / 7.6 GB.

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

## `prepare.py` — define the VQG task 👈

```python
def build_examples(instances: list[dict]) -> list[dict]:
    ...
```

This is **the most important place to edit**. It's currently a placeholder:

```python
{
    "id": inst.get("annotation_id"),
    "input": inst.get("confirmed_task", ""),   # TODO: real context (cleaned_html, task, ...)
    "target": "",                              # TODO: the expected question
    "meta": {"website": ..., "domain": ...},
}
```

When designing it, consider:

- **Input**: only `confirmed_task`? Add the `cleaned_html` of one step? How much HTML to keep?
  (Mind2Web HTML is very long and usually needs truncation or element filtering.)
- **Target**: which question should be asked to clarify the task, and where does it come from
  in the data?
- **Example count**: one example per instance, or one per action?
- Keep `id` and `meta` so you can trace results back and analyze by website/domain.

Both `evaluate.py` and `train.py` call this function, so one edit applies to both.

## `__init__.py`

Re-exports `SPLITS`, `load_split`, `load_all_splits` for convenience.
