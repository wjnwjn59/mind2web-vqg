# `data/` — Dataset (do NOT commit to GitHub)

Holds the downloaded Mind2Web dataset and the data processed for the VQG
(question generation) task.

```
data/
├── raw/          # raw data downloaded from HuggingFace (written by m2w-download)
│   └── data/
│       ├── train/          train_*.json
│       ├── test_task/      test_task_*.json
│       ├── test_website/   test_website_*.json
│       └── test_domain/    test_domain_*.json
├── multimodal/   # screenshots + HTML (m2w-download-mm) — action-level parquet
│   └── data/
│       ├── train-0000*-of-00027-*.parquet
│       ├── test_task-0000*-of-00005-*.parquet
│       ├── test_website-0000*-of-00004-*.parquet
│       └── test_domain-00000*-of-00011-*.parquet
└── processed/    # data converted into (input/target) examples
    └── multimodal/<split>/
        ├── images/*.jpg        # written by `m2w-mm-export`
        └── examples.jsonl      # one JSON object per action
```

The two `.gitkeep` files only keep the folders alive in git; everything else is blocked by
`.gitignore` because the Mind2Web license **forbids redistributing the unzipped data**.

## Downloading the data

```bash
# install the environment first (once)
uv sync

# download train + test.zip and auto-extract it (password: mind2web)
uv run m2w-download

# train only
uv run m2w-download --skip-test

# test only
uv run m2w-download --skip-train

# re-download from scratch
uv run m2w-download --force

# custom destination
uv run m2w-download --dest D:\datasets\Mind2Web\raw
```

Quick sanity check after downloading:

```bash
uv run python -m mind2web_vqg.data.load
# train           1009 instances
# test_task        252 instances
# test_website     177 instances
# test_domain      912 instances
```

## Moving the data elsewhere

`src/mind2web_vqg/paths.py` reads the `MIND2WEB_DATA_DIR` environment variable:

```powershell
# PowerShell
$env:MIND2WEB_DATA_DIR = "D:\datasets\Mind2Web"
```

```bash
# bash / server
export MIND2WEB_DATA_DIR=/mnt/data/datasets/Mind2Web
```

Then `RAW_DIR = $MIND2WEB_DATA_DIR/raw`, i.e. the data lives at
`$MIND2WEB_DATA_DIR/raw/data/<split>/`.

## Screenshots: Multimodal-Mind2Web

The JSON above has no images (the `raw_html`/`cleaned_html` were stripped of styles and
assets, so they cannot be rendered back into a real page). Screenshots come from a second,
action-level dataset: [`osunlp/Multimodal-Mind2Web`](https://huggingface.co/datasets/osunlp/Multimodal-Mind2Web).

Download it with `m2w-download-mm` (it prints the file count, total size and free space,
and refuses nothing — check the numbers before letting it run):

```bash
uv run m2w-download-mm --dry-run                    # 5 files, ~1.3 GB (test_task)
uv run m2w-download-mm --splits test_task           # smallest split first
uv run m2w-download-mm --splits test_website test_domain
uv run m2w-download-mm --splits all                 # + train, ~13.6 GB total
```

Destination defaults to `$MIND2WEB_MM_DIR`, else `./data/multimodal`:

```bash
export MIND2WEB_MM_DIR=/mnt/data/datasets/Multimodal-Mind2Web   # lab convention
```

> ⚠️ If `MIND2WEB_MM_DIR` is unset **and** `MIND2WEB_DATA_DIR=/mnt/data/datasets/Mind2Web`,
> the default becomes `/mnt/data/datasets/Mind2Web/multimodal` — nested *inside* the Mind2Web
> folder. `m2w-download-mm` now warns about this. Already downloaded there? Move it (a plain
> `mv` on the same filesystem is instant, no copy):
> `mv /mnt/data/datasets/Mind2Web/multimodal /mnt/data/datasets/Multimodal-Mind2Web`

Layout and coverage (rows are **actions**, not tasks):

```
data/multimodal/data/
├── train-0000*-of-00027-*.parquet        7775 rows   7.6 GB
├── test_task-0000*-of-00005-*.parquet    1339 rows   1.3 GB
├── test_website-0000*-of-00004-*.parquet 1019 rows   1.1 GB
└── test_domain-00000*-of-00011-*.parquet 4060 rows   3.6 GB
```

Coverage is lower than the JSON dataset (177/142/694 tasks vs 252/177/912) because only
human-verified actions have images — which is why both datasets are kept side by side.

`m2w-download-mm` verifies afterwards by reading only parquet metadata (no image decoding):

```
[mm] layout under .../data
  test_task       5 file(s)    1339 rows  (screenshot=yes)
  total           5 file(s)    1339 rows
```

Re-running resumes: existing files are verified and skipped. Never pass `--force` unless you
really want to re-download.

## From parquet to something a model can read

```bash
# export screenshots as JPEG + metadata (default: data/processed/multimodal/<split>/)
uv run m2w-mm-export --split test_task --limit 5

# build the self-supervised VQG set from those screenshots
uv run m2w-build-vqg --split test_task --limit 5
# -> data/processed/vqg/test_task/examples.jsonl

# then run the pipeline smoke test
uv run m2w-vlm-smoke --split test_task --limit 5 --config configs/smoke.yaml
```

`examples.jsonl` fields (`m2w-mm-export`): `id`, `annotation_id`, `action_uid`, `task`,
`action_history`, `target_action`, `op`, `value`, `target_tag`, `target_type`,
`target_role`, `target_text`, `target_bbox`, `source_size`, `image_size`, `image`.
See `src/mind2web_vqg/data/README.md` for the full pipeline.

`m2w-build-vqg` turns every screenshot into up to four answered questions; each output line
is `{id, input, target, answer, image, meta}` where `input` is the generation prompt,
`target` the reference question and `answer` its ground truth. The four categories
(`target_label`, `element_type`, `target_region` and the opt-in `field_value`) are documented
once in `src/mind2web_vqg/data/README.md`.

```bash
uv run m2w-build-vqg --split test_task --categories target_label target_region
uv run m2w-build-vqg --split test_task --categories field_value
```

### Check what the HTML render looks like (optional, once per machine)

```bash
uv sync --extra render
uv run playwright install chromium
uv run m2w-render-html --split test_task --limit 3 --compare-mm
# -> outputs/render_check/test_task/index.html  (real screenshot vs rendered HTML)
```

This is a **sanity check**, not a data source: the JSON HTML had its CSS, fonts and assets
stripped, so rendering it produces an unstyled page. Screenshots only come from
Multimodal-Mind2Web.

## How data flows

Two chains exist: the **screenshot chain** (the real VQG task) and the original JSON chain.

```
# screenshots (the VQG input)
Multimodal-Mind2Web parquet --(data/multimodal.py)--> JPEGs + examples.jsonl
                            --(data/vqg.py)--------> [{input, target, answer, image, ...}]
                                                             |
                                                  evaluate.py / train.py

# JSON (still the complete benchmark)
data/raw/ --(data/load.py)--> list[dict]   (task text, HTML, actions)
```

- `load.py`: only reads JSON and concatenates every file in a split.
- `multimodal.py`: `m2w-mm-export` — parquet rows -> JPEGs + `examples.jsonl`
  (`data/processed/multimodal/<split>/`).
- `vqg.py`: **where the VQG task is defined** (`build_examples`). Rule-based,
  self-supervised questions whose answers are on the screen: `target_label` /
  `element_type` / `target_region`, plus opt-in `field_value`.
- `prepare.py`: the documented entry point — re-exports `build_examples` from `vqg.py`.
- `processed/`: output of the two steps above so it can be shared/reused
  (`m2w-mm-export` -> `data/processed/multimodal/`, `m2w-build-vqg` -> `data/processed/vqg/`).

## Main fields of an instance

- `annotation_id`, `website`, `domain`, `subdomain`
- `confirmed_task` — the task description in natural language
- `action_reprs` — human-readable action sequence
- `actions[]` — each step, containing:
  - `raw_html`, `cleaned_html` — page HTML **before** the step is performed
  - `operation` — `{op: CLICK|TYPE|SELECT, value}`
  - `pos_candidates[]`, `neg_candidates[]` — candidate elements

## Notes

- ⚠️ Don't `git add data/` — it's ignored, and don't try to force it in.
- It's fairly large (several GB); the first download takes a while, so run it in a stable
  terminal or inside `screen` on a server.
- `test.zip` can be deleted after extraction to save space.
