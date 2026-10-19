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
└── processed/    # data converted into (input/target) examples — currently empty
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

## How data flows

```
data/raw/  --(data/load.py)-->  list[dict]  --(data/prepare.py)-->  [{input, target, ...}]
                                                                          |
                                                              evaluate.py / train.py
```

- `load.py`: only reads JSON and concatenates every file in a split.
- `prepare.py`: **where you define the VQG task** (`build_examples`). Currently a skeleton:
  `input = confirmed_task`, `target = ""`.
- `processed/`: intended for the output of `prepare.py` so it can be shared/reused.
  No code writes here yet.

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
