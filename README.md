# mind2web-vqg

Research on **question generation (VQG)** built on the [Mind2Web](https://osu-nlp-group.github.io/Mind2Web/) dataset.

Current goals:
- ✅ Set up the repository skeleton and environment (using [`uv`](https://docs.astral.sh/uv/)).
- ✅ Download and load the Mind2Web data (train + test).
- 🔜 Test the "question-asking model" (an LLM **API** baseline **and** a local HuggingFace model).
- 🔜 Fine-tune on the cloud (Colab / Kaggle / rented GPU) when needed.

---

## 1. Requirements

| Requirement | Notes |
| --- | --- |
| [uv](https://docs.astral.sh/uv/) | Manages Python + dependencies. Already available on your machine. |
| Python 3.11+ | Installed automatically by `uv`; no manual install needed. |
| `git-lfs` | Only needed if you `git clone` the dataset from HF (the download script is preferred). |
| GPU | **Not required for testing.** Training should run on the cloud. |

---

## 2. Installation

```bash
# Clone (if you don't have it yet)
git clone https://github.com/wjnwjn59/mind2web-vqg.git
cd mind2web-vqg

# Install the core dependencies (creates .venv)
uv sync

# Install extras as needed:
uv sync --extra api     # LLM API baseline (openai)
uv sync --extra local   # run a local HF model (torch + transformers)
uv sync --extra vlm     # local vision-language model (screenshots)
uv sync --extra render  # Playwright, to render HTML as a sanity check
uv sync --extra train   # fine-tuning (adds peft, sentencepiece)
```

Quick optional step: copy the environment template

```bash
cp .env.example .env   # then fill in OPENAI_API_KEY...
```

---

## 3. Repository structure

```
mind2web-vqg/
├── pyproject.toml            # project metadata + dependencies (managed by uv)
├── .python-version           # Python version (3.11)
├── .gitignore                # ignores data/, outputs/, .venv/, .env...
├── .env.example              # environment template (API keys...)
├── README.md
│
├── configs/
│   └── default.yaml          # run config for eval/train (split, model, prompt...)
│
├── data/                     # ⚠️ DO NOT commit (gitignored)
│   ├── raw/                  # downloaded Mind2Web data (see section 4)
│   ├── multimodal/           # Multimodal-Mind2Web parquet + screenshots (section 5)
│   └── processed/            # data preprocessed for the VQG task
│
├── hf-cache/                 # ⚠️ downloaded model weights (gitignored, see section 8)
│
├── src/mind2web_vqg/         # main code (Python package)
│   ├── __init__.py
│   ├── paths.py              # canonical project paths
│   ├── config.py             # YAML config loader
│   ├── cli.py                # CLI helpers (UTF-8 output...)
│   ├── data/
│   │   ├── download.py       # download Mind2Web (train + test.zip)
│   │   ├── download_multimodal.py # download Multimodal-Mind2Web (screenshots)
│   │   ├── multimodal.py     # parquet -> JPEGs + examples.jsonl
│   │   ├── render_html.py    # Playwright render check (HTML -> PNG)
│   │   ├── load.py           # load splits into list[dict]
│   │   └── prepare.py        # 👉 define the VQG task HERE
│   ├── models/
│   │   ├── base.py           # QuestionGenerator interface (text + image)
│   │   ├── api.py            # API baseline (OpenAI, vision-capable)
│   │   ├── hf.py             # local HuggingFace model (text)
│   │   ├── vlm.py            # local vision-language model (Qwen-VL/Qwen3.5)
│   │   └── download.py       # download weights via the hf-cache convention
│   ├── prompts/
│   │   └── question_gen.json # prompt template
│   ├── smoke_vlm.py          # screenshot -> VLM smoke test (6 simple probes)
│   ├── evaluate.py           # evaluation loop -> outputs/
│   └── train.py              # fine-tuning skeleton (run on the cloud)
│
├── notebooks/                # data exploration (Jupyter)
├── tests/                    # pytest
└── outputs/                  # ⚠️ DO NOT commit (gitignored)
```

**Conventions:** code lives in `src/mind2web_vqg/`, data in `data/`, results in `outputs/`,
and model weights in `hf-cache/`.

### Per-folder docs (start here if you're new to the project)

| Folder | Contents |
| --- | --- |
| [`configs/README.md`](configs/README.md) | What each YAML field means, and which fields the code doesn't read yet |
| [`data/README.md`](data/README.md) | How to download the dataset, folder layout, instance fields |
| [`src/README.md`](src/README.md) | Why the `src/` layout, list of CLI commands |
| [`src/mind2web_vqg/README.md`](src/mind2web_vqg/README.md) | Eval flow, `paths`/`config`/`cli`, which file to edit |
| [`src/mind2web_vqg/data/README.md`](src/mind2web_vqg/data/README.md) | `download` → `load` → `prepare`, plus the screenshot dataset |
| [`src/mind2web_vqg/models/README.md`](src/mind2web_vqg/models/README.md) | Model interface, API/HF/VLM backends, `hf-cache` convention |
| [`src/mind2web_vqg/prompts/README.md`](src/mind2web_vqg/prompts/README.md) | Prompt templates (note: not wired into eval yet) |
| [`notebooks/README.md`](notebooks/README.md) | Running Jupyter, snippets for exploring the data |
| [`tests/README.md`](tests/README.md) | Running pytest, what to test next |
| [`outputs/README.md`](outputs/README.md) | What's inside `predictions.json` |

---

## 4. Downloading the Mind2Web data

The data is hosted on HuggingFace: [`osunlp/Mind2Web`](https://huggingface.co/datasets/osunlp/Mind2Web).

```bash
# Download everything (train + test), auto-extracting test.zip (password: mind2web)
uv run m2w-download

# Train only
uv run m2w-download --skip-test

# Test only
uv run m2w-download --skip-train

# Re-download from scratch
uv run m2w-download --force
```

After downloading, the layout is:

```
data/raw/
├── data/
│   ├── train/          train_*.json
│   ├── test_task/      test_task_*.json
│   ├── test_website/   test_website_*.json
│   └── test_domain/    test_domain_*.json
└── test.zip            (already extracted, safe to delete)
```

> On the lab server, change the target directory with an env var (see section 8):
> `export MIND2WEB_DATA_DIR=/mnt/data/datasets/Mind2Web`

Quick sanity check:

```bash
uv run python -m mind2web_vqg.data.load
# train           1009 instances
# test_task        252 instances
# test_website     177 instances
# test_domain      912 instances
```

### Splits

| Split | # instances | Meaning |
| --- | --- | --- |
| `train` | 1,009 | Training |
| `test_task` | 252 | Seen websites, **new tasks** (Cross-Task) |
| `test_website` | 177 | **New websites** (Cross-Website) |
| `test_domain` | 912 | **New domains** (Cross-Domain) |

### Main fields of each instance

- `annotation_id`, `website`, `domain`, `subdomain`
- `confirmed_task` — the task description in natural language
- `action_reprs` — human-readable action sequence
- `actions[]` — each step, containing:
  - `raw_html`, `cleaned_html` — the page HTML before the step is performed
  - `operation` — `{op: CLICK|TYPE|SELECT, value}`
  - `pos_candidates[]` / `neg_candidates[]` — candidate elements

---

## 5. Screenshots, the VLM smoke test, and the VQG task

Mind2Web's JSON has **no images** — `raw_html` / `cleaned_html` are DOM dumps whose CSS,
fonts and assets were stripped. So there are two separate things to check:

**A. Render the HTML with Playwright (visual check only).** This shows how unusable the
stripped HTML is as an "image", and it's the fastest way to convince yourself that the
screenshots must come from elsewhere.

```bash
uv sync --extra render
uv run playwright install chromium            # once per machine, ~150 MB

uv run m2w-render-html --split test_task --limit 3 --compare-mm
# -> outputs/render_check/test_task/index.html  (raw | cleaned | real screenshot)
```

**B. Use the real screenshots from `osunlp/Multimodal-Mind2Web`** (action-level parquet,
image embedded in each row). Download, export, then run a VLM:

```bash
export MIND2WEB_MM_DIR=/mnt/data/datasets/Multimodal-Mind2Web   # lab convention
uv run m2w-download-mm --dry-run                                # file count + size only
uv run m2w-download-mm --splits test_task                       # 1.3 GB

uv run m2w-mm-export --split test_task --limit 5                # -> data/processed/multimodal/...
uv run m2w-vlm-smoke --split test_task --limit 5                # -> outputs/smoke_vlm/...
```

`m2w-vlm-smoke` asks six deliberately simple probes per screenshot, one per classic VQA
type (`what` / `descriptive` / `yes-no` / `count` / `how` / `where`), and writes
`results.jsonl` + `report.md`. The point is to verify three things before designing the
real task: the image is readable, the model actually sees it, and latency is acceptable.

Model choices:

```bash
# local transformers: first fetch the weights (lab convention -> $MIND2WEB_HF_CACHE)
uv run m2w-download-model Qwen/Qwen3.5-9B       # -> /mnt/data/hf-cache/Qwen_Qwen3.5-9B

# needs `uv sync --extra vlm`; ~18 GB VRAM for 9B in bf16
uv run m2w-vlm-smoke --model vlm/Qwen/Qwen3.5-9B

# a served model (vLLM / SGLang / transformers serve) — much faster, no local weights
export OPENAI_BASE_URL=http://localhost:8000/v1
uv run m2w-vlm-smoke --model api/Qwen/Qwen3.5-9B
```

> Why `Qwen3.5-9B`: it is natively multimodal (`pipeline_tag: image-text-to-text`), fits
> one 24 GB card at bf16, and works with `transformers>=5.20`. Qwen3.5 **thinks by
> default**; `enable_thinking: false` in `configs/smoke.yaml` gives short, direct answers.

### The VQG task (self-supervised)

The task is **what the model must ask**, and it lives in `data/vqg.py`
(`data/prepare.py::build_examples` re-exports it). Given a screenshot, the model asks a
question whose answer is **on the screen** — so the reference question and its answer are
derived by rule from the action metadata, with no human or LLM labelling:

| Category | Question is about | Answer comes from |
| --- | --- | --- |
| `target_label` | the text shown on the element the task needs | the element's visible text |
| `element_type` | what kind of UI element it is (button, link, text field, ...) | tag + input `type` |
| `target_region` | where on the page it is located | the element's bounding box |
| `field_value` | the value to type or select *(opt-in)* | the task + operation |

The first three are **screenshot-grounded** and are the default; `field_value` is the only
one answered from the task text, so it must be requested explicitly.

```bash
uv run m2w-build-vqg --split test_task --limit 5
# [vqg] 5 screenshot(s) -> 13 example(s) -> data/processed/vqg/test_task/examples.jsonl

uv run m2w-build-vqg --split test_task --limit 5 --categories field_value
```

Each line is `{id, input, target, answer, image, meta}`: `input` is the generation prompt
(instruction + the user's task), `target` the reference question, `answer` its ground truth.
A category whose answer is missing for a row (no label, unmapped tag, off-page box) is
skipped, so every emitted example stays answerable.

### EDA — what the data actually contains

Before synthesizing questions from the whole page tree, measure its class balance: the tags
that dominate the tree will dominate the generated data.

```bash
uv run m2w-eda --split test_task --limit 200   # first 200 tasks of one split
uv run m2w-eda --split train --limit 0         # the whole split (0 = no limit)
uv run m2w-eda --all-splits --limit 0          # every downloaded split + comparison

# -> outputs/eda/<split>/{eda.md,eda.json,tags.csv}
#    outputs/eda/{summary.md,summary.json}     (only with --all-splits)
```

Parsing is CPU-bound (pure-Python HTML parser): expect minutes per split and ~10-30 min
for everything, so run the full pass inside `screen`. The per-split report answers "which
types dominate"; `summary.md` answers "do the splits differ".

It parses `actions[*].cleaned_html` and reports: the element-type distribution and the share
of steps containing each type; the **recorded targets** next to the tree with a `lift` column
(target share / tree share) so a type the agent is forced to use shows up; the VQG potential
per category; and side aspects (input types, operations, task/step lengths, sites, candidate
counts, repeated page states).

---

## 6. Running evaluation

```bash
uv run m2w-eval --config configs/default.yaml
```

`m2w-eval` writes `outputs/eval_<split>/predictions.json` plus `metrics.json` with offline
question-quality metrics (format, diversity, ROUGE-L/chrF vs the reference). Recompute it, or
add the model-based round-trip answerability metric, with:

```bash
uv run m2w-metrics --split test_task --round-trip
```

For a readable view of the same run (metrics tables plus a screenshot gallery):

```bash
uv run m2w-report --split test_task
```

Select the model in `configs/default.yaml`:

```yaml
model:
  name: "api/gpt-4o-mini"           # API baseline
  # name: "hf/google/flan-t5-base"  # local HF model (resolved from hf-cache if downloaded)
```

For a one-off run, override the model and the sample cap on the command line instead of
editing the config — e.g. six examples with the local VLM:

```bash
CUDA_VISIBLE_DEVICES=1 uv run m2w-eval --model vlm/Qwen/Qwen3.5-9B --max-samples 6
```

> Changing the task: the VQG definition lives in `src/mind2web_vqg/data/vqg.py`, re-exported
> by `src/mind2web_vqg/data/prepare.py::build_examples`. Edit the categories/templates there,
> then rebuild with `uv run m2w-build-vqg`.

---

## 7. Fine-tuning (cloud / lab server)

The training skeleton lives in `src/mind2web_vqg/train.py` (still a TODO). Suggested workflow:

1. **Prepare locally:** download the data, then `m2w-mm-export` + `m2w-build-vqg` →
   `data/processed/vqg/`.
2. **Move to the cloud/server:** Colab/Kaggle/rented GPU or the lab server (section 8);
   install `uv` (or pip) + `uv sync --extra train`.
3. **Train:** `uv run m2w-train --config configs/default.yaml`.
4. **Bring the model back locally** to re-evaluate it with `hf/...`.

---

## 8. Running on the lab server (AIVN-VLAI)

Mandatory lab conventions (summary — read the full original document too):

- For **research purposes only**; do all your work under `/home/{username}`.
- **GPUs:** general-purpose work (testing ideas...) uses **card index 1**; if you need more
  GPUs or have an urgent task, tell Mr. Thang first. Select GPUs with `CUDA_VISIBLE_DEVICES`.
- **Datasets:** prefer downloading into `/mnt/data/datasets`, **grouped by dataset name**.
- **Model weights:** use the `hf` CLI, store under `/mnt/data/hf-cache`, with folder names
  `<namespace>_<repo_name>` (replace `/` with `_`).
- **`screen`:** name it `servername + task` (e.g. `thangdd_vqg_train`).
- ⛔ **Do NOT use/install `flash-attn`** (per Mr. Thuan's update).

### Installation

The server ships with conda, but `uv` works too:

```bash
# (optional) install uv if the server doesn't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# or use conda as the lab instructions suggest
# conda create -n "<servername>_env" python=3.11 -y && conda activate "<servername>_env"
```

### Point the paths at the shared directories

```bash
export MIND2WEB_DATA_DIR=/mnt/data/datasets/Mind2Web   # data -> /mnt/data/datasets/Mind2Web/raw
export MIND2WEB_MM_DIR=/mnt/data/datasets/Multimodal-Mind2Web   # screenshots
#   (omit this line to reuse the default $MIND2WEB_DATA_DIR/multimodal, which nests the
#    screenshots inside the Mind2Web folder — fine if you plan to keep them together)
export MIND2WEB_HF_CACHE=/mnt/data/hf-cache            # weights -> /mnt/data/hf-cache
```

Make them permanent so every new shell (and every `screen` session) agrees — add them to
`~/.bashrc`. To check what the current shell will use (do this before any multi-GB download):

```bash
uv run python -m mind2web_vqg.paths
# env var -> resolved directory
#   MIND2WEB_DATA_DIR    set (/mnt/data/datasets/Mind2Web)
#   ...
# directory            state     path
#   raw              exists    /mnt/data/datasets/Mind2Web/raw
#   multimodal       exists    /mnt/data/datasets/Mind2Web/multimodal
#   ...
```

> The download scripts and the readers import these *same* constants, so there is nothing to
> "register" afterwards: in one shell, download → export → build → eval all agree.

### Download the data (following the lab convention)

```bash
uv run m2w-download
# -> /mnt/data/datasets/Mind2Web/raw/data/{train,test_task,test_website,test_domain}
```

### Download weights (following the lab convention)

```bash
uv run m2w-download-model google/flan-t5-base
# -> /mnt/data/hf-cache/google_flan-t5-base

# equivalent hf CLI command:
hf download google/flan-t5-base --local-dir /mnt/data/hf-cache/google_flan-t5-base
```

After that, `hf/google/flan-t5-base` in the config automatically points to the downloaded folder.

### Running (remember to pick a GPU + use screen)

```bash
# general work -> card 1
CUDA_VISIBLE_DEVICES=1 uv run m2w-eval --config configs/default.yaml

# long jobs (training/heavy downloads) inside screen so they survive SSH drops
screen -S "<servername>_vqg_train"
CUDA_VISIBLE_DEVICES=1 uv run m2w-train --config configs/default.yaml
# detach from screen: CTRL+A then D ; reattach: screen -r "<servername>_vqg_train"
```

---

## 9. Important notes

- ⚠️ **Do NOT commit the Mind2Web data to GitHub.** The dataset license requires:
  > "Please DO NOT redistribute the unzipped data files online."
  The `data/` directory is gitignored.
- ⚠️ **Do NOT commit `.env`** (it holds API keys). Only commit `.env.example`.
- 🏷️ The dataset is licensed under **CC BY 4.0**; the Mind2Web code under **MIT**.
- 🧾 Cite Mind2Web if you use it in research (see section 11).

---

## 10. Roadmap

- [x] Define the concrete VQG task in `data/vqg.py` (self-supervised, screenshot-grounded).
- [x] Render HTML with Playwright (sanity check) and export real screenshots.
- [x] Screenshot -> VLM smoke test (Qwen3.5-9B, 6 simple probes).
- [x] Pick the question taxonomy (`target_label` / `element_type` / `target_region` / `field_value`).
- [ ] Generate the first real VQG set with `m2w-build-vqg` and eyeball it.
- [ ] Add evaluation metrics (BLEU / ROUGE / LLM-as-judge).
- [ ] API baseline + comparison against local HF models.
- [ ] A complete fine-tuning pipeline.
- [ ] Export `data/processed/` for easier sharing/management.

---

## 11. Citation

```bibtex
@misc{deng2023mind2web,
  title={Mind2Web: Towards a Generalist Agent for the Web},
  author={Deng, Xiang and Gu, Yu and Zheng, Boyuan and Chen, Shijie and Stevens, Samuel and Wang, Boshi and Sun, Huan and Su, Yu},
  year={2023},
  eprint={2306.06070},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
```
