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
│   └── processed/            # data preprocessed for the VQG task
│
├── hf-cache/                 # ⚠️ downloaded model weights (gitignored, see section 7)
│
├── src/mind2web_vqg/         # main code (Python package)
│   ├── __init__.py
│   ├── paths.py              # canonical project paths
│   ├── config.py             # YAML config loader
│   ├── cli.py                # CLI helpers (UTF-8 output...)
│   ├── data/
│   │   ├── download.py       # download Mind2Web (train + test.zip)
│   │   ├── load.py           # load splits into list[dict]
│   │   └── prepare.py        # 👉 define the VQG task HERE
│   ├── models/
│   │   ├── base.py           # QuestionGenerator interface
│   │   ├── api.py            # API baseline (OpenAI)
│   │   ├── hf.py             # local HuggingFace model
│   │   └── download.py       # download weights via the hf-cache convention
│   ├── prompts/
│   │   └── question_gen.json # prompt template
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
| [`src/mind2web_vqg/models/README.md`](src/mind2web_vqg/models/README.md) | Model interface, API/HF backends, `hf-cache` convention |
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

> On the lab server, change the target directory with an env var (see section 7):
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

## 5. Running evaluation

```bash
uv run m2w-eval --config configs/default.yaml
```

Results are written to `outputs/eval_<split>/predictions.json`.

Select the model in `configs/default.yaml`:

```yaml
model:
  name: "api/gpt-4o-mini"           # API baseline
  # name: "hf/google/flan-t5-base"  # local HF model (resolved from hf-cache if downloaded)
```

> Changing the task: edit `src/mind2web_vqg/data/prepare.py::build_examples` to define the
> input/target for VQG. This is the first place you should modify.

---

## 6. Fine-tuning (cloud / lab server)

The training skeleton lives in `src/mind2web_vqg/train.py` (still a TODO). Suggested workflow:

1. **Prepare locally:** download the data, write `build_examples` → export `data/processed/`.
2. **Move to the cloud/server:** Colab/Kaggle/rented GPU or the lab server (section 7);
   install `uv` (or pip) + `uv sync --extra train`.
3. **Train:** `uv run m2w-train --config configs/default.yaml`.
4. **Bring the model back locally** to re-evaluate it with `hf/...`.

---

## 7. Running on the lab server (AIVN-VLAI)

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
export MIND2WEB_HF_CACHE=/mnt/data/hf-cache            # weights -> /mnt/data/hf-cache
```

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

## 8. Important notes

- ⚠️ **Do NOT commit the Mind2Web data to GitHub.** The dataset license requires:
  > "Please DO NOT redistribute the unzipped data files online."
  The `data/` directory is gitignored.
- ⚠️ **Do NOT commit `.env`** (it holds API keys). Only commit `.env.example`.
- 🏷️ The dataset is licensed under **CC BY 4.0**; the Mind2Web code under **MIT**.
- 🧾 Cite Mind2Web if you use it in research (see section 10).

---

## 9. Roadmap

- [ ] Define the concrete VQG task in `data/prepare.py`.
- [ ] Add evaluation metrics (BLEU / ROUGE / LLM-as-judge).
- [ ] API baseline + comparison against local HF models.
- [ ] A complete fine-tuning pipeline.
- [ ] Export `data/processed/` for easier sharing/management.

---

## 10. Citation

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
