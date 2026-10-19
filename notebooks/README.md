# `notebooks/` — Data exploration

A place for Jupyter notebooks used to **poke at and inspect the data**, for example:

- what a single Mind2Web instance looks like (how long is the HTML, how many actions)
- how `cleaned_html` differs from `raw_html`
- statistics over `website`, `domain`, and `confirmed_task` length
- trying prompts / trying a model on a few samples before running a full split eval

The folder currently only contains `.gitkeep`.

## Running notebooks

`jupyterlab` is **not** a project dependency. Run it ad hoc without touching `pyproject.toml`:

```bash
uv run --with jupyterlab jupyter lab
```

Or add it as a dev dependency permanently:

```bash
uv add --dev jupyterlab
uv run jupyter lab
```

## Starter snippet

```python
from mind2web_vqg.data.load import load_split

instances = load_split("test_task")
print(len(instances))

inst = instances[0]
print(inst["confirmed_task"])
print(inst["action_reprs"])

step = inst["actions"][0]
print(step["operation"])
print(len(step["cleaned_html"]))
```

## Conventions

- Name notebooks with a number + purpose: `01_explore_data.ipynb`, `02_prompt_playground.ipynb`.
- Don't commit large outputs (images, json); `*.ipynb_checkpoints/` is already ignored.
- Notebooks are for experiments only — once logic stabilizes, move it into `src/mind2web_vqg/`.
