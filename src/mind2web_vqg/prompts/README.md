# `src/mind2web_vqg/prompts/` — Prompt templates

Holds the prompts used for the question generation (VQG) task.

## Files

| File | Contents |
| --- | --- |
| `question_gen.json` | `system` prompt + a `templates` list (one template named `default`) |
| `__init__.py` | `DEFAULT_TEMPLATE` (Python string) and `load_prompt(name)` |

`question_gen.json`:

```json
{
  "system": "You are a helpful assistant that asks concise clarifying questions about web pages.",
  "templates": [
    {
      "name": "default",
      "template": "Task: {task}\n\nPage (cleaned HTML):\n{html}\n\nGenerate one clarifying question about this page to help complete the task.\nQuestion:"
    }
  ]
}
```

## Usage

```python
from mind2web_vqg.prompts import DEFAULT_TEMPLATE, load_prompt

prompts = load_prompt()                       # dict from question_gen.json
system = prompts["system"]
tpl = prompts["templates"][0]["template"]

text = tpl.format(task="Find a red shirt", html="<html>...</html>")
```

## ⚠️ Current state

- `load_prompt()` and `DEFAULT_TEMPLATE` are **not called by `evaluate.py` yet**.
  `evaluate.py` feeds `example["input"]` straight into the model.
- `configs/default.yaml` has `prompt.file` / `prompt.template` but nothing reads them.
- So: to make the prompt take effect you must wire it into `evaluate.py` (or format it inside
  `prepare.py::build_examples` — the simpler route, and the direction the design is heading).

## Prompt-writing tips

- HTML is long → trim it, or keep only text/important elements before inserting it.
- Keep placeholders `{task}` / `{html}` in sync with the `.format(...)` call.
- Add a new template by appending to the `templates` array, then select it by `name`.
- The prompt drives baseline quality a lot — experiment in `notebooks/` first.
