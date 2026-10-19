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
  "system": "You ask short questions about a web page screenshot whose answers are visible on the screen.",
  "templates": [
    {
      "name": "default",
      "template": "You are shown a screenshot of a web page. Ask exactly one question about the page whose answer can be read from the screenshot, focusing on {focus}.\nThe user's task is: {task}\nOutput only the question."
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

text = tpl.format(focus="where the element is", task="Find a red shirt")
```

## ⚠️ Current state

- The **live** prompt for every built example comes from
  `data/vqg.py::build_input` (instruction + the user's task); `DEFAULT_TEMPLATE` here is a
  plain-string mirror of that instruction, for notebook experiments.
- `load_prompt()` / `question_gen.json` are **not read by `evaluate.py`**.
  `evaluate.py` feeds `example["input"]` (built by `build_examples` → `vqg.build_input`)
  straight into the model.
- `configs/default.yaml` has `prompt.file` / `prompt.template` but nothing reads them.

## Where are the screenshot prompts?

- The **task** prompts are produced by `data/vqg.py` (`INSTRUCTION` + `CATEGORY_HINTS`),
  one per category (`target_label` / `element_type` / `target_region` / `field_value`).
- The **smoke-test** probes live in `smoke_vlm.py::PROBES` — throwaway probes (one per
  classic VQA type) used to check the vision pipeline:

| Probe | Type | Example question |
| --- | --- | --- |
| `what_heading` | what | "What is the main heading of this web page?" |
| `descriptive` | descriptive | "In one sentence, what can a user do on this page?" |
| `yes_no` | yes/no | "Is a search box visible? Answer yes or no." |
| `count` | count | "How many items are in the top navigation bar?" |
| `how_next` | how | "The task is {task}. What is the next action and on which element?" |
| `where_target` | where | "The task is {task}. Where is the element they should use next?" |

## Prompt-writing tips

- The model needs to *see* the page: keep prompts image-first, and avoid asking for things
  that are not visible on the screen.
- Keep placeholders `{focus}` / `{task}` in sync with the `.format(...)` call.
- Add a new template by appending to the `templates` array, then select it by `name`.
- The prompt drives baseline quality a lot — experiment in `notebooks/` first.
