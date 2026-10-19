"""Preprocessing: turn Mind2Web instances into question-generation examples.

THIS IS WHERE YOU DEFINE YOUR VQG TASK.

This is only a skeleton: ``build_examples`` returns a list of ``{input, target}``
dicts with a placeholder definition. Replace it with your actual task definition.
"""

from __future__ import annotations

from typing import Any

#: Example fields of a Mind2Web instance (for reference when writing the logic):
#:   annotation_id, website, domain, subdomain, confirmed_task,
#:   action_reprs (list[str]), actions (list[dict])
#: Each action has: action_uid, raw_html, cleaned_html, operation{op,value},
#:   pos_candidates (list), neg_candidates (list)


def build_examples(instances: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert raw instances into ``(input, target)`` VQG examples.

    TODO: define exactly what the input is (HTML? cleaned_html? the task
    description?) and what the target question should be.
    """
    examples: list[dict[str, Any]] = []
    for inst in instances:
        examples.append(
            {
                # TODO: replace with the real context (cleaned_html, confirmed_task, ...)
                "id": inst.get("annotation_id"),
                "input": inst.get("confirmed_task", ""),
                # TODO: replace with the expected question
                "target": "",
                "meta": {
                    "website": inst.get("website"),
                    "domain": inst.get("domain"),
                },
            }
        )
    return examples
