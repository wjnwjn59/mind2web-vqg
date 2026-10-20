# `tests/` — Testing

Uses `pytest` (a dev dependency, installed by `uv sync`).

## Running

```bash
uv run pytest                 # everything
uv run pytest -v              # verbose
uv run pytest tests/test_smoke.py::test_config_loads
```

## Current files

`test_smoke.py` — minimal checks that the project is still healthy:

| Test | Checks |
| --- | --- |
| `test_version` | `mind2web_vqg.__version__` is non-empty |
| `test_config_loads` | `configs/default.yaml` loads and has `model.name` |
| `test_project_root` | `PROJECT_ROOT/pyproject.toml` exists |
| `test_paths_main_reports_resolved_dirs` | `python -m mind2web_vqg.paths` prints the env vars + resolved dirs |

`test_multimodal.py` — pure helpers of the screenshot pipeline (no parquet, no PIL needed):

| Test | Checks |
| --- | --- |
| `test_as_json_handles_strings_and_passthrough` | JSON-string columns decode, bad JSON falls back |
| `test_parse_bbox` | `"x,y,w,h"` -> floats; malformed input -> `None` |
| `test_target_element_prefers_original_target` | picks the `is_original_target` candidate |
| `test_from_row_parses_the_real_schema` | row -> example dict (task, op, bbox, history) |
| `test_image_name_is_filesystem_safe` | ids with `/` or spaces become safe file names |
| `test_probe_task_substitution` | `{task}` is filled in the smoke probes |

These are smoke tests: fast, no data, no API calls, no GPU.

`test_download_multimodal.py` — the download command's pure helpers:

| Test | Checks |
| --- | --- |
| `test_patterns_for` | split -> `data/<split>-*.parquet` globs |
| `test_hint_when_default_nests_inside_mind2web` | warns when `MIND2WEB_MM_DIR` is unset on the lab server |
| `test_no_hint_once_destination_has_data` | no nag on resume runs once files are present |
| `test_dest_has_data_detects_parquet` | detects an in-use destination |
| `test_no_hint_when_mm_dir_is_explicit` / `test_no_hint_for_local_default` | no false alarms |

`test_vqg.py` — the self-supervised VQG task (`data/vqg.py`), pure and image-free:

| Test | Checks |
| --- | --- |
| `test_role_from_tag` / `test_role_refines_input_by_type` | tag (and input `type`) -> human element word |
| `test_role_none_for_unknown_or_missing_tag` | unmapped tags return `None` | 
| `test_region_thirds` / `test_region_rescales_from_source_to_image` | bounding box -> coarse region |
| `test_region_none_when_coordinates_are_off_page` | bad/off-page boxes are dropped |
| `test_default_categories_are_screenshot_grounded` | defaults exclude the task-answered category |
| `test_questions_and_answers` | the derived question/answer text |
| `test_field_value_is_opt_in_and_answered_from_task` | `field_value` needs opting in |
| `test_categories_skipped_when_answer_missing` | each category drops only when its own answer is missing |
| `test_build_examples_ids_and_prompt` / `test_build_examples_can_omit_task` | ids, prompt, `--no-task` |
| `test_category_stats` / `test_write_examples_round_trip` | counts + `examples.jsonl` round trip |
| `test_action_repr_label` | `"[link]  Budget Truck -> CLICK"` -> `"Budget Truck"`; non-repr shape -> `""` |
| `test_label_falls_back_to_the_action_repr` | no attribute text -> the action repr supplies the label |
| `test_type_and_region_survive_a_missing_label` | `element_type` / `target_region` no longer require a label |

`test_metrics.py` — question-quality metrics (`metrics.py`):

| Test | Checks |
| --- | --- |
| `test_normalize` / `test_exact_match_and_containment` / `test_token_f1` | answer comparison for the round-trip |
| `test_rouge_l` / `test_chrf` | overlap with the reference question |
| `test_distinct_and_unique` / `test_self_bleu_detects_collapse` | diversity, i.e. mode collapse |
| `test_is_well_formed` | format validity |
| `test_compute_metrics_*` | overall + per-category aggregation, empty predictions, empty run |
| `test_round_trip_*` | stub-model round-trip, image plumbing, and the reference ceiling |

`test_report.py` — the readable report (`report.py`):

| Test | Checks |
| --- | --- |
| `test_markdown_has_summary_metrics_and_samples` | summary line, per-category table, example rows |
| `test_markdown_respects_the_limit` | `--limit` caps the sample table |
| `test_markdown_escapes_table_pipes` | `\|` inside a question cannot break the table |
| `test_html_has_images_and_escapes_markup` | thumbnails wired up, HTML escaped |
| `test_write_report_creates_both_files` | `report.md` + `report.html` land in the output dir |
| `test_reports_handle_an_empty_run` | an empty run still produces a valid report |

`test_eda.py` — corpus EDA over the HTML tree (`eda.py`):

| Test | Checks |
| --- | --- |
| `test_parse_html_counts_tags_and_skips_scripts` | tag counts, `script` subtree ignored |
| `test_parse_html_detects_labels` | text / `aria-label` / `alt` / `placeholder` / `value` |
| `test_parse_html_collects_input_types` | `<input type=...>` histogram (defaults to `text`) |
| `test_parse_html_tolerates_unclosed_tags` | malformed dumps do not crash the run |
| `test_recorded_target_*` | `is_original_target` pick, first-candidate fallback, empty |
| `test_summarize_counts_elements_and_targets` | tasks/steps/elements and target rates |
| `test_summarize_computes_the_target_lift` | `lift` = target share / tree share |
| `test_summarize_estimates_vqg_potential` | label / type / region question capacity |
| `test_summarize_detects_repeated_page_states` | identical step HTML is counted once |
| `test_render_markdown_has_the_main_sections` / `test_write_report_*` | report + `tags.csv` |
| `test_merge_of_accumulators_matches_a_single_pass` | merged splits equal one pass over both |
| `test_merge_keeps_repeated_page_states_exact` | digest union, not summed counts |
| `test_available_splits_lists_only_existing_directories` | whole-corpus mode skips splits that weren't downloaded |
| `test_render_summary_has_one_row_per_split_plus_all` | cross-split table with an `all` row |
| `test_main_all_splits_writes_a_summary` | `--all-splits` end to end (per-split + merged + summary) |

`test_download.py` — the JSON download helpers (`data/download.py`, `data/load.py`); no network:

| Test | Checks |
| --- | --- |
| `test_unzip_moves_the_test_splits_under_data` | `test.zip` root layout ends as `raw/data/<split>` |
| `test_unzip_keeps_an_archive_that_already_has_the_data_prefix` | an archive with `data/…` is left alone |
| `test_unzip_explains_a_corrupt_archive` | `BadZipFile` becomes a `RuntimeError` naming `--force` |
| `test_normalise_*` | idempotent, repairs a broken file, drops the loose shell, creates `data/` |
| `test_verify_accepts_a_healthy_layout` | per-split file/row counts, exit 0 |
| `test_verify_names_a_truncated_file` | prints `BROKEN <file>` and exits 1 |
| `test_verify_reports_a_missing_split_without_failing` | a split that was not downloaded is not an error |
| `test_verify_counts_a_wrapped_instance_list` | `{"data": [...]}` counts its rows |
| `test_iter_split_reads_instances` | streaming load of one split |
| `test_iter_split_names_a_truncated_file` | `ValueError` names the path and suggests `--verify` |
| `test_iter_split_explains_a_missing_split` | `FileNotFoundError` points at `m2w-download` |

`test_eda_report.py` — the LaTeX report (`eda_report.py`); no TeX distribution required:
| Test | Checks |
| --- | --- |
| `test_tex_escapes_every_special_character` / `test_tex_does_not_double_escape_*` | LaTeX escaping is single-pass (`\_`, `\%`, `\textbackslash{}`) |
| `test_number_formatting` | thousands separators, `84.0` -> `84`, escaped `%` |
| `test_concentration_of_an_even_pool` / `..._of_a_dominated_pool` | HHI, effective type count, types for 50%/80% |
| `test_policy_*` | `share x N` vs `min(labelled, N/n)`, and the shortfall when the corpus is small |
| `test_find_reports_separates_the_merged_report` | `all/` is returned separately from the splits |
| `test_find_reports_explains_a_missing_directory` | the error names `m2w-eda` |
| `test_render_tex_contains_every_section` / `..._has_valid_environments` | all sections, balanced `\begin`/`\end` |
| `test_render_tex_has_tables_without_figures` | no `\includegraphics` when no figure mapping is passed |
| `test_render_tex_embeds_a_chart_per_section` | one `\includegraphics` per figure, and only for the named sections |
| `test_render_tex_with_several_splits_adds_the_drift_section` | small multiples only with 2+ splits |
| `test_build_figures_writes_png_files` | six real PNGs (magic bytes) at the paths the `.tex` expects |
| `test_build_figures_without_targets_still_draws_the_tree` | a chart is skipped, not faked, when it has no data |
| `test_main_writes_the_report` / `test_main_renders_figures` / `test_main_reports_an_unknown_split` | CLI end to end, with and without charts |

The chart tests call `pytest.importorskip("matplotlib")`, so a checkout without the `report`
extra still runs the suite (it just skips them).

`test_model_download.py` — the weights-download helpers (`models/download.py`):

| Test | Checks |
| --- | --- |
| `test_flatten_repo_id` | `Qwen/Qwen3.5-9B` -> `Qwen_Qwen3.5-9B` |
| `test_target_dir_uses_namespace_repo_name` | `<cache>/<ns>_<repo>` layout |
| `test_hint_when_env_is_unset` | warns when weights would land in the repo |
| `test_no_hint_when_env_is_set` | no warning when `MIND2WEB_HF_CACHE` is set |

## What's worth adding

- `load_split` against a fake directory (pytest `tmp_path`) with sample JSON → verify file
  concatenation, dict-wrapped lists, and the error for an unknown split.
- `load_exported` against a temporary `examples.jsonl` (no export attempt when the file exists).
- `models.build_model` → unknown prefix raises `ValueError`; an `api/` prefix returns the
  right class (mocked, no API key needed).
- `resolve_model_path` → a repo id maps to `<ns>_<repo>` when that folder exists.

## Conventions

- No test may hit the network, need an API key, or require the real dataset.
- Use pytest's `tmp_path` fixture instead of creating files inside the repo.
- File names `test_*.py`, function names `test_*`.
- Lint alongside: `uv run ruff check src tests`.
