"""Local HuggingFace transformers models for inference and fine-tuning."""

from __future__ import annotations

from pathlib import Path

from mind2web_vqg.models.base import QuestionGenerator
from mind2web_vqg.paths import HF_CACHE_DIR, flatten_repo_id


def resolve_model_path(model_name: str) -> str:
    """Return the local path if the model was downloaded via the hf-cache convention.

    Example: ``google/flan-t5-base`` -> ``<HF_CACHE_DIR>/google_flan-t5-base`` when present.
    """
    if Path(model_name).exists():
        return model_name
    local = HF_CACHE_DIR / flatten_repo_id(model_name)
    return str(local) if local.exists() else model_name


class HFModel(QuestionGenerator):
    """Wrapper for seq2seq / causal HuggingFace models.

    Install: ``uv sync --extra local`` (or ``--extra train`` for fine-tuning).

    ``model_name`` can be a repo id (``google/flan-t5-base``) or a path to a
    downloaded directory (see ``m2w-download-model``).
    """

    def __init__(
        self,
        model_name: str,
        *,
        device: str | None = None,
        max_new_tokens: int = 256,
        load_in_8bit: bool = False,
    ) -> None:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        resolved = resolve_model_path(model_name)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(resolved)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            resolved,
            load_in_8bit=load_in_8bit or None,
        )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.max_new_tokens = max_new_tokens

    def generate(self, prompt: str, image: str | Path | None = None) -> str:
        if image is not None:
            raise ValueError("HFModel is text-only; use `vlm/<repo_id>` for screenshots.")
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        return self.tokenizer.decode(out[0], skip_special_tokens=True).strip()
