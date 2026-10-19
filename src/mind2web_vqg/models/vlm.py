"""Local vision-language models (Qwen2.5-VL / Qwen3-VL / Qwen3.5).

Install::

    uv sync --extra vlm

Then either point at a downloaded folder (lab convention) or a Hub repo id::

    model = VLMModel("Qwen/Qwen3.5-9B")
    model = VLMModel("Qwen/Qwen3.5-9B")   # found in $MIND2WEB_HF_CACHE if downloaded

Notes
-----
* ``transformers>=5.20`` is required for Qwen3.5; older Qwen-VL families work with
  ``transformers>=4.49``.
* Qwen3.5 thinks by default. ``enable_thinking=False`` asks the chat template for a
  direct answer, which is what we want for short VQA answers.
* Do **not** use ``flash-attn`` (lab rule); the default SDPA attention is fine.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from mind2web_vqg.models.base import QuestionGenerator
from mind2web_vqg.models.hf import resolve_model_path

_THINK_RE = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.DOTALL | re.IGNORECASE)


def strip_thinking(text: str) -> str:
    """Drop a leading ``<think>...</think>`` block, if the model emitted one."""
    return _THINK_RE.sub("", text).strip()


class VLMModel(QuestionGenerator):
    """Wrapper for ``*ForConditionalGeneration`` image-text models."""

    def __init__(
        self,
        model_name: str,
        *,
        device_map: str | None = "auto",
        max_new_tokens: int = 256,
        enable_thinking: bool = False,
        load_in_4bit: bool = False,
        max_pixels: int | None = None,
        min_pixels: int | None = None,
    ) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        resolved = resolve_model_path(model_name)
        self.torch = torch
        self.processor = self._load_processor(
            AutoProcessor, resolved, min_pixels=min_pixels, max_pixels=max_pixels
        )
        self.model = self._load_model(
            AutoModelForImageTextToText, resolved, device_map=device_map, load_in_4bit=load_in_4bit
        )
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.enable_thinking = enable_thinking

    # -- loading ---------------------------------------------------------- #
    @staticmethod
    def _load_processor(cls: Any, path: str, **pixel_bounds: int | None) -> Any:
        bounds = {k: v for k, v in pixel_bounds.items() if v is not None}
        try:
            return cls.from_pretrained(path, **bounds)
        except (TypeError, ValueError):
            # Some processors don't accept min/max_pixels (e.g. Qwen3.5 uses edges).
            return cls.from_pretrained(path)

    @staticmethod
    def _load_model(
        cls: Any,
        path: str,
        *,
        device_map: str | None,
        load_in_4bit: bool,
    ) -> Any:
        import torch

        kwargs: dict[str, Any] = {"device_map": device_map}
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        try:
            return cls.from_pretrained(path, dtype=dtype, **kwargs)
        except TypeError:  # transformers < 5 spells it torch_dtype
            return cls.from_pretrained(path, torch_dtype=dtype, **kwargs)

    # -- inference -------------------------------------------------------- #
    def _build_messages(self, prompt: str, image: str | Path | None) -> list[dict[str, Any]]:
        from PIL import Image

        content: list[dict[str, Any]] = []
        if image is not None:
            content.append({"type": "image", "image": Image.open(image).convert("RGB")})
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    def generate(self, prompt: str, image: str | Path | None = None) -> str:
        messages = self._build_messages(prompt, image)
        try:
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        except TypeError:
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        images = [c["image"] for c in messages[0]["content"] if c["type"] == "image"]
        inputs = self.processor(text=[text], images=images or None, return_tensors="pt")
        inputs = inputs.to(self._input_device())

        with self.torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        new_tokens = out[0][inputs["input_ids"].shape[-1] :]
        answer = self.processor.batch_decode([new_tokens], skip_special_tokens=True)[0]
        return strip_thinking(answer)

    def _input_device(self) -> Any:
        """Device to move inputs to (works with both plain models and ``device_map``)."""
        device = getattr(self.model, "device", None)
        if device is not None and str(device) != "meta":
            return device
        return next(self.model.parameters()).device
