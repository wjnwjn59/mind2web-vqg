"""Question generator model classes."""

from mind2web_vqg.models.base import QuestionGenerator

__all__ = ["QuestionGenerator", "build_model"]


def build_model(name: str, **kwargs):
    """Simple factory: ``build_model("api/gpt-4o-mini")`` or ``build_model("hf/...")``.

    The prefix selects the backend:
      - ``api/...`` -> APIModel
      - ``hf/...``  -> HFModel (text only)
      - ``vlm/...`` -> VLMModel (screenshots, local transformers)
    """
    if name.startswith("api/"):
        from mind2web_vqg.models.api import APIModel

        return APIModel(name.removeprefix("api/"), **kwargs)
    if name.startswith("hf/"):
        from mind2web_vqg.models.hf import HFModel

        return HFModel(name.removeprefix("hf/"), **kwargs)
    if name.startswith("vlm/"):
        from mind2web_vqg.models.vlm import VLMModel

        return VLMModel(name.removeprefix("vlm/"), **kwargs)
    raise ValueError(
        f"Unknown backend for model {name!r}. Use the 'api/', 'hf/' or 'vlm/' prefix."
    )
