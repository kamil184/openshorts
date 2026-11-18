"""The one door every model call goes through.

Callers say what they need, not which model or vendor answers:

    answer, cost = llm_backend.generate_json(prompt, Schema, task=VISION, media=frames)

``task`` picks the configured model for that kind of work, and the stages that
have a choice ask what that model can take (``supports(model, "video")``)
instead of checking its name.

Models per task, all optional:
  ``LLM_MODEL``         text stages (moment scoring and detail, pro-mode titles)
  ``LLM_VISION_MODEL``  stages that look at frames; defaults to ``LLM_MODEL``
  ``LLM_VIDEO_MODEL``   the silent-footage pass; defaults to ``LLM_VISION_MODEL``

Providers (``LLM_PROVIDER`` forces one; otherwise the first configured wins):
  ``openai``  any OpenAI-compatible server at ``LLM_BASE_URL`` (OpenRouter,
              Ollama, vLLM, LM Studio, llama.cpp), see ``llm_openai``
  ``gemini``  Google's API with ``GEMINI_API_KEY``, see ``llm_gemini``

Media parts are passed in order, the prompt goes last: ``str`` is text,
``bytes`` is a JPEG frame and ``Video`` is an encoded clip on disk. Every
answer is validated against the pydantic schema here, so a model that drops a
field fails with a readable error (retried by the caller) instead of deep
inside the pipeline.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Type, Union

from pydantic import BaseModel

import llm_gemini
import llm_openai

TEXT = "text"
VISION = "vision"
VIDEO = "video"

_PROVIDERS = {llm_openai.NAME: llm_openai, llm_gemini.NAME: llm_gemini}
_ALIASES = {"ollama": "openai", "local": "openai", "openai-compatible": "openai"}


@dataclass(frozen=True)
class Video:
    """An encoded clip on disk, for models that take video input."""
    path: str
    mime_type: str = "video/mp4"


MediaPart = Union[str, bytes, Video]


class BlockedError(ValueError):
    """The provider refused the request for content-policy reasons.

    Deterministic: the same payload is rejected every time, so callers fail
    fast (or bisect the batch) instead of retrying."""


def provider():
    """The configured provider module, or None when no model is configured."""
    explicit = (os.environ.get("LLM_PROVIDER") or "").strip().lower()
    explicit = _ALIASES.get(explicit, explicit)
    if explicit in _PROVIDERS:
        chosen = _PROVIDERS[explicit]
        return chosen if chosen.configured() else None
    for chosen in (llm_openai, llm_gemini):
        if chosen.configured():
            return chosen
    return None


def available() -> bool:
    return provider() is not None


def provider_name() -> Optional[str]:
    chosen = provider()
    return chosen.NAME if chosen else None


def model_for(task: str = TEXT) -> str:
    text = (os.environ.get("LLM_MODEL") or "").strip()
    if not text:
        chosen = provider()
        text = chosen.default_model() if chosen else llm_openai.DEFAULT_MODEL
    if task == TEXT:
        return text
    vision = (os.environ.get("LLM_VISION_MODEL") or "").strip() or text
    if task == VISION:
        return vision
    if task == VIDEO:
        return (os.environ.get("LLM_VIDEO_MODEL") or "").strip() or vision
    raise ValueError(f"unknown model task: {task!r}")


def input_modalities(model: str) -> frozenset:
    chosen = provider()
    return chosen.input_modalities(model) if chosen else frozenset()


def supports(model: str, modality: str) -> bool:
    """Whether ``model`` accepts ``modality`` ("image", "video", "audio") as input."""
    return modality in input_modalities(model)


def image_model(task: str = VISION) -> Optional[str]:
    """The configured model for ``task`` when it can look at images, else None
    (no model configured, or a text-only one): frame stages then skip."""
    if not available():
        return None
    model = model_for(task)
    return model if supports(model, "image") else None


def video_segment_seconds() -> float:
    """Longest stretch of video one call should carry for this provider."""
    chosen = provider() or llm_openai
    return chosen.VIDEO_SEGMENT_SECONDS


def score_batch_size() -> int:
    """Transcript windows per scoring call that suits this provider's models."""
    chosen = provider() or llm_openai
    return chosen.SCORE_BATCH


def describe() -> Optional[dict]:
    """What ``/api/config`` tells the dashboard about the configured models."""
    chosen = provider()
    if chosen is None:
        return None
    info = chosen.describe(model_for(TEXT))
    info["visionModel"] = model_for(VISION)
    info["videoModel"] = model_for(VIDEO)
    return info


def generate_json(prompt: str, schema: Type[BaseModel], *, task: str = TEXT,
                  model: Optional[str] = None,
                  media: Optional[Sequence[MediaPart]] = None,
                  gemini_key: Optional[str] = None,
                  ) -> Tuple[dict, Optional[dict]]:
    """One call that must come back as JSON matching ``schema``.

    Returns ``(validated_dict, cost_analysis)``. Raises ``BlockedError`` on a
    content-policy refusal and other exceptions on transport errors, empty
    bodies and schema violations; retry policy belongs to the caller.

    ``gemini_key`` is a caller's own Gemini key (the hosted edition's per-request
    keys); it is used only when the server configures no model of its own.
    """
    chosen = provider()
    if chosen is None and gemini_key:
        chosen = llm_gemini
        model = model or llm_gemini.default_model()
    if chosen is None:
        raise RuntimeError("No model configured: set LLM_BASE_URL (OpenAI-compatible "
                           "server) or GEMINI_API_KEY.")
    model = model or model_for(task)
    for item in media or []:
        needed = ("video" if isinstance(item, Video)
                  else "image" if isinstance(item, (bytes, bytearray)) else None)
        if needed and needed not in chosen.input_modalities(model):
            raise ValueError(f"{model} does not take {needed} input")
    parsed, cost = chosen.generate_json(prompt, schema, model, media,
                                        **({"api_key": gemini_key} if chosen is llm_gemini else {}))
    return schema.model_validate(parsed).model_dump(), cost
