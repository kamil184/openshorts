"""Text-only LLM backend for the moment picker: any OpenAI-compatible server.

Ollama, LM Studio, vLLM, llama.cpp's server, LocalAI and OpenRouter all speak
``POST {base}/chat/completions``. When ``LLM_BASE_URL`` is set (or
``LLM_PROVIDER=openai``), the transcript scoring and detail passes in
``main.get_viral_clips`` go here instead of Gemini, so a self-hosted install
can run the whole pipeline without a Google key.

What stays on Gemini, because it needs a model that can look at frames or a
video file: the layout picker (``layout_picker.py``), the on-screen content
detector (``screencast_layout.py``) and the silent-video path
(``main.get_visual_clips``). Without a Gemini key those degrade the way they
already did: the first two return "none", the third fails the job with a clear
message. Nothing in this module is imported by them.

Structured output: the prompts already spell out the exact JSON shape, so a
plain ``json_object`` mode is enough for most models. The request first asks
for ``json_schema`` (Ollama, vLLM, llama.cpp and LM Studio enforce it); a
server that rejects that field gets the same request again with
``json_object``, then with no ``response_format`` at all. Whatever comes back
is validated with the same pydantic model Gemini's ``response_schema`` uses,
so ``main.py`` sees one shape regardless of provider.
"""
from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Type, Union

import httpx
from pydantic import BaseModel

DEFAULT_MODEL = "llama3.1:8b"
DEFAULT_TIMEOUT = 600.0  # local models on CPU are slow; a scoring batch can take minutes


def provider() -> str:
    """``"openai"`` when a compatible endpoint is configured, else ``"gemini"``."""
    explicit = (os.environ.get("LLM_PROVIDER") or "").strip().lower()
    if explicit in ("openai", "ollama", "local", "openai-compatible"):
        return "openai"
    if explicit == "gemini":
        return "gemini"
    return "openai" if base_url() else "gemini"


def base_url() -> str:
    return (os.environ.get("LLM_BASE_URL") or "").strip().rstrip("/")


def model_name() -> str:
    return (os.environ.get("LLM_MODEL") or "").strip() or DEFAULT_MODEL


def active() -> bool:
    """True when the moment picker should call the OpenAI-compatible server."""
    return provider() == "openai" and bool(base_url())


def vision_model() -> str:
    """Model for the stages that look at frames (``LLM_VISION_MODEL``)."""
    return (os.environ.get("LLM_VISION_MODEL") or "").strip() or model_name()


def video_model() -> str:
    """Model for the silent-footage pass (``LLM_VIDEO_MODEL``)."""
    return (os.environ.get("LLM_VIDEO_MODEL") or "").strip() or vision_model()


def video_input_mode() -> str:
    """``"video"`` sends the proxy file itself; ``"frames"`` sends timestamped
    stills, for models that take images but not video (``LLM_VIDEO_INPUT``)."""
    mode = (os.environ.get("LLM_VIDEO_INPUT") or "video").strip().lower()
    if mode not in ("video", "frames"):
        raise ValueError("LLM_VIDEO_INPUT must be 'video' or 'frames'")
    return mode


@dataclass(frozen=True)
class Video:
    """An encoded video clip handed to a model that accepts video input."""
    data: bytes
    mime_type: str = "video/mp4"


MediaPart = Union[str, bytes, Video]


def _data_url(mime_type: str, data: bytes) -> str:
    return f"data:{mime_type};base64,{base64.b64encode(data).decode('ascii')}"


def _content(prompt: str, media: Optional[Sequence[MediaPart]]):
    """Chat message content: plain text, or ordered parts with the prompt last.

    ``str`` items are text, ``bytes`` are JPEG frames and ``Video`` is a clip,
    in the same order the Gemini callers pass their parts.
    """
    if not media:
        return prompt
    parts = []
    for item in media:
        if isinstance(item, Video):
            parts.append({"type": "video_url",
                          "video_url": {"url": _data_url(item.mime_type, item.data)}})
        elif isinstance(item, (bytes, bytearray)):
            parts.append({"type": "image_url",
                          "image_url": {"url": _data_url("image/jpeg", bytes(item))}})
        elif isinstance(item, str):
            parts.append({"type": "text", "text": item})
        else:
            raise TypeError(f"unsupported media part: {type(item).__name__}")
    parts.append({"type": "text", "text": prompt})
    return parts


def describe() -> Optional[dict]:
    """What ``/api/config`` tells the dashboard, or ``None`` when inactive."""
    if not active():
        return None
    return {"provider": "openai", "model": model_name(), "baseUrl": base_url()}


def _timeout() -> float:
    try:
        return float(os.environ.get("LLM_TIMEOUT") or DEFAULT_TIMEOUT)
    except ValueError:
        return DEFAULT_TIMEOUT


def _headers() -> dict:
    # Ollama ignores the key but the OpenAI client convention (and vLLM with
    # --api-key) wants the header present; "ollama" is the documented placeholder.
    key = (os.environ.get("LLM_API_KEY") or "ollama").strip()
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _extra_body() -> dict:
    """Provider-specific request fields from ``LLM_EXTRA_BODY`` (a JSON object).

    OpenRouter, for example, takes ``{"reasoning": {"effort": "low"},
    "provider": {"require_parameters": true}}``. A value that is not a JSON
    object is an operator error and fails loudly rather than being ignored.
    """
    raw = (os.environ.get("LLM_EXTRA_BODY") or "").strip()
    if not raw:
        return {}
    extra = json.loads(raw)
    if not isinstance(extra, dict):
        raise ValueError("LLM_EXTRA_BODY must be a JSON object")
    return extra


def _client(**kwargs) -> httpx.Client:
    """Factory so tests can swap in ``httpx.MockTransport``."""
    return httpx.Client(timeout=_timeout(), **kwargs)


def _response_formats(schema: Type[BaseModel]):
    name = getattr(schema, "__name__", "response").lower()
    yield {"type": "json_schema",
           "json_schema": {"name": name, "schema": schema.model_json_schema()}}
    yield {"type": "json_object"}
    yield None


def _is_unroutable_temperature(resp: httpx.Response) -> bool:
    """OpenRouter's 404 when ``provider.require_parameters`` leaves no endpoint
    that accepts every sent parameter (reasoning models reject temperature)."""
    return resp.status_code == 404 and "requested parameters" in resp.text.lower()


def _is_format_rejection(resp: httpx.Response) -> bool:
    if resp.status_code not in (400, 422):
        return False
    body = resp.text.lower()
    return "response_format" in body or "json_schema" in body or "json_object" in body \
        or "format" in body


def generate_json(prompt: str, schema: Type[BaseModel], model: Optional[str] = None,
                  media: Optional[Sequence[MediaPart]] = None,
                  ) -> Tuple[dict, Optional[dict]]:
    """One chat completion that must come back as JSON matching ``schema``.

    Returns ``(parsed_dict, cost_analysis)`` in the exact shape
    ``main._run_gemini_stage`` returns, so the caller does not branch on the
    provider. Raises on HTTP errors, empty bodies and schema violations; the
    retry policy lives in the caller, same as for Gemini.
    """
    import gemini_worker  # local import: keeps this module free of the google SDK

    url = f"{base_url()}/chat/completions"
    model = model or model_name()
    messages = [
        {"role": "system", "content": "You answer with a single JSON object and nothing else."},
        {"role": "user", "content": _content(prompt, media)},
    ]
    last_rejection: Optional[str] = None
    with _client() as client:
        for fmt in _response_formats(schema):
            body = {**_extra_body(), "model": model, "messages": messages,
                    "temperature": 0.2, "stream": False}
            if fmt is not None:
                body["response_format"] = fmt
            resp = client.post(url, json=body, headers=_headers())
            if _is_unroutable_temperature(resp):
                # Reasoning models take no temperature; a gateway told to
                # require every parameter then finds no endpoint at all.
                body.pop("temperature")
                resp = client.post(url, json=body, headers=_headers())
            if fmt is not None and _is_format_rejection(resp):
                # The server does not know this response_format flavour; the
                # next loop iteration asks for a looser one.
                last_rejection = resp.text[:200]
                continue
            if resp.status_code >= 400:
                raise RuntimeError(
                    f"LLM server {resp.status_code} from {url}: {resp.text[:300]}")
            data = resp.json()
            break
        else:
            raise RuntimeError(
                f"LLM server rejected every response_format variant: {last_rejection}")

    choices = data.get("choices") or []
    text = ""
    if choices:
        msg = choices[0].get("message") or {}
        text = msg.get("content") or ""
        if isinstance(text, list):  # some servers return content parts
            text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
    parsed = gemini_worker._parse_json_response_text(text)
    # Validate against the same schema Gemini enforces server-side, so a
    # local model that drops a field fails here with a readable error
    # (retried by the caller) instead of deep inside the clip pipeline.
    validated = schema.model_validate(parsed).model_dump()

    usage = data.get("usage") or {}
    # Gateways that bill per call (OpenRouter) report the charge in
    # ``usage.cost``; a local server reports none and stays at zero.
    billed = usage.get("cost")
    details = usage.get("completion_tokens_details") or {}
    cost = {
        "input_tokens": int(usage.get("prompt_tokens") or 0),
        "output_tokens": int(usage.get("completion_tokens") or 0),
        "thinking_tokens": int(details.get("reasoning_tokens") or 0),
        "input_cost": 0.0,
        "output_cost": 0.0,
        "total_cost": float(billed) if isinstance(billed, (int, float)) else 0.0,
        "model": model,
        "price_estimated": False,
        "local": not isinstance(billed, (int, float)),
    }
    return validated, cost
