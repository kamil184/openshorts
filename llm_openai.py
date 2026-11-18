"""OpenAI-compatible provider: any server that speaks ``POST {base}/chat/completions``.

Ollama, LM Studio, vLLM, llama.cpp's server, LocalAI and OpenRouter all do.
Images travel as ``image_url`` parts and video as a ``video_url`` part, both
inline as base64 data URLs, so nothing is uploaded ahead of the call.

Structured output: the request first asks for ``json_schema`` (Ollama, vLLM,
llama.cpp, LM Studio and OpenRouter enforce it); a server that rejects that
field gets the same request again with ``json_object``, then with no
``response_format`` at all. The caller validates the answer against the
pydantic schema either way.

What a model can take as input comes from the server's ``GET {base}/models``
(``architecture.input_modalities``, as OpenRouter publishes it). A server that
does not say is treated as text-only unless ``LLM_INPUT_MODALITIES`` declares
the model, so a vision stage never sends frames to a model that cannot see.
"""
from __future__ import annotations

import base64
import json
import os
from typing import Optional, Sequence, Tuple, Type

import httpx
from pydantic import BaseModel

import prompts

NAME = "openai"
DEFAULT_MODEL = "llama3.1:8b"
DEFAULT_TIMEOUT = 600.0  # local models on CPU are slow; a scoring batch can take minutes
# Inline video rides in the request body, so stretches stay well under the
# gateways' body limits (a 360p/1 fps proxy is ~1 MB per minute).
VIDEO_SEGMENT_SECONDS = 600.0
# Small local models lose track of long prompts: fewer windows per scoring call.
SCORE_BATCH = 3


def base_url() -> str:
    return (os.environ.get("LLM_BASE_URL") or "").strip().rstrip("/")


def configured() -> bool:
    return bool(base_url())


def default_model() -> str:
    return DEFAULT_MODEL


def describe(model: str) -> dict:
    return {"provider": NAME, "model": model, "baseUrl": base_url()}


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


# --- capabilities -------------------------------------------------------------

_modalities_cache: Optional[dict] = None


def _declared_modalities() -> dict:
    """``LLM_INPUT_MODALITIES``: ``{"model": ["text", "image"], ...}``, for
    servers whose model list does not say what each model accepts."""
    raw = (os.environ.get("LLM_INPUT_MODALITIES") or "").strip()
    if not raw:
        return {}
    declared = json.loads(raw)
    if not isinstance(declared, dict):
        raise ValueError("LLM_INPUT_MODALITIES must be a JSON object")
    for model, modalities in declared.items():
        if not isinstance(modalities, list) or not all(isinstance(m, str) for m in modalities):
            raise ValueError(f"LLM_INPUT_MODALITIES[{model!r}] must be a list of strings")
    return {str(k): frozenset(v) for k, v in declared.items()}


def _listed_modalities() -> dict:
    """``{model id: input modalities}`` from the server's model list, fetched
    once per process; empty when the server does not publish modalities."""
    global _modalities_cache
    if _modalities_cache is None:
        listed = {}
        try:
            with _client() as client:
                resp = client.get(f"{base_url()}/models", headers=_headers(), timeout=30.0)
            if resp.status_code < 400:
                for entry in resp.json().get("data") or []:
                    mods = (entry.get("architecture") or {}).get("input_modalities")
                    if entry.get("id") and mods:
                        listed[entry["id"]] = frozenset(map(str, mods))
        except (httpx.HTTPError, ValueError) as e:
            print(f"   ℹ️ Could not read the model list from {base_url()}: {e}")
        _modalities_cache = listed
    return _modalities_cache


def input_modalities(model: str) -> frozenset:
    declared = _declared_modalities()
    if model in declared:
        return declared[model] | {"text"}
    return _listed_modalities().get(model, frozenset()) | {"text"}


# --- generation ---------------------------------------------------------------

def _data_url(mime_type: str, data: bytes) -> str:
    return f"data:{mime_type};base64,{base64.b64encode(data).decode('ascii')}"


def _content(prompt: str, media):
    """Chat message content: plain text, or ordered parts with the prompt last."""
    import llm_backend

    if not media:
        return prompt
    parts = []
    for item in media:
        if isinstance(item, llm_backend.Video):
            with open(item.path, "rb") as f:
                url = _data_url(item.mime_type, f.read())
            parts.append({"type": "video_url", "video_url": {"url": url}})
        elif isinstance(item, (bytes, bytearray)):
            parts.append({"type": "image_url",
                          "image_url": {"url": _data_url("image/jpeg", bytes(item))}})
        elif isinstance(item, str):
            parts.append({"type": "text", "text": item})
        else:
            raise TypeError(f"unsupported media part: {type(item).__name__}")
    parts.append({"type": "text", "text": prompt})
    return parts


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


def generate_json(prompt: str, schema: Type[BaseModel], model: str,
                  media: Optional[Sequence] = None) -> Tuple[dict, dict]:
    """One chat completion; returns ``(unvalidated_dict, cost_analysis)``."""
    url = f"{base_url()}/chat/completions"
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
        from llm_backend import BlockedError

        msg = choices[0].get("message") or {}
        if choices[0].get("finish_reason") == "content_filter" or msg.get("refusal"):
            raise BlockedError("The model provider refused the request for content-policy reasons.")
        text = msg.get("content") or ""
        if isinstance(text, list):  # some servers return content parts
            text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
    parsed = prompts.parse_json_text(text)

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
    return parsed, cost
