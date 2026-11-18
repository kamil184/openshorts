"""Gemini provider (Google AI Studio key in ``GEMINI_API_KEY``).

Images go inline; a video is uploaded to the Files API, waited on until it is
ACTIVE and deleted after the call. Every Gemini model takes text, images,
video and audio, so capabilities need no lookup.
"""
from __future__ import annotations

import mimetypes
import os
import time
from typing import Optional, Sequence, Tuple, Type

from pydantic import BaseModel

import prompts
from clip_selection import lookup_model_prices

NAME = "gemini"
DEFAULT_MODEL = "gemini-3.1-flash-lite"
# Uploaded files may be long; past this a silent source is watched in stretches.
VIDEO_SEGMENT_SECONDS = 2400.0
SCORE_BATCH = 8  # 1M-token context
INPUT_MODALITIES = frozenset({"text", "image", "video", "audio"})
UPLOAD_TIMEOUT_SECONDS = 180


def configured_key() -> str:
    return (os.environ.get("GEMINI_API_KEY") or "").strip()


def configured() -> bool:
    return bool(configured_key())


def default_model() -> str:
    return (os.environ.get("GEMINI_MODEL") or "").strip() or DEFAULT_MODEL


def describe(model: str) -> dict:
    return {"provider": NAME, "model": model}


def input_modalities(model: str) -> frozenset:
    return INPUT_MODALITIES


def upload_media(client, path, mime_type=None):
    """Upload a local file to the Gemini Files API, by handle and never by path.

    Handed a path, the SDK copies ``os.path.basename(path)`` verbatim into the
    ``X-Goog-Upload-File-Name`` header, and httpx encodes header values as
    ASCII: every video whose title is written in Japanese, Cyrillic, Arabic or
    Greek died before a byte left the container with ``'ascii' codec can't
    encode characters``. Passing an open handle skips that header entirely;
    the readable name still travels as ``display_name``, which goes in the
    JSON body and is UTF-8 all the way.
    """
    guessed = mime_type or mimetypes.guess_type(path)[0] or ""
    # Every caller uploads the source video; an extension the stdlib does not
    # know (or knows as octet-stream) must not reach the API as a type it
    # refuses, so fall back to the container the pipeline always writes.
    if not guessed.startswith(("video/", "audio/", "image/")):
        guessed = "video/mp4"
    with open(path, "rb") as fh:
        return client.files.upload(
            file=fh,
            config={"mime_type": guessed,
                    "display_name": os.path.basename(path)},
        )


def _wait_active(client, upload):
    deadline = time.time() + UPLOAD_TIMEOUT_SECONDS
    while True:
        info = client.files.get(name=upload.name)
        state = str(getattr(getattr(info, "state", info), "name", "")).upper()
        if state == "ACTIVE":
            return
        if state == "FAILED":
            raise RuntimeError("Gemini could not process the uploaded video")
        if time.time() > deadline:
            raise RuntimeError("Gemini video processing timed out")
        time.sleep(2)


_BLOCKED_FINISH_REASONS = {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST",
                           "SPII", "IMAGE_SAFETY", "RECITATION"}


def raise_if_blocked(response):
    """Raise ``llm_backend.BlockedError`` when the API refused on policy grounds."""
    from llm_backend import BlockedError

    pf = getattr(response, "prompt_feedback", None)
    reason = getattr(pf, "block_reason", None)
    if reason:
        name = getattr(reason, "name", None) or str(reason)
        raise BlockedError(
            f"The model provider blocked this video's content ({name}). Its usage "
            "policies reject this material, so it can't be analyzed.")
    for c in (getattr(response, "candidates", None) or []):
        fr = getattr(c, "finish_reason", None)
        name = (getattr(fr, "name", None) or str(fr or "")).upper()
        if name in _BLOCKED_FINISH_REASONS:
            raise BlockedError(
                f"The model provider blocked its answer for this video ({name}). Its "
                "usage policies reject this material, so it can't be analyzed.")


def response_text(response) -> str:
    try:
        text = response.text
        if text:
            return text
    except Exception:
        pass
    parts = []
    for candidate in getattr(response, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", []) or []:
            part_text = getattr(part, "text", None)
            if part_text:
                parts.append(part_text)
    return "\n".join(parts).strip()


def cost_analysis(response, model_name: str) -> Optional[dict]:
    usage = getattr(response, "usage_metadata", None)
    if not usage:
        return None
    prices = lookup_model_prices(model_name)
    price_estimated = prices is None
    if prices is None:
        # Unknown model: conservative estimate so the UI shows something sane.
        prices = (0.50, 3.00)
    input_price_per_million, output_price_per_million = prices
    prompt_tokens = usage.prompt_token_count or 0
    output_tokens = usage.candidates_token_count or 0
    # Thinking tokens bill at the output rate even though they are invisible.
    thinking_tokens = getattr(usage, "thoughts_token_count", 0) or 0
    input_cost = (prompt_tokens / 1_000_000) * input_price_per_million
    output_cost = ((output_tokens + thinking_tokens) / 1_000_000) * output_price_per_million
    return {
        "input_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "thinking_tokens": thinking_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + output_cost,
        "model": model_name,
        "price_estimated": price_estimated,
    }


def generate_json(prompt: str, schema: Type[BaseModel], model: str,
                  media: Optional[Sequence] = None,
                  api_key: Optional[str] = None) -> Tuple[dict, Optional[dict]]:
    """One schema-enforced call; returns ``(unvalidated_dict, cost_analysis)``."""
    from google import genai
    from google.genai import types as genai_types

    import llm_backend

    # Keep the client in a name: one built inline is closed before its call
    # is sent ("client has been closed").
    client = genai.Client(api_key=api_key or configured_key())
    uploads = []
    try:
        parts = []
        for item in media or []:
            if isinstance(item, llm_backend.Video):
                upload = upload_media(client, item.path, item.mime_type)
                uploads.append(upload)
                _wait_active(client, upload)
                parts.append(upload)
            elif isinstance(item, (bytes, bytearray)):
                parts.append(genai_types.Part.from_bytes(data=bytes(item),
                                                         mime_type="image/jpeg"))
            elif isinstance(item, str):
                parts.append(item)
            else:
                raise TypeError(f"unsupported media part: {type(item).__name__}")
        response = client.models.generate_content(
            model=model,
            contents=parts + [prompt],
            config=genai_types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
            ))
        raise_if_blocked(response)
        parsed_obj = getattr(response, "parsed", None)
        if parsed_obj is not None:
            parsed = parsed_obj.model_dump() if hasattr(parsed_obj, "model_dump") else parsed_obj
        else:
            # Gemini sometimes returns 200 with an empty body; this raises and
            # the caller's retry recovers it.
            parsed = prompts.parse_json_text(response_text(response))
        return parsed, cost_analysis(response, model)
    finally:
        for upload in uploads:
            try:
                client.files.delete(name=upload.name)
            except Exception:
                pass
