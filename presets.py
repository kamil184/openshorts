"""Campaign presets: YAML loading, job-default expansion, prompt rules,
and deterministic post-caption enforcement.

Presets live outside the repository in the directory named by ``PRESETS_DIR``
and are passed to pipeline subprocesses via ``CLIP_PRESET``.
"""
from __future__ import annotations

import math
import os
import re
from typing import Any, Mapping, Optional

import yaml

PRESETS_DIR_ENV = "PRESETS_DIR"
ACTIVE_PRESET_ENV = "CLIP_PRESET"

_PRESET_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_TOP_KEYS = frozenset({
    "clip_format", "facecam_box", "layouts", "target_clips", "clip_seconds",
    "auto_hook", "captions", "trim_dead_footage",
})
_CAPTION_KEYS = frozenset({
    "language", "required_tags", "mention_one_of", "extra_instructions",
})
_DESCRIPTION_FIELDS = (
    "video_description_for_tiktok",
    "video_description_for_instagram",
)
_CAPTION_FIELDS = _DESCRIPTION_FIELDS + ("video_title_for_youtube_short",)


class PresetError(ValueError):
    """Base error for missing or invalid presets."""


class PresetNotFoundError(PresetError):
    """Raised when PRESETS_DIR or the requested preset file does not exist."""


class PresetValidationError(PresetError):
    """Raised when a preset name or YAML body fails validation."""


def _is_provided(value: Any) -> bool:
    """True when a request field was explicitly supplied by the caller."""
    if value is None:
        return False
    if isinstance(value, str) and not value.strip():
        return False
    return True


def _validate_name(name: str) -> str:
    if not isinstance(name, str):
        raise PresetValidationError("Preset name must be a string.")
    clean = name.strip()
    if not _PRESET_NAME_RE.match(clean):
        raise PresetValidationError(
            "Invalid preset name: use 1-64 alphanumeric, '-' or '_' characters."
        )
    return clean


def _dedupe_ci(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def _validate_captions(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise PresetValidationError("Preset 'captions' must be a mapping.")
    unknown = sorted(set(raw) - _CAPTION_KEYS)
    if unknown:
        raise PresetValidationError(
            f"Unknown preset 'captions' field(s): {', '.join(unknown)}."
        )
    out: dict[str, Any] = {}

    if raw.get("language") is not None:
        lang = raw["language"]
        if not isinstance(lang, str) or not lang.strip():
            raise PresetValidationError("Preset 'captions.language' must be a non-empty string.")
        out["language"] = lang.strip()

    if raw.get("required_tags") is not None:
        tags_raw = raw["required_tags"]
        if not isinstance(tags_raw, list):
            raise PresetValidationError("Preset 'captions.required_tags' must be a list.")
        tags: list[str] = []
        for entry in tags_raw:
            if not isinstance(entry, str) or not entry.strip():
                raise PresetValidationError("Each entry in 'captions.required_tags' must be a non-empty string.")
            tag = entry.strip()
            if any(ch.isspace() for ch in tag):
                raise PresetValidationError(f"Hashtag {tag!r} must not contain whitespace.")
            if not tag.startswith("#"):
                tag = f"#{tag}"
            if len(tag) < 2:
                raise PresetValidationError("Hashtag cannot be empty '#'.")
            tags.append(tag)
        out["required_tags"] = _dedupe_ci(tags)

    if raw.get("mention_one_of") is not None:
        mentions_raw = raw["mention_one_of"]
        if not isinstance(mentions_raw, list):
            raise PresetValidationError("Preset 'captions.mention_one_of' must be a list.")
        mentions: list[str] = []
        for entry in mentions_raw:
            if not isinstance(entry, str) or not entry.strip():
                raise PresetValidationError("Each entry in 'captions.mention_one_of' must be a non-empty string.")
            mentions.append(entry.strip())
        out["mention_one_of"] = _dedupe_ci(mentions)

    if raw.get("extra_instructions") is not None:
        extra = raw["extra_instructions"]
        if not isinstance(extra, str):
            raise PresetValidationError("Preset 'captions.extra_instructions' must be a string.")
        if extra.strip():
            out["extra_instructions"] = extra.strip()

    return out


def validate_preset(data: Any, *, name: str = "") -> dict[str, Any]:
    """Validate and normalize a parsed preset mapping."""
    if not isinstance(data, Mapping):
        label = f" {name!r}" if name else ""
        raise PresetValidationError(f"Preset{label} must be a YAML mapping.")
    unknown = sorted(set(data) - _TOP_KEYS)
    if unknown:
        raise PresetValidationError(f"Unknown preset field(s): {', '.join(unknown)}.")

    out: dict[str, Any] = {}

    if data.get("clip_format") is not None:
        cf = data["clip_format"]
        if not isinstance(cf, str) or not cf.strip():
            raise PresetValidationError("Preset 'clip_format' must be a non-empty string.")
        out["clip_format"] = cf.strip()

    if data.get("facecam_box") is not None:
        raw_box = data["facecam_box"]
        if not isinstance(raw_box, (list, tuple)) or len(raw_box) != 4:
            raise PresetValidationError("Preset 'facecam_box' must be a 4-item [x, y, w, h] list.")
        parsed_box: list[float] = []
        for coord_name, val in zip(("x", "y", "w", "h"), raw_box):
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
                raise PresetValidationError(f"Preset 'facecam_box' {coord_name} must be a finite number.")
            parsed_box.append(float(val))
        bx, by, bw, bh = parsed_box
        if not (0.0 <= bx < 1.0 and 0.0 <= by < 1.0 and 0.0 < bw <= 1.0 and 0.0 < bh <= 1.0):
            raise PresetValidationError(
                "Preset 'facecam_box' must use normalised 0-1 coordinates with 0 <= x,y < 1 and 0 < w,h <= 1."
            )
        if bx + bw > 1.0 + 1e-6 or by + bh > 1.0 + 1e-6:
            raise PresetValidationError("Preset 'facecam_box' x+w and y+h must not exceed 1.0.")
        out["facecam_box"] = parsed_box

    if data.get("layouts") is not None:
        layouts_raw = data["layouts"]
        if not isinstance(layouts_raw, list):
            raise PresetValidationError("Preset 'layouts' must be a list of layout names.")
        layouts: list[str] = []
        for item in layouts_raw:
            if not isinstance(item, str) or not item.strip():
                raise PresetValidationError("Each entry in 'layouts' must be a non-empty string.")
            layouts.append(item.strip().lower())
        out["layouts"] = layouts

    if data.get("target_clips") is not None:
        tc = data["target_clips"]
        if isinstance(tc, bool) or not isinstance(tc, int) or not (1 <= tc <= 15):
            raise PresetValidationError("Preset 'target_clips' must be an integer between 1 and 15.")
        out["target_clips"] = tc

    if data.get("clip_seconds") is not None:
        band = data["clip_seconds"]
        if not isinstance(band, (list, tuple)) or len(band) != 2:
            raise PresetValidationError("Preset 'clip_seconds' must be a 2-item [min, max] list.")
        lo, hi = band
        for bound_name, val in (("min", lo), ("max", hi)):
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
                raise PresetValidationError(f"Preset 'clip_seconds' {bound_name} must be a finite number.")
        lo_f, hi_f = float(lo), float(hi)
        if not (5 <= lo_f <= 175) or not (10 <= hi_f <= 180):
            raise PresetValidationError("Preset 'clip_seconds' bounds must stay within [5..175, 10..180].")
        if hi_f < lo_f + 5:
            raise PresetValidationError("Preset 'clip_seconds' max must be at least 5s above min.")
        out["clip_seconds"] = [lo, hi]

    if data.get("auto_hook") is not None:
        ah = data["auto_hook"]
        if not isinstance(ah, bool):
            raise PresetValidationError("Preset 'auto_hook' must be a boolean.")
        out["auto_hook"] = ah

    if data.get("trim_dead_footage") is not None:
        tdf = data["trim_dead_footage"]
        if not isinstance(tdf, bool):
            raise PresetValidationError("Preset 'trim_dead_footage' must be a boolean.")
        out["trim_dead_footage"] = tdf

    if data.get("captions") is not None:
        captions = _validate_captions(data["captions"])
        if captions:
            out["captions"] = captions

    return out


def _resolve_preset_path(name: str, directory: Optional[str] = None) -> str:
    clean = _validate_name(name)
    base_dir = directory if directory is not None else os.environ.get(PRESETS_DIR_ENV, "").strip()
    if not base_dir or not os.path.isdir(base_dir):
        raise PresetNotFoundError(f"Unknown preset '{clean}'.")
    real_base = os.path.realpath(base_dir)
    for ext in (".yaml", ".yml"):
        candidate = os.path.realpath(os.path.join(real_base, f"{clean}{ext}"))
        if os.path.commonpath([real_base, candidate]) != real_base:
            raise PresetValidationError("Invalid preset path.")
        if os.path.isfile(candidate):
            return candidate
    raise PresetNotFoundError(f"Unknown preset '{clean}'.")


def load_preset(name: str, directory: Optional[str] = None) -> dict[str, Any]:
    """Load and validate a named preset YAML file from PRESETS_DIR."""
    path = _resolve_preset_path(name, directory=directory)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise PresetValidationError(f"Invalid YAML in preset '{name}': {e}") from e
    except OSError as e:
        raise PresetNotFoundError(f"Could not read preset '{name}': {e}") from e
    return validate_preset(data, name=name)


def load_active_preset(
    env: Optional[Mapping[str, str]] = None,
    directory: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """Load the preset named by CLIP_PRESET in env, or None when unset."""
    source = os.environ if env is None else env
    name = (source.get(ACTIVE_PRESET_ENV) or "").strip()
    if not name:
        return None
    return load_preset(name, directory=directory)


def expand_job_defaults(
    preset: Optional[Mapping[str, Any]],
    overrides: Mapping[str, Any],
) -> dict[str, Any]:
    """Fill unset request fields from ``preset`` while keeping explicit values."""
    out = dict(overrides)
    if not preset:
        return out

    if not _is_provided(out.get("clip_format")) and "clip_format" in preset:
        out["clip_format"] = preset["clip_format"]

    if not _is_provided(out.get("facecam_box")) and "facecam_box" in preset:
        out["facecam_box"] = list(preset["facecam_box"])

    if not _is_provided(out.get("layouts")) and "layouts" in preset:
        out["layouts"] = list(preset["layouts"])

    if not _is_provided(out.get("target_clips")) and "target_clips" in preset:
        out["target_clips"] = preset["target_clips"]

    if "clip_seconds" in preset:
        lo, hi = preset["clip_seconds"]
        if not _is_provided(out.get("clip_min_seconds")):
            out["clip_min_seconds"] = lo
        if not _is_provided(out.get("clip_max_seconds")):
            out["clip_max_seconds"] = hi

    if not _is_provided(out.get("auto_hook")) and "auto_hook" in preset:
        out["auto_hook"] = preset["auto_hook"]

    return out


def apply_request_preset(
    preset_name: Any,
    fields: Mapping[str, Any],
    directory: Optional[str] = None,
) -> tuple[Optional[str], dict[str, Any]]:
    """Load ``preset_name`` when non-empty and expand defaults into ``fields``."""
    if preset_name is None or (isinstance(preset_name, str) and not preset_name.strip()):
        return None, dict(fields)
    clean = _validate_name(preset_name) if isinstance(preset_name, str) else _validate_name(str(preset_name))
    loaded = load_preset(clean, directory=directory)
    return clean, expand_job_defaults(loaded, fields)


def caption_language(preset: Optional[Mapping[str, Any]]) -> str:
    """Return the preset's required caption language, or '' when unset."""
    if not preset:
        return ""
    return str((preset.get("captions") or {}).get("language") or "").strip()


def caption_rules_prompt(preset: Optional[Mapping[str, Any]]) -> str:
    """Render the preset's caption rules block for selection prompts, or ''."""
    if not preset:
        return ""
    captions = preset.get("captions") or {}
    if not captions:
        return ""

    lines: list[str] = []
    lang = caption_language(preset)
    if lang:
        lines.append(
            f"- Language: ALL text fields (`video_description_for_tiktok`, "
            f"`video_description_for_instagram`, `video_title_for_youtube_short`, "
            f"`viral_hook_text`) MUST be written in {lang} "
            f"(overriding TRANSCRIPT_LANGUAGE if different)."
        )

    tags = captions.get("required_tags") or []
    if tags:
        lines.append(
            f"- Required hashtags: BOTH `video_description_for_tiktok` and "
            f"`video_description_for_instagram` MUST include: {' '.join(tags)}."
        )

    mentions = captions.get("mention_one_of") or []
    if mentions:
        lines.append(
            f"- Required mention: EVERY description and `video_title_for_youtube_short` "
            f"MUST name at least one of: {', '.join(mentions)}."
        )

    extra = (captions.get("extra_instructions") or "").strip()
    if extra:
        lines.append(f"- Additional instructions: {extra}")

    if not lines:
        return ""
    return "CAMPAIGN CAPTION RULES (mandatory):\n" + "\n".join(lines)


def _contains_token(text: str, token: str) -> bool:
    pattern = rf"(?i)(?<!\w){re.escape(token)}(?!\w)"
    return bool(re.search(pattern, text))


def enforce_clip(clip: dict[str, Any], preset: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    """Enforce caption rules on one clip dict in place.

    - Appends missing ``required_tags`` to ``video_description_for_tiktok`` and
      ``video_description_for_instagram`` only; compliant text is left untouched.
    - Checks ``mention_one_of`` across both descriptions and
      ``video_title_for_youtube_short`` without inventing text, recording any
      missing fields in ``clip['caption_issues']``.
    """
    if not preset or not isinstance(clip, dict):
        return clip
    captions = preset.get("captions") or {}
    required_tags = captions.get("required_tags") or []
    mentions = captions.get("mention_one_of") or []

    if required_tags:
        for field in _DESCRIPTION_FIELDS:
            current = clip.get(field)
            text = current if isinstance(current, str) else ""
            missing = [tag for tag in required_tags if not _contains_token(text, tag)]
            if missing:
                base = text.rstrip()
                suffix = " ".join(missing)
                clip[field] = f"{base} {suffix}" if base else suffix

    if mentions:
        issues: list[str] = []
        allowed = ", ".join(mentions)
        for field in _CAPTION_FIELDS:
            current = clip.get(field)
            text = current if isinstance(current, str) else ""
            if not any(_contains_token(text, name) for name in mentions):
                issues.append(f"{field}: missing required mention (one of: {allowed})")
        if issues:
            clip["caption_issues"] = issues
        else:
            clip.pop("caption_issues", None)

    return clip


def enforce_clips(
    shorts: list[dict[str, Any]],
    preset: Optional[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Enforce caption rules across a list of clip dicts in place."""
    if not preset:
        return shorts
    for clip in shorts or ():
        enforce_clip(clip, preset)
    return shorts
