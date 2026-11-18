"""Registry of clip formats by name."""
import os
from typing import Optional

from .base import ClipFormat
from .default import DefaultFormat
from .gaming_stack import GamingStackFormat

_REGISTRY: dict[str, ClipFormat] = {}


def register(fmt: ClipFormat) -> ClipFormat:
    """Register a ``ClipFormat`` instance under its lowercase ``name``."""
    key = str(fmt.name or "").strip().lower()
    if not key:
        raise ValueError("ClipFormat.name must be non-empty")
    _REGISTRY[key] = fmt
    return fmt


def get(name: Optional[str]) -> ClipFormat:
    """Return the registered ``ClipFormat`` for ``name`` (empty/None -> ``default``)."""
    key = str(name or "default").strip().lower() or "default"
    if key not in _REGISTRY:
        raise ValueError(f"Unknown clip format: {name!r} (available: {', '.join(available())})")
    return _REGISTRY[key]


def available() -> tuple[str, ...]:
    """Sorted tuple of registered format names."""
    return tuple(sorted(_REGISTRY))


def from_env(var: str = "CLIP_FORMAT") -> ClipFormat:
    """Resolve the active ``ClipFormat`` from environment variable ``var``."""
    return get(os.environ.get(var))


register(DefaultFormat())
register(GamingStackFormat())
