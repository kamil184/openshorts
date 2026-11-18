"""Adapter that preserves the default scene-by-scene reframe pipeline."""
from typing import Any, Optional

from .base import ClipFormat, ClipPlan


class DefaultFormat(ClipFormat):
    """Delegates directly to ``reframe_v2.render``."""

    name = "default"

    def analyze(self, input_video: str, orig_w: int, orig_h: int,
                out_w: int, out_h: int, duration: float) -> Any:
        return None

    def compose(self, analysis: Any, orig_w: int, orig_h: int,
                out_w: int, out_h: int, duration: float) -> ClipPlan:
        raise NotImplementedError("DefaultFormat renders via reframe_v2 directly")

    def render(self, input_video: str, final_output_video: str,
               aspect_ratio: float, watermark: bool = False,
               force_strategy: Optional[str] = None,
               crop_overrides: Optional[dict] = None) -> bool:
        import reframe_v2

        return reframe_v2.render(
            input_video, final_output_video, aspect_ratio,
            force_strategy=force_strategy,
            crop_overrides=crop_overrides,
            watermark=watermark,
        )
