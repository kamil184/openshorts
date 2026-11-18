"""Minimal interface for whole-clip vertical formatting strategies."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import os
import shutil
import tempfile
from typing import Any, Optional


@dataclass(frozen=True)
class ClipPlan:
    """Filtergraph and sidecar layout ranges produced by a format."""

    filtergraph: str
    layout_ranges: list[dict[str, Any]]
    meta: dict[str, Any] = field(default_factory=dict)


def _probe_video(input_video: str) -> tuple[int, int, float, float]:
    """Return (width, height, fps, duration_seconds) for ``input_video``."""
    import cv2

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_video}")
    try:
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()
    if orig_w <= 0 or orig_h <= 0:
        raise RuntimeError(f"Invalid video dimensions ({orig_w}x{orig_h}) for {input_video}")
    duration = total / fps if total > 0 else 0.0
    if duration <= 0.0:
        import subprocess
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", input_video],
            stderr=subprocess.STDOUT, timeout=30,
        ).decode().strip()
        duration = max(0.1, float(out))
    return orig_w, orig_h, fps, duration


class ClipFormat(ABC):
    """Whole-clip vertical composition interface.

    Subclasses implement ``analyze`` and ``compose``; ``render`` encodes the
    resulting filtergraph through the shared ffmpeg/GPU pipeline and writes
    the layout sidecar for captions and hooks.
    """

    name: str = ""

    @abstractmethod
    def analyze(self, input_video: str, orig_w: int, orig_h: int,
                out_w: int, out_h: int, duration: float,
                facecam_box: Any = None) -> Any:
        """Inspect the clip once and return format-specific analysis.

        ``facecam_box``: the job's webcam box (normalised ``[x, y, w, h]`` or
        ``"x,y,w,h"``); None reads ``GAMING_FACECAM_BOX`` from the environment."""

    @abstractmethod
    def compose(self, analysis: Any, orig_w: int, orig_h: int,
                out_w: int, out_h: int, duration: float) -> ClipPlan:
        """Build the 9:16 filtergraph and caption/hook layout ranges."""

    def render(self, input_video: str, final_output_video: str,
               aspect_ratio: float, watermark: bool = False,
               force_strategy: Optional[str] = None,
               crop_overrides: Optional[dict] = None,
               facecam_box: Any = None) -> bool:
        """Analyze, compose, encode ``final_output_video`` and write layout sidecar."""
        import layout_ranges
        import main as m
        import reframe_v2

        orig_w, orig_h, fps, duration = _probe_video(input_video)
        out_w, out_h = reframe_v2.delivery_size(orig_w, orig_h, aspect_ratio)
        analysis = self.analyze(input_video, orig_w, orig_h, out_w, out_h, duration,
                                facecam_box=facecam_box)
        plan = self.compose(analysis, orig_w, orig_h, out_w, out_h, duration)

        logo = m.watermark_logo_path() if watermark else None
        if logo and not os.path.exists(logo):
            logo = None

        workdir = tempfile.mkdtemp(prefix=f"clipfmt_{self.name}_")
        try:
            reframe_v2.encode_scenes(
                input_video, [(0.0, duration, plan.filtergraph, None)],
                orig_w, orig_h, out_w, out_h, fps, workdir, logo, final_output_video,
            )
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

        layout_ranges.write(final_output_video, plan.layout_ranges)
        return True
