"""Static 30/70 gaming stack: webcam on top, centred gameplay below."""
import math
import os
from typing import Any, Optional

from .base import ClipFormat, ClipPlan

DEFAULT_CAMERA_RATIO = 0.30
MIN_CAMERA_RATIO = 0.10
MAX_CAMERA_RATIO = 0.60
# Top headroom above detected inset boxes, matching camera_inset.FACE_HEADROOM
# so a face-derived box never slices off the streamer's hair.
CAMERA_HEADROOM = 0.35


def camera_ratio() -> float:
    """Configured top camera height fraction from ``GAMING_CAMERA_RATIO``."""
    raw = os.environ.get("GAMING_CAMERA_RATIO", str(DEFAULT_CAMERA_RATIO)).strip()
    try:
        val = float(raw)
    except ValueError:
        return DEFAULT_CAMERA_RATIO
    if not math.isfinite(val) or not (MIN_CAMERA_RATIO <= val <= MAX_CAMERA_RATIO):
        return DEFAULT_CAMERA_RATIO
    return val


def normalised_facecam_box(raw: Any) -> Optional[list[float]]:
    """``[x, y, w, h]`` (normalised 0-1) from ``"x,y,w,h"`` or a 4-item list,
    or None when it is absent or not a valid box."""
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, (list, tuple)):
        parts = list(raw)
    else:
        if not str(raw).strip():
            return None
        parts = [p.strip() for p in str(raw).split(",")]
    if len(parts) != 4 or any(isinstance(p, bool) for p in parts):
        return None
    try:
        nx, ny, nw, nh = (float(p) for p in parts)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (nx, ny, nw, nh)):
        return None
    if not (0.0 <= nx < 1.0 and 0.0 <= ny < 1.0 and 0.0 < nw <= 1.0 and 0.0 < nh <= 1.0):
        return None
    return [nx, ny, nw, nh]


def parse_facecam_box(raw: Any, orig_w: int, orig_h: int) -> Optional[tuple[int, int, int, int]]:
    """Parse ``GAMING_FACECAM_BOX=x,y,w,h`` (normalised 0-1) into pixel bounds."""
    box = normalised_facecam_box(raw)
    if box is None:
        return None
    nx, ny, nw, nh = box
    x = max(0, min(orig_w - 2, int(round(nx * orig_w))))
    y = max(0, min(orig_h - 2, int(round(ny * orig_h))))
    w = max(2, min(orig_w - x, int(round(nw * orig_w))))
    h = max(2, min(orig_h - y, int(round(nh * orig_h))))
    return (x, y, w, h)


def band_heights(out_h: int, ratio: Optional[float] = None) -> tuple[int, int]:
    """Return even ``(cam_h, game_h)`` summing to ``out_h``."""
    r = camera_ratio() if ratio is None else ratio
    cam_h = max(2, min(out_h - 2, int(round(out_h * r))))
    cam_h -= cam_h % 2
    cam_h = max(2, cam_h)
    game_h = out_h - cam_h
    return cam_h, game_h


def _align_even_box(x: int, y: int, w: int, h: int, max_w: int, max_h: int) -> tuple[int, int, int, int]:
    """Clamp and align a crop rectangle to even chroma coordinates."""
    w = max(2, min(max_w - (max_w % 2), int(w) - (int(w) % 2)))
    h = max(2, min(max_h - (max_h % 2), int(h) - (int(h) % 2)))
    x = max(0, min(max_w - w, int(x) - (int(x) % 2)))
    y = max(0, min(max_h - h, int(y) - (int(y) % 2)))
    return x, y, w, h


def camera_crop_box(box: tuple[int, int, int, int], orig_w: int, orig_h: int,
                    out_w: int, cam_h: int, known_bounds: bool = False) -> tuple[int, int, int, int]:
    """Crop rectangle for the top camera band without stretching or cropping the head.

    When ``known_bounds`` is True (an explicit ``GAMING_FACECAM_BOX``), the
    crop stays inside ``box`` and anchors to its top edge when trimming height
    so full headroom inside the webcam is kept. Otherwise (an estimated box
    from ``camera_inset.detect``), it adds ``CAMERA_HEADROOM`` above the box
    and grows to the band's aspect ratio without losing top headroom.
    """
    bx = max(0, min(orig_w - 2, int(box[0])))
    by = max(0, min(orig_h - 2, int(box[1])))
    bw = max(2, min(orig_w - bx, int(box[2])))
    bh = max(2, min(orig_h - by, int(box[3])))
    target_aspect = out_w / float(cam_h)

    if known_bounds:
        if bw / float(bh) < target_aspect:
            w = bw
            h = max(2, min(bh, int(round(bw / target_aspect))))
            x = bx
            # Anchor at the top of the webcam box rather than centring: hair
            # sits near the top border while chest/desk at the bottom can trim.
            y = by
        else:
            h = bh
            w = max(2, min(bw, int(round(bh * target_aspect))))
            x = bx + (bw - w) // 2
            y = by
        x = x + (x % 2)
        y = y + (y % 2)
        w = max(2, (min(bx + bw - x, w) // 2) * 2)
        h = max(2, (min(by + bh - y, h) // 2) * 2)
        return _align_even_box(x, y, w, h, orig_w, orig_h)

    # Detected inset boxes from camera_inset sit close to the brow line; add
    # top headroom before matching the band aspect so hair stays intact.
    pad_top = int(round(bh * CAMERA_HEADROOM))
    new_y = max(0, by - pad_top)
    x, y, w, h = bx, new_y, bw, bh + (by - new_y)

    if w / float(h) < target_aspect:
        want_w = min(orig_w, int(round(h * target_aspect)))
        x = int(round(x + w / 2.0 - want_w / 2.0))
        w = want_w
        if w / float(h) < target_aspect * 0.99:
            h = max(2, int(round(w / target_aspect)))
    else:
        want_h = min(orig_h, int(round(w / target_aspect)))
        h = want_h
        if w / float(h) > target_aspect * 1.01:
            want_w = max(2, int(round(h * target_aspect)))
            x = int(round(x + (w - want_w) / 2.0))
            w = want_w
    return _align_even_box(x, y, w, h, orig_w, orig_h)


def _overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx + bw and ax + aw > bx and ay < by + bh and ay + ah > by


def _fit_in_region(rx0: int, ry0: int, rx1: int, ry1: int,
                   aspect: float, orig_w: int, orig_h: int) -> Optional[tuple[int, int, int, int]]:
    """Largest rectangle of ``aspect`` inside [rx0..rx1, ry0..ry1], closest to frame centre."""
    rw, rh = rx1 - rx0, ry1 - ry0
    if rw < 2 or rh < 2:
        return None
    if rw / float(rh) >= aspect:
        ch = rh - (rh % 2)
        cw = int(round(ch * aspect))
        cw -= cw % 2
        if cw > rw:
            cw = rw - (rw % 2)
    else:
        cw = rw - (rw % 2)
        ch = int(round(cw / aspect))
        ch -= ch % 2
        if ch > rh:
            ch = rh - (rh % 2)
    if cw < 2 or ch < 2:
        return None
    cx = max(rx0, min(rx1 - cw, int(round(orig_w / 2.0 - cw / 2.0))))
    cy = max(ry0, min(ry1 - ch, int(round(orig_h / 2.0 - ch / 2.0))))
    cx = max(rx0, min(rx1 - cw, cx - (cx % 2)))
    cy = max(ry0, min(ry1 - ch, cy - (cy % 2)))
    return (cx, cy, cw, ch)


def gameplay_crop_box(orig_w: int, orig_h: int, out_w: int, game_h: int,
                      exclude_box: Optional[tuple[int, int, int, int]] = None) -> tuple[int, int, int, int]:
    """Static centre crop for gameplay, avoiding ``exclude_box`` if it overlaps."""
    aspect = out_w / float(game_h)
    centred = _fit_in_region(0, 0, orig_w, orig_h, aspect, orig_w, orig_h)
    assert centred is not None
    if not exclude_box or not _overlaps(centred, exclude_box):
        return centred

    fx, fy, fw, fh = exclude_box
    regions = [
        (fx + fw, 0, orig_w, orig_h),
        (0, 0, fx, orig_h),
        (0, 0, orig_w, fy),
        (0, fy + fh, orig_w, orig_h),
    ]
    candidates = [c for c in (_fit_in_region(*r, aspect, orig_w, orig_h) for r in regions) if c is not None]
    if not candidates:
        return centred

    def _rank(c: tuple[int, int, int, int]) -> tuple[int, float]:
        cx, cy, cw, ch = c
        dist2 = (cx + cw / 2.0 - orig_w / 2.0) ** 2 + (cy + ch / 2.0 - orig_h / 2.0) ** 2
        return (-(cw * ch), dist2)

    return min(candidates, key=_rank)


def build_filtergraph(out_w: int, out_h: int,
                      cam_crop: Optional[tuple[int, int, int, int]],
                      game_crop: tuple[int, int, int, int],
                      cam_h: int, game_h: int) -> str:
    """Build the ffmpeg/osrender filtergraph for stacked or fallback gameplay."""
    gx, gy, gw, gh = game_crop
    if cam_crop is None:
        return f"[0:v]crop=w={gw}:h={gh}:x={gx}:y={gy},scale={out_w}:{out_h},setsar=1[v]"
    cx, cy, cw, ch = cam_crop
    return (
        f"[0:v]split=2[ca][ga];"
        f"[ca]crop=w={cw}:h={ch}:x={cx}:y={cy},scale={out_w}:{cam_h}[cam];"
        f"[ga]crop=w={gw}:h={gh}:x={gx}:y={gy},scale={out_w}:{game_h}[game];"
        f"[cam][game]vstack=inputs=2,setsar=1[v]"
    )


def _with_headroom(box: tuple[int, int, int, int], orig_w: int, orig_h: int) -> tuple[int, int, int, int]:
    """``box`` grown upward by ``CAMERA_HEADROOM``, as ``camera_crop_box`` does."""
    bx, by, bw, bh = (int(v) for v in box)
    new_y = max(0, by - int(round(bh * CAMERA_HEADROOM)))
    return (bx, new_y, min(bw, orig_w - bx), min(bh + (by - new_y), orig_h - new_y))


class GamingStackFormat(ClipFormat):
    """30/70 static stack: webcam on top, centred gameplay below."""

    name = "gaming_stack"

    def analyze(self, input_video: str, orig_w: int, orig_h: int,
                out_w: int, out_h: int, duration: float,
                facecam_box: Any = None) -> dict[str, Any]:
        raw = os.environ.get("GAMING_FACECAM_BOX") if facecam_box is None else facecam_box
        env_box = parse_facecam_box(raw, orig_w, orig_h)
        if env_box is not None:
            return {"box": env_box, "known_bounds": True, "source": "env"}

        import camera_inset
        try:
            box = camera_inset.detect(input_video)
        except Exception as e:
            print(f"   ⚠️ Gaming stack inset detection failed ({e}) — static gameplay fallback")
            box = None
        return {"box": box, "known_bounds": False, "source": "detect" if box else "none"}

    def compose(self, analysis: Any, orig_w: int, orig_h: int,
                out_w: int, out_h: int, duration: float) -> ClipPlan:
        box = (analysis or {}).get("box")
        known_bounds = bool((analysis or {}).get("known_bounds"))
        dur = round(float(duration), 3)

        if not box:
            game_crop = gameplay_crop_box(orig_w, orig_h, out_w, out_h, exclude_box=None)
            graph = build_filtergraph(out_w, out_h, None, game_crop, 0, out_h)
            return ClipPlan(
                filtergraph=graph,
                layout_ranges=[{"start": 0.0, "end": dur, "layout": "general"}],
                meta={"fallback": True, "game_crop": game_crop},
            )

        cam_h, game_h = band_heights(out_h)
        cam_crop = camera_crop_box(box, orig_w, orig_h, out_w, cam_h, known_bounds=known_bounds)
        # Keep gameplay off the webcam itself (plus the headroom above a
        # detected box), not off the cam crop: that crop is widened to the
        # band's aspect, and avoiding the widening shoves gameplay off-centre.
        exclude = box if known_bounds else _with_headroom(box, orig_w, orig_h)
        game_crop = gameplay_crop_box(orig_w, orig_h, out_w, game_h, exclude_box=exclude)
        graph = build_filtergraph(out_w, out_h, cam_crop, game_crop, cam_h, game_h)
        seam = round(cam_h / float(out_h), 4)
        return ClipPlan(
            filtergraph=graph,
            layout_ranges=[{"start": 0.0, "end": dur, "layout": "stack", "seam": seam}],
            meta={
                "fallback": False,
                "cam_crop": cam_crop,
                "game_crop": game_crop,
                "cam_h": cam_h,
                "game_h": game_h,
                "seam": seam,
            },
        )
