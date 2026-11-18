"""Trim dead footage off the head and tail of selected clips.

Clip selection picks ranges from the transcript, so nothing stops a clip from
starting or ending on footage where nothing happens on screen: a game loading
screen with a spinner, a black transition, a frozen frame. A streamer who keeps
talking over a loading screen makes it worse, because the word snap then pulls
the end right into it.

This module looks at the first and last ``max_trim`` seconds of each clip and
moves the boundary inward past the dead stretch. It is deterministic: a few
seconds of tiny greyscale frames decoded by ffmpeg, two statistics per frame,
fixed thresholds.

What counts as dead (luma 0-255 on a 160x90 greyscale proxy, 4 samples a
second, the facecam box masked out when one is configured):

* frozen: the median frame-to-frame difference over one second is below
  ``FROZEN_DIFF``. A static image with codec noise measures 0.03-0.12; an
  idle game lobby with animated characters 0.2-0.8; gameplay 20-50.
* dark and still: mean luma below ``DARK_MEAN`` with spread below
  ``DARK_STD`` and the median difference below ``DARK_DIFF``. Loading
  screens, black frames and fades. A spinner moves a few hundred pixels and
  measures 0.4-0.7; a dark game scene with action measures far above 2.

A stretch is trimmed only when at least ``MIN_DEAD`` seconds of it touch the
clip edge. Live islands of up to ``MAX_GAP`` seconds inside it (a scene cut,
a second of a static menu) do not break it.

Speech: the cut never falls inside a word. It lands in a pause (a gap of at
least ``MIN_PAUSE`` seconds, or the clip's silent edge), and that pause must
reach into the dead stretch, so every word removed is spoken entirely over
dead footage. When speech runs through the dead stretch without a pause, the
clip keeps it.

The clip never gets shorter than the minimum clip length and never loses more
than ``max_trim`` seconds on one edge (``CLIP_DEAD_TRIM_MAX_SECONDS``).

Switch: ``CLIP_DEAD_TRIM=0`` turns it off and ``CLIP_DEAD_TRIM=1`` on for the
process; without it, a preset's ``trim_dead_footage`` decides; the default is
on.
"""
from __future__ import annotations

import math
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

ENV_SWITCH = "CLIP_DEAD_TRIM"
ENV_MAX_SECONDS = "CLIP_DEAD_TRIM_MAX_SECONDS"
PRESET_KEY = "trim_dead_footage"

SAMPLE_FPS = 4.0
PROXY_W, PROXY_H = 160, 90

FROZEN_DIFF = 0.15
DARK_MEAN = 40.0
DARK_STD = 40.0
DARK_DIFF = 2.0
# Samples in the rolling median of the frame difference (one second).
MEDIAN_SAMPLES = 5

MIN_DEAD = 1.5
MAX_GAP = 1.5
MIN_TRIM = 1.0
DEFAULT_MAX_TRIM = 12.0
MEASURE_WORKERS = 3

MIN_PAUSE = 0.3
# How far a cut may sit from the word it follows or precedes, as in
# clip_selection.snap_clip_to_words.
TAIL_PAD = 0.45
LEAD_PAD = 0.35
EDGE_PAD = 0.1

_OFF = {"0", "false", "no", "off"}
_ON = {"1", "true", "yes", "on"}


# --- Switches ----------------------------------------------------------------

def enabled(preset: Optional[Mapping[str, Any]] = None,
            env: Optional[Mapping[str, str]] = None) -> bool:
    """Whether dead-footage trimming runs: env switch, then preset, then on."""
    source = os.environ if env is None else env
    raw = (source.get(ENV_SWITCH) or "").strip().lower()
    if raw in _OFF:
        return False
    if raw in _ON:
        return True
    if preset and PRESET_KEY in preset:
        return bool(preset[PRESET_KEY])
    return True


def max_trim_seconds(env: Optional[Mapping[str, str]] = None) -> float:
    source = os.environ if env is None else env
    try:
        val = float(source.get(ENV_MAX_SECONDS, ""))
    except ValueError:
        return DEFAULT_MAX_TRIM
    if not math.isfinite(val) or val <= 0:
        return DEFAULT_MAX_TRIM
    return min(val, 60.0)


def facecam_mask(raw_box: Optional[str], width: int = PROXY_W,
                 height: int = PROXY_H) -> Optional[np.ndarray]:
    """Boolean mask of the gameplay area: everything but the facecam box.

    ``raw_box`` is ``GAMING_FACECAM_BOX`` (``x,y,w,h``, normalised 0-1). The
    streamer's face moves while the game shows a loading screen, so it must
    not count as action. None when no valid box is given.
    """
    if not raw_box or not str(raw_box).strip():
        return None
    try:
        x, y, w, h = (float(p) for p in str(raw_box).split(","))
    except ValueError:
        return None
    if not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
        return None
    x0, y0 = max(0, int(x * width)), max(0, int(y * height))
    x1 = min(width, int(math.ceil((x + w) * width)))
    y1 = min(height, int(math.ceil((y + h) * height)))
    mask = np.ones((height, width), dtype=bool)
    mask[y0:y1, x0:x1] = False
    if not mask.any():
        return None
    return mask


# --- Frame statistics --------------------------------------------------------

def frame_stats(frames: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Per frame ``(mean, std, diff)`` over the masked area.

    ``frames`` is ``(n, h, w)`` greyscale. ``diff`` is the mean absolute
    difference to the previous frame (the first frame copies the second's).
    """
    frames = np.asarray(frames, dtype=np.float32)
    n = len(frames)
    if n == 0:
        return np.zeros((0, 3), dtype=np.float32)
    flat = frames[:, mask] if mask is not None else frames.reshape(n, -1)
    out = np.zeros((n, 3), dtype=np.float32)
    out[:, 0] = flat.mean(axis=1)
    out[:, 1] = flat.std(axis=1)
    if n > 1:
        out[1:, 2] = np.abs(np.diff(flat, axis=0)).mean(axis=1)
        out[0, 2] = out[1, 2]
    return out


def dead_flags(stats: np.ndarray) -> np.ndarray:
    """Which samples show dead footage (see the module docstring)."""
    n = len(stats)
    if n == 0:
        return np.zeros(0, dtype=bool)
    diff = stats[:, 2]
    half = MEDIAN_SAMPLES // 2
    padded = np.pad(diff, half, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, MEDIAN_SAMPLES)
    motion = np.median(windows, axis=1)
    frozen = motion < FROZEN_DIFF
    dark = (stats[:, 0] < DARK_MEAN) & (stats[:, 1] < DARK_STD) & (motion < DARK_DIFF)
    return frozen | dark


def _edge_run(flags: Sequence[bool], step: float) -> int:
    """Length in samples of the dead run touching index 0, or 0.

    Live gaps of up to ``MAX_GAP`` seconds inside the run are absorbed; the
    run ends at its last dead sample. Shorter than ``MIN_DEAD`` seconds of dead
    samples counts as no run.
    """
    max_gap = int(round(MAX_GAP / step))
    last_dead, dead_count, gap = -1, 0, 0
    for i, dead in enumerate(flags):
        if dead:
            last_dead, gap = i, 0
            dead_count += 1
        else:
            gap += 1
            if gap > max_gap:
                break
    if last_dead < 0 or dead_count * step < MIN_DEAD:
        return 0
    return last_dead + 1


def dead_head_end(times: Sequence[float], flags: Sequence[bool]) -> Optional[float]:
    """End time of the dead stretch at the start of the samples, or None."""
    if len(times) < 2:
        return None
    step = float(times[1] - times[0])
    run = _edge_run(list(flags), step)
    if not run:
        return None
    return float(times[run - 1] + step)


def dead_tail_start(times: Sequence[float], flags: Sequence[bool]) -> Optional[float]:
    """Start time of the dead stretch at the end of the samples, or None."""
    if len(times) < 2:
        return None
    step = float(times[1] - times[0])
    run = _edge_run(list(flags)[::-1], step)
    if not run:
        return None
    return float(times[len(times) - run])


# --- Speech-safe cut points --------------------------------------------------

def _word_spans(words: Iterable[Mapping[str, Any]]) -> list[tuple[float, float]]:
    """Merged ``(start, end)`` speech intervals from ``{'s','e'}`` words."""
    spans = sorted((float(w["s"]), max(float(w["s"]), float(w["e"]))) for w in words
                   if w.get("s") is not None and w.get("e") is not None)
    merged: list[list[float]] = []
    for s, e in spans:
        if merged and s - merged[-1][1] < MIN_PAUSE:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def _pauses(words) -> list[tuple[float, float]]:
    """Silent intervals between utterances, open-ended at both sides."""
    spans = _word_spans(words)
    if not spans:
        return [(-math.inf, math.inf)]
    out = [(-math.inf, spans[0][0])]
    out += [(a[1], b[0]) for a, b in zip(spans, spans[1:])]
    out.append((spans[-1][1], math.inf))
    return out


def tail_cut(dead_start: float, words) -> Optional[float]:
    """Where to end a clip whose footage is dead from ``dead_start`` on.

    The earliest pause that reaches past ``dead_start``: inside it the cut sits
    at the dead start itself; a later pause is cut ``TAIL_PAD`` after its last
    word (half the pause at most). None when no pause exists.
    """
    for a, b in _pauses(words):
        if b <= dead_start:
            continue
        if a <= dead_start:
            return min(max(dead_start, a + EDGE_PAD), b)
        return a + min(TAIL_PAD, (b - a) / 2)
    return None


def head_cut(dead_end: float, words) -> Optional[float]:
    """Where to start a clip whose footage is dead until ``dead_end``.

    Mirror of :func:`tail_cut`: the latest pause that starts before
    ``dead_end``.
    """
    for a, b in reversed(_pauses(words)):
        if a >= dead_end:
            continue
        if b >= dead_end:
            return max(min(dead_end, b - EDGE_PAD), a)
        return b - min(LEAD_PAD, (b - a) / 2)
    return None


# --- Sampling ------------------------------------------------------------------

def sample_frames(video_path: str, start: float, end: float,
                  fps: float = SAMPLE_FPS) -> np.ndarray:
    """Greyscale ``PROXY_W`` x ``PROXY_H`` frames of [start, end] at ``fps``."""
    duration = max(0.0, end - start)
    if duration <= 0:
        return np.zeros((0, PROXY_H, PROXY_W), dtype=np.uint8)
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-threads", "2",
           "-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", video_path,
           "-an", "-sn", "-dn",
           "-vf", f"fps={fps:g},scale={PROXY_W}:{PROXY_H},format=gray",
           "-f", "rawvideo", "-"]
    proc = subprocess.run(cmd, capture_output=True, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", "replace").strip()[-400:])
    size = PROXY_W * PROXY_H
    n = len(proc.stdout) // size
    return np.frombuffer(proc.stdout[:n * size], dtype=np.uint8).reshape(n, PROXY_H, PROXY_W)


def _flags_for(video_path, start, end, mask):
    frames = sample_frames(video_path, start, end)
    times = start + np.arange(len(frames)) / SAMPLE_FPS
    return times, dead_flags(frame_stats(frames, mask))


# --- Refinement ----------------------------------------------------------------

def refine_range(start: float, end: float, words, min_len: float,
                 head_dead_end: Optional[float], tail_dead_start: Optional[float],
                 max_trim: float = DEFAULT_MAX_TRIM) -> tuple[float, float]:
    """Apply the measured dead stretches to ``[start, end]``.

    Each edge moves by at least ``MIN_TRIM`` and at most ``max_trim`` seconds,
    or not at all. When both edges together would leave less than ``min_len``,
    the larger single trim that still fits wins.
    """
    new_start, new_end = start, end
    if head_dead_end is not None:
        cut = head_cut(head_dead_end, words)
        if cut is not None and MIN_TRIM <= cut - start <= max_trim:
            new_start = cut
    if tail_dead_start is not None:
        cut = tail_cut(tail_dead_start, words)
        if cut is not None and MIN_TRIM <= end - cut <= max_trim:
            new_end = cut
    if new_end - new_start >= min_len:
        return round(new_start, 3), round(new_end, 3)
    options = [(s, e) for s, e in ((start, new_end), (new_start, end))
               if e - s >= min_len]
    if not options:
        return start, end
    s, e = min(options, key=lambda r: r[1] - r[0])
    return round(s, 3), round(e, 3)


def refine_clip(video_path: str, start: float, end: float, words, min_len: float,
                mask: Optional[np.ndarray] = None,
                max_trim: float = DEFAULT_MAX_TRIM) -> tuple[float, float]:
    """Measure the clip's edges in ``video_path`` and return the refined range."""
    if end - start <= min_len + MIN_TRIM:
        return start, end
    window = min(max_trim, end - start)
    times, flags = _flags_for(video_path, start, start + window, mask)
    head = dead_head_end(times, flags)
    times, flags = _flags_for(video_path, end - window, end, mask)
    tail = dead_tail_start(times, flags)
    if tail is not None and tail <= end - window + 1e-6:
        # The whole window is dead: the stretch starts earlier still, and the
        # cut can only go as far as the window reaches.
        tail = end - window
    if head is not None and head >= start + window - 1e-6:
        head = start + window
    return refine_range(start, end, words, min_len, head, tail, max_trim)


def transcript_words(transcript: Optional[Mapping[str, Any]]) -> list[dict]:
    words = []
    for seg in (transcript or {}).get("segments", []) or []:
        for w in seg.get("words", []) or []:
            if w.get("start") is not None and w.get("end") is not None:
                words.append({"s": float(w["start"]), "e": float(w["end"])})
    return words


def refine_clips(video_path: str, shorts: list[dict],
                 transcript: Optional[Mapping[str, Any]], min_len: float,
                 preset: Optional[Mapping[str, Any]] = None,
                 env: Optional[Mapping[str, str]] = None) -> int:
    """Trim dead head/tail footage of every clip in ``shorts`` in place.

    A trimmed clip records its range before trimming under ``dead_trim``.
    A clip that cannot be measured keeps its range. Returns the number of
    clips changed.
    """
    if not shorts or not enabled(preset, env):
        return 0
    source = os.environ if env is None else env
    mask = facecam_mask(source.get("GAMING_FACECAM_BOX"))
    max_trim = max_trim_seconds(env)
    words = transcript_words(transcript)

    def measure(clip):
        start, end = float(clip["start"]), float(clip["end"])
        return start, end, refine_clip(video_path, start, end, words, min_len,
                                       mask=mask, max_trim=max_trim)

    # Each clip costs two short decodes; a few at once keep the job's wait
    # down without taking the whole machine (each ffmpeg runs on 2 threads).
    with ThreadPoolExecutor(max_workers=min(MEASURE_WORKERS, len(shorts))) as pool:
        futures = [pool.submit(measure, clip) for clip in shorts]
    changed = 0
    for i, (clip, future) in enumerate(zip(shorts, futures)):
        try:
            start, end, (new_start, new_end) = future.result()
        except Exception as e:  # measuring is best effort; the clip stays as picked
            print(f"   ℹ️ Dead-footage check skipped for clip {i + 1} "
                  f"({type(e).__name__}: {e}).")
            continue
        if (new_start, new_end) == (start, end):
            continue
        clip["dead_trim"] = {"from": [start, end],
                             "head": round(new_start - start, 3),
                             "tail": round(end - new_end, 3)}
        clip["start"], clip["end"] = new_start, new_end
        changed += 1
        print(f"   ✂️ Clip {i + 1}: trimmed dead footage {start:.2f}-{end:.2f}s → "
              f"{new_start:.2f}-{new_end:.2f}s")
    return changed

def apply(video_path: str, shorts: list[dict],
          transcript: Optional[Mapping[str, Any]]) -> int:
    """Job entry point: :func:`refine_clips` with the job's preset and the
    configured minimum clip length."""
    import presets
    from clip_selection import clip_duration_bounds

    try:
        preset = presets.load_active_preset()
    except presets.PresetError as e:
        print(f"   ℹ️ Dead-footage trim uses its default ({e}).")
        preset = None
    min_len, _ = clip_duration_bounds()
    return refine_clips(video_path, shorts, transcript, min_len, preset=preset)
