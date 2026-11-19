"""The free-plan watermark: geometry, the overlay filter, and the marked copy.

Moved out of main.py so the renderers (reframe_v2, clip_formats, recut) and
the API's delivery path can reach it without importing the pipeline. The
naming rules for the ``wm_`` copies stay in watermarked.py, which must remain
standard-library only.
"""
import os
import shutil
import subprocess

import gpu_decode
from ffmpeg_utils import video_encode_args, QUALITY, METADATA_SCRUB


# Watermark geometry, as fractions of the clip width/height.
#
# Vertical placement is the whole point: the top and bottom strips of a 9:16
# clip are either black bars or blurred filler (GENERAL layout), so a mark up
# there is cropped away without touching a single pixel of real footage. At 40%
# of the height it sits inside the content band — a 16:9 source letterboxed
# into 9:16 spans roughly 34%-66% — so removing the mark means cutting into the
# picture. Left-aligned, like OpusClip's.
WATERMARK_WIDTH_RATIO = 0.30
WATERMARK_MARGIN_RATIO = 0.05
WATERMARK_Y_RATIO = 0.40
WATERMARK_OPACITY = 0.85


def watermark_logo_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "assets", "watermark.png")


def watermark_filter(vw, vh, video="[0:v]", logo="[1:v]", out=""):
    """filter_complex chain overlaying the logo input on a vw x vh video."""
    wm_w = max(80, int(vw * WATERMARK_WIDTH_RATIO))
    x = int(vw * WATERMARK_MARGIN_RATIO)
    y = int(vh * WATERMARK_Y_RATIO)
    return (
        f"{logo}scale={wm_w}:-1,format=rgba,"
        f"colorchannelmixer=aa={WATERMARK_OPACITY}[wm];"
        f"{video}[wm]overlay=x={x}:y={y}{out}"
    )


def apply_watermark(video_path, output_path=None):
    """Burn the OpenShorts watermark into a finished clip (free plan).

    One re-encode pass over the final file so every output format (TRACK,
    GENERAL, horizontal passthrough) gets the mark. In place by default; with
    ``output_path`` the source stays untouched and the marked copy is written
    there, which is how the pipeline keeps the clean file next to the served
    one (see ``mark_delivery``).
    """
    logo_path = watermark_logo_path()
    if not os.path.exists(logo_path):
        print(f"   ⚠️ Watermark asset missing ({logo_path}); clip kept unmarked.")
        return False

    # Scale the lockup from the clip's real width: overlay can't read the other
    # input's size, and computing it here avoids the deprecated scale2ref.
    try:
        probe = subprocess.check_output(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", video_path],
            stderr=subprocess.STDOUT, timeout=60,
        ).decode().strip().split("x")
        vw, vh = int(probe[0]), int(probe[1])
    except Exception as e:
        print(f"   ⚠️ Could not probe clip for watermark ({e}); clip kept unmarked.")
        return False

    filt = watermark_filter(vw, vh)
    tmp_path = (output_path or video_path) + ".wm.mp4"
    cmd = ["ffmpeg", "-y", "-i", video_path, "-i", logo_path,
           "-filter_complex", filt,
           *video_encode_args(QUALITY), "-c:a", "copy", *METADATA_SCRUB,
           "-movflags", "+faststart", tmp_path]
    result = gpu_decode.run(
        lambda on_gpu: gpu_decode.on_card(cmd, video_path) if on_gpu else cmd,
        lambda c: subprocess.run(c, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                 timeout=1800),
        ok=lambda r: r.returncode == 0 and os.path.exists(tmp_path),
        path=video_path)
    if result.returncode == 0 and os.path.exists(tmp_path):
        os.replace(tmp_path, output_path or video_path)
        return True
    err = (result.stderr or b"").decode(errors="ignore")[-300:]
    print(f"   ⚠️ Watermark pass failed (clip kept unmarked): {err}")
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    return False


def mark_delivery(final_path, marker=None):
    """The free-plan copy of a finished clip to serve: ``wm_<final>`` next to it.

    The clean ``final_path`` is left as it is. That is the whole point: an
    upgrade re-points the clip at the clean twin instead of re-running the
    job (``watermarked`` module). An existing copy is reused, so re-choosing
    a file the user already had (captions off, hook off) costs no encode.

    Returns the marked path, or ``final_path`` itself when the copy could not
    be made: the mark must never cost the user the clip (same fail-open rule
    apply_watermark has always had).
    """
    import watermarked
    out_path = os.path.join(os.path.dirname(final_path),
                            watermarked.marked_name(final_path))
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return out_path
    ok = (marker or apply_watermark)(final_path, out_path)
    if ok and os.path.exists(out_path):
        return out_path
    # Still the wm_ name: the /videos guard refuses the clean deliverables of
    # a free job, so serving the clean name here would be a clip the user
    # cannot play. The copy simply carries no mark.
    print(f"   ⚠️ Serving {os.path.basename(final_path)} unmarked (copy).")
    try:
        shutil.copyfile(final_path, out_path)
        return out_path
    except OSError:
        return final_path
