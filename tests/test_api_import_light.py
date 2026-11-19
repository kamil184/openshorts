"""Importing the API must not load the heavy ML stack.

Target rule (import layering): the api and jobs layers never import torch,
ultralytics or mediapipe, even transitively. Each check runs in a fresh
interpreter so the result does not depend on what other tests already imported.

Importing ``app`` alone never was the problem: app reached main lazily, inside
handlers, so the API process loaded YOLO and MediaPipe the first time one of
them ran. The other checks pin that runtime path: the helpers the API calls
live outside main, the detection modules load their models on first use, and
importing main itself no longer builds them.
"""
import ast
import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HEAVY_ROOTS = ("torch", "ultralytics", "mediapipe")
IMPORT_TIMEOUT_S = 120

# What the API may still take from main: the Thumbnail Studio download, which
# is the whole source stage and moves out with it.
APP_MAIN_IMPORTS_ALLOWED = {"download_youtube_video"}

# Every module the API or a renderer imports for detection, scenes, the
# watermark or the caption pass.
LIGHT_MODULES = (
    "vision", "scene_analysis", "watermark_overlay", "subtitles",
    "transcribe_backends", "recut", "reframe_v2", "camera_inset",
    "split_layout", "panel_layout", "screencast_layout", "clip_formats.base",
)


def _loaded_after(code):
    probe = (
        "import sys\n"
        f"{code}\n"
        f"heavy = {HEAVY_ROOTS!r}\n"
        "print(sorted(m for m in sys.modules if m.split('.')[0] in heavy))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=IMPORT_TIMEOUT_S,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    return proc.stdout.strip().splitlines()[-1]


def test_api_import_does_not_load_ml_stack():
    loaded = _loaded_after("import app")
    assert loaded == "[]", f"ML stack imported with the API: {loaded}"


def test_app_takes_nothing_else_from_main():
    tree = ast.parse(open(os.path.join(REPO_ROOT, "app.py"), encoding="utf-8").read())
    taken = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(a.name != "main" for a in node.names), \
                f"app.py:{node.lineno} imports the whole pipeline module"
        elif isinstance(node, ast.ImportFrom) and node.module == "main":
            taken |= {a.name for a in node.names}
    assert taken <= APP_MAIN_IMPORTS_ALLOWED, taken - APP_MAIN_IMPORTS_ALLOWED


def test_detection_and_render_modules_import_without_models():
    loaded = _loaded_after("\n".join(f"import {m}" for m in LIGHT_MODULES))
    assert loaded == "[]", f"ML stack imported with the renderers: {loaded}"


def test_api_helpers_run_without_models(tmp_path):
    """The calls app makes into these modules, short of real detection."""
    clip = tmp_path / "t_clip_1.mp4"
    clip.write_bytes(b"x")
    loaded = _loaded_after(
        "import scene_analysis, subtitles, watermark_overlay\n"
        f"clip = {str(clip)!r}\n"
        "assert subtitles.auto_caption_clip(clip, {'segments': []}, 0.0, 1.0) is None\n"
        "served = watermark_overlay.mark_delivery(clip, marker=lambda src, dst: False)\n"
        "assert served.endswith('wm_t_clip_1.mp4'), served\n"
        "try:\n"
        "    scene_analysis.get_video_resolution(clip + '.missing')\n"
        "except IOError:\n"
        "    pass\n"
    )
    assert loaded == "[]", f"ML stack imported by an API helper: {loaded}"


def test_pipeline_import_does_not_build_models():
    # main needs yt_dlp, tqdm and friends, which the thin CI install lacks.
    for dep in ("yt_dlp", "tqdm", "dotenv"):
        pytest.importorskip(dep)
    loaded = _loaded_after("import main")
    assert loaded == "[]", f"importing main loaded the ML stack: {loaded}"
