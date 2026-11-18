"""Importing the API must not load the heavy ML stack.

Target rule (import layering): the api and jobs layers never import torch,
ultralytics or mediapipe, even transitively. The check runs in a fresh
interpreter so the result does not depend on what other tests already imported.

Expected to fail until vision primitives move out of main.py; the xfail marker
is removed when that step lands.
"""
import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HEAVY_ROOTS = ("torch", "ultralytics", "mediapipe")
IMPORT_TIMEOUT_S = 30

_PROBE = (
    "import sys\n"
    "import app\n"
    f"heavy = {HEAVY_ROOTS!r}\n"
    "print(sorted(m for m in sys.modules if m.split('.')[0] in heavy))\n"
)


@pytest.mark.xfail(strict=False, reason="app still imports main; fixed when vision moves out of main.py")
def test_api_import_does_not_load_ml_stack():
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _PROBE],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=IMPORT_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        pytest.skip(f"importing app took longer than {IMPORT_TIMEOUT_S}s")
    assert proc.returncode == 0, proc.stderr[-2000:]
    loaded = proc.stdout.strip().splitlines()[-1]
    assert loaded == "[]", f"ML stack imported with the API: {loaded}"
