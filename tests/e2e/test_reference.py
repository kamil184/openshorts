"""Pytest entrypoint for the E2E reference verification suite."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_E2E_DIR = str(Path(__file__).resolve().parent)
if _E2E_DIR not in sys.path:
    sys.path.insert(0, _E2E_DIR)

from run_reference import run_suite  # noqa: E402

pytestmark = pytest.mark.e2e


def test_e2e_reference():
    fixture = os.environ.get("E2E_FIXTURE", "").strip()
    reference = os.environ.get("E2E_REFERENCE_DIR", "").strip()
    if not fixture or not reference:
        pytest.fail("Set E2E_FIXTURE and E2E_REFERENCE_DIR to run the e2e reference suite.")
    record = os.environ.get("E2E_RECORD", "").strip().lower() in ("1", "true", "yes")
    rc = run_suite(
        fixture_path=Path(fixture),
        reference_dir=Path(reference),
        record=record,
    )
    assert rc == 0
