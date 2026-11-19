"""Unit tests for the deterministic fake LLM and E2E comparison helpers."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from urllib.request import Request, urlopen

import pytest

_E2E_DIR = str(Path(__file__).resolve().parent / "e2e")
if _E2E_DIR not in sys.path:
    sys.path.insert(0, _E2E_DIR)

import burned_subtitles  # noqa: E402
import prompts  # noqa: E402
from fake_llm import DEFAULT_MODEL_ID, FakeLLMServer, canned_response_for_request  # noqa: E402
from run_reference import (  # noqa: E402
    build_child_env,
    compare_job,
    compare_sidecars,
    parse_clip_ready_lines,
)


def _schema_req(schema_cls, prompt_text: str, *, images: int = 0) -> dict:
    name = schema_cls.__name__.lower()
    if images:
        content = [
            *({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AA=="}} for _ in range(images)),
            {"type": "text", "text": prompt_text},
        ]
    else:
        content = prompt_text
    return {
        "model": DEFAULT_MODEL_ID,
        "messages": [{"role": "user", "content": content}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": name, "schema": {}},
        },
    }


def test_fake_llm_server_endpoints_and_schemas():
    with FakeLLMServer() as srv:
        with urlopen(f"{srv.base_url}/models", timeout=5) as resp:
            models_payload = json.loads(resp.read().decode("utf-8"))
        mods = models_payload["data"][0]["architecture"]["input_modalities"]
        assert set(mods) == {"text", "image", "video"}

        score_prompt = (
            'WINDOWS_JSON:\n[{"id": "window_001", "start": 0.0, "end": 90.0}, '
            '{"id": "window_002", "start": 60.0, "end": 150.0}]\n\nReturn only:'
        )
        req_body = _schema_req(prompts.ScoreResponse, score_prompt)
        req = Request(
            f"{srv.base_url}/chat/completions",
            data=json.dumps(req_body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=5) as resp:
            chat_payload = json.loads(resp.read().decode("utf-8"))
        parsed_score = prompts.ScoreResponse.model_validate_json(
            chat_payload["choices"][0]["message"]["content"]
        )
        assert [w.id for w in parsed_score.windows] == ["window_001", "window_002"]


def test_fake_llm_canned_schemas_validate():
    detail_prompt = 'CANDIDATE_WINDOWS_JSON:\n[{"id": "window_001", "start": 0, "end": 90}]'
    detail = canned_response_for_request(_schema_req(prompts.DetailResponse, detail_prompt))
    validated = prompts.DetailResponse.model_validate(detail)
    assert len(validated.shorts) == 2

    floor_prompt = 'CANDIDATE_WINDOWS_JSON:\n[{"id": "window_002", "start": 60, "end": 150}]'
    floor = canned_response_for_request(_schema_req(prompts.DetailResponse, floor_prompt))
    assert prompts.DetailResponse.model_validate(floor).shorts == []

    prompts.LayoutChoice.model_validate(
        canned_response_for_request(_schema_req(prompts.LayoutChoice, "pick"))
    )
    prompts.GroundedHook.model_validate(
        canned_response_for_request(_schema_req(prompts.GroundedHook, "hook"))
    )
    prompts.ShotContentResponse.model_validate(
        canned_response_for_request(_schema_req(prompts.ShotContentResponse, "shots"))
    )

    from pydantic import BaseModel

    class FrameSubtitle(BaseModel):
        frame: int
        subtitle_top: float | None

    class SubtitleAnswer(BaseModel):
        frames: list[FrameSubtitle]

    sub_ans = canned_response_for_request(
        _schema_req(SubtitleAnswer, burned_subtitles.PROMPT, images=12)
    )
    assert len(SubtitleAnswer.model_validate(sub_ans).frames) == 12


def test_build_child_env_strips_secrets_and_sets_llm():
    env = build_child_env(
        llm_base_url="http://127.0.0.1:9999",
        llm_model="m1",
        asr_lock_dir="/tmp/locks",
        extra_env={"CLIP_FORMAT": "gaming_stack"},
        base_env={
            "PATH": "/bin",
            "GEMINI_API_KEY": "secret",
            "CLIP_PRESET": "other",
            "AUTO_LAYOUT": "1",
        },
    )
    assert "GEMINI_API_KEY" not in env
    assert "CLIP_PRESET" not in env
    assert "AUTO_LAYOUT" not in env
    assert env["LLM_BASE_URL"] == "http://127.0.0.1:9999"
    assert env["CLIP_FORMAT"] == "gaming_stack"
    assert env["AUTO_HOOK"] == "1"
    assert env["AUTO_CAPTIONS"] == "1"


def test_parse_clip_ready_lines():
    out = "hello\nCLIP_READY 0 subtitled_1_hooked_1_fixture_clip_1.mp4\nCLIP_READY 1 clip2.mp4\n"
    assert parse_clip_ready_lines(out) == {
        0: "subtitled_1_hooked_1_fixture_clip_1.mp4",
        1: "clip2.mp4",
    }


def test_compare_sidecars_detects_diffs():
    a = {"ranges": [{"start": 0.0, "end": 10.0, "layout": "track"}]}
    b = {"ranges": [{"start": 0.0, "end": 10.0, "layout": "track"}]}
    assert compare_sidecars(a, b) == []

    c = {"ranges": [{"start": 0.0, "end": 10.0, "layout": "wide"}]}
    diffs = compare_sidecars(a, c)
    assert any("layout differs" in line for line in diffs)


def test_compare_job_pass_and_fail(tmp_path: Path):
    ref_dir = tmp_path / "ref"
    (ref_dir / "clips").mkdir(parents=True)
    ref_sidecar = ref_dir / "clips" / "clip_1.layout.json"
    ref_sidecar.write_text('{"ranges": [{"start": 0.0, "end": 25.0, "layout": "track"}]}')
    ref_video = ref_dir / "clips" / "clip_1.mp4"
    ref_video.write_bytes(b"ref")
    (ref_dir / "manifest.json").write_text(
        json.dumps({
            "job": "track",
            "clips": [
                {
                    "index": 1,
                    "start": 20.0,
                    "end": 45.0,
                    "sidecar": "clips/clip_1.layout.json",
                    "video": "clips/clip_1.mp4",
                }
            ],
        })
    )

    actual_video = tmp_path / "actual.mp4"
    actual_video.write_bytes(b"act")
    actual = {
        "clips": [
            {
                "index": 1,
                "start": 20.05,
                "end": 44.95,
                "sidecar": {"ranges": [{"start": 0.0, "end": 25.0, "layout": "track"}]},
                "video_path": actual_video,
            }
        ]
    }
    ok, lines = compare_job("track", actual, ref_dir, ssim_fn=lambda a, b: (0.999, 0.995, 25))
    assert ok
    assert "PASS" in lines[0]

    ok_low, lines_low = compare_job("track", actual, ref_dir, ssim_fn=lambda a, b: (0.91, 0.85, 25))
    assert not ok_low
    assert "FAIL" in lines_low[0] and "SSIM too low" in lines_low[0]

    bad_bounds = {
        "clips": [
            {
                **actual["clips"][0],
                "end": 46.5,
            }
        ]
    }
    ok_bounds, lines_bounds = compare_job(
        "track", bad_bounds, ref_dir, ssim_fn=lambda a, b: (0.999, 0.999, 25)
    )
    assert not ok_bounds
    assert "boundaries mismatch" in lines_bounds[0]
