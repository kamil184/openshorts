"""A job runs with the preset it was submitted with.

The API loads and validates the preset once, at request time, and hands its
content to the job (``CLIP_PRESET_JSON``), so an edit or a delete while the
job waits in the queue, or a remote worker without the presets directory,
cannot change or break it. A preset that cannot be loaded fails loudly
instead of turning into "the model returned no clips".
"""
import asyncio
import json

import httpx
import pytest

import presets
import remote_worker

PRESET_YAML = (
    "clip_format: gaming_stack\n"
    "target_clips: 4\n"
    "captions:\n"
    "  language: en\n"
    "  required_tags: ['#tag1']\n"
    "  mention_one_of: [CreatorOne]\n"
)


@pytest.fixture()
def app_env(tmp_path, monkeypatch):
    app_module = pytest.importorskip("app")
    for name in ("output", "uploads", "presets"):
        (tmp_path / name).mkdir()
    monkeypatch.setattr(app_module, "OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setattr(app_module, "UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("PRESETS_DIR", str(tmp_path / "presets"))

    async def fake_put(item):
        pass

    monkeypatch.setattr(app_module.job_queue, "put", fake_put)
    (tmp_path / "presets" / "camp.yaml").write_text(PRESET_YAML, encoding="utf-8")
    return app_module, tmp_path / "presets"


def _submit(app_module):
    async def _do():
        transport = httpx.ASGITransport(app=app_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
            return await c.post("/api/process",
                                data={"acknowledged": "true", "preset": "camp"},
                                files={"file": ("clip.mp4", b"x" * 100, "video/mp4")},
                                headers={"X-Gemini-Key": "test-key"})
    return asyncio.run(_do())


def test_the_job_keeps_the_submitted_preset_after_the_file_changes(app_env, monkeypatch):
    app_module, presets_dir = app_env
    resp = _submit(app_module)
    assert resp.status_code == 200, resp.text
    env = app_module.jobs.pop(resp.json()["job_id"])["env"]
    submitted = presets.load_preset("camp", directory=str(presets_dir))
    assert json.loads(env[presets.PRESET_JSON_ENV]) == submitted

    (presets_dir / "camp.yaml").unlink()
    monkeypatch.delenv("PRESETS_DIR")
    assert presets.load_active_preset(env) == submitted


def test_the_preset_content_reaches_a_remote_worker(app_env, monkeypatch):
    app_module, presets_dir = app_env
    resp = _submit(app_module)
    env = app_module.jobs.pop(resp.json()["job_id"])["env"]
    options = remote_worker.job_options(env, {})
    assert options[presets.PRESET_JSON_ENV] == env[presets.PRESET_JSON_ENV]

    worker = pytest.importorskip("worker")
    monkeypatch.delenv("PRESETS_DIR")
    worker_env = worker.job_env(options)
    assert presets.load_active_preset(worker_env) == presets.load_preset(
        "camp", directory=str(presets_dir))


@pytest.mark.parametrize("raw", [
    "not json",
    json.dumps({"bogus": 1}),
    json.dumps({"captions": {"required_tags": None}}),
    "x" * (presets.PRESET_JSON_MAX_BYTES + 1),
])
def test_a_bad_preset_payload_never_crosses_to_a_worker(raw):
    assert not remote_worker.allowed_option(presets.PRESET_JSON_ENV, raw)


def test_a_bad_preset_payload_fails_the_job(monkeypatch):
    with pytest.raises(presets.PresetValidationError):
        presets.load_active_preset({presets.PRESET_JSON_ENV: "{broken"})


def test_selection_does_not_hide_a_preset_that_cannot_be_loaded(monkeypatch):
    main = pytest.importorskip("main")
    monkeypatch.setenv("CLIP_PRESET", "gone")
    monkeypatch.delenv(presets.PRESET_JSON_ENV, raising=False)
    monkeypatch.delenv("PRESETS_DIR", raising=False)
    monkeypatch.setattr(main.llm_backend, "available", lambda: True)
    monkeypatch.setattr(main.llm_backend, "model_for", lambda task: "test-model")
    monkeypatch.setattr(main, "_run_model_stage", lambda model, prompt, schema: (
        {"windows": [{"id": "window_001", "start": 0, "end": 90, "score": 90, "reason": "r"}]}, None))
    transcript = {"language": "en", "segments": [
        {"start": 0.0, "end": 15.0, "text": "a few words here",
         "words": [{"word": "a", "start": 0.0, "end": 0.4}]}]}
    with pytest.raises(presets.PresetError):
        main.get_viral_clips(transcript, 60.0)
