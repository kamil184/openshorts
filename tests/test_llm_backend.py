"""The OpenAI-compatible moment-picker backend (llm_backend.py).

Self-hosters asked for a pipeline that never calls Google. The transcript
passes go to any /chat/completions server when LLM_BASE_URL is set; these
tests pin the contract main.py relies on: same (parsed, cost) shape as the
Gemini stage, schema validation, and the response_format fallback ladder.
"""
import json

import httpx
import pytest

import gemini_worker
import llm_backend


@pytest.fixture
def local(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://llm.test/v1")
    monkeypatch.setenv("LLM_MODEL", "qwen2.5:14b")
    monkeypatch.delenv("LLM_PROVIDER", raising=False)


def _serve(handler, monkeypatch):
    """Route llm_backend's httpx client through an in-process handler."""
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(llm_backend, "_client",
                        lambda **kw: httpx.Client(transport=transport, **kw))


def _completion(payload, usage=None):
    return httpx.Response(200, json={
        "choices": [{"message": {"role": "assistant", "content": json.dumps(payload)}}],
        "usage": usage or {"prompt_tokens": 120, "completion_tokens": 40},
    })


# --- activation -----------------------------------------------------------

def test_inactive_without_a_base_url(monkeypatch):
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert llm_backend.active() is False
    assert llm_backend.describe() is None


def test_base_url_alone_activates_and_describes(local):
    assert llm_backend.active() is True
    assert llm_backend.describe() == {
        "provider": "openai", "model": "qwen2.5:14b", "baseUrl": "http://llm.test/v1"}


def test_explicit_gemini_provider_wins_over_base_url(local, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    assert llm_backend.active() is False


# --- generate_json ---------------------------------------------------------

def test_returns_validated_payload_and_zero_cost(local, monkeypatch):
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        seen["auth"] = request.headers.get("authorization")
        return _completion({"windows": [{"id": "w0", "start": 0, "end": 90, "score": 88, "reason": "hook"}]})

    _serve(handler, monkeypatch)
    parsed, cost = llm_backend.generate_json("prompt", gemini_worker.ScoreResponse)

    assert seen["url"] == "http://llm.test/v1/chat/completions"
    assert seen["body"]["model"] == "qwen2.5:14b"
    assert seen["body"]["response_format"]["type"] == "json_schema"
    assert seen["auth"] == "Bearer ollama"  # documented placeholder when no key is set
    assert parsed["windows"][0]["score"] == 88
    assert cost["total_cost"] == 0.0 and cost["local"] is True
    assert cost["input_tokens"] == 120 and cost["output_tokens"] == 40


def test_falls_back_to_json_object_when_schema_mode_is_rejected(local, monkeypatch):
    formats = []

    def handler(request):
        body = json.loads(request.content)
        fmt = (body.get("response_format") or {}).get("type")
        formats.append(fmt)
        if fmt == "json_schema":
            return httpx.Response(400, json={"error": {"message": "response_format json_schema not supported"}})
        return _completion({"windows": []})

    _serve(handler, monkeypatch)
    parsed, _ = llm_backend.generate_json("prompt", gemini_worker.ScoreResponse)
    assert formats == ["json_schema", "json_object"]
    assert parsed == {"windows": []}


def test_code_fenced_json_is_accepted(local, monkeypatch):
    """Small models wrap the object in ```json fences even when told not to."""
    def handler(request):
        text = "```json\n" + json.dumps({"windows": []}) + "\n```"
        return httpx.Response(200, json={"choices": [{"message": {"content": text}}], "usage": {}})

    _serve(handler, monkeypatch)
    parsed, _ = llm_backend.generate_json("prompt", gemini_worker.ScoreResponse)
    assert parsed == {"windows": []}


def test_schema_violation_raises_instead_of_leaking_into_the_pipeline(local, monkeypatch):
    _serve(lambda r: _completion({"windows": [{"id": "w0"}]}), monkeypatch)
    with pytest.raises(Exception) as exc:
        llm_backend.generate_json("prompt", gemini_worker.ScoreResponse)
    assert "validation error" in str(exc.value)


def test_server_errors_surface_with_status_and_url(local, monkeypatch):
    _serve(lambda r: httpx.Response(503, text="loading model"), monkeypatch)
    with pytest.raises(RuntimeError) as exc:
        llm_backend.generate_json("prompt", gemini_worker.ScoreResponse)
    assert "503" in str(exc.value) and "loading model" in str(exc.value)


# --- main.py routing (needs the heavy deps; skipped on minimal CI) ------------

def test_stage_routes_to_local_backend_and_retries_transient(local, monkeypatch):
    main = pytest.importorskip("main")
    monkeypatch.setattr(main.time, "sleep", lambda *_: None)
    calls = {"n": 0}

    def fake_generate(prompt, schema, model=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("ConnectError: connection refused")
        return {"windows": [{"id": "w0", "start": 0, "end": 1, "score": 50, "reason": ""}]}, {"total_cost": 0.0}

    monkeypatch.setattr(llm_backend, "generate_json", fake_generate)
    parsed, cost = main._run_gemini_stage(None, "qwen2.5:14b", "prompt", gemini_worker.ScoreResponse)
    assert calls["n"] == 2
    assert parsed["windows"][0]["score"] == 50
    assert cost["total_cost"] == 0.0


def test_score_batch_shrinks_for_local_models(local, monkeypatch):
    main = pytest.importorskip("main")
    monkeypatch.delenv("LLM_SCORE_BATCH", raising=False)
    assert main.score_batch_size() == 3
    monkeypatch.setenv("LLM_SCORE_BATCH", "5")
    assert main.score_batch_size() == 5
    monkeypatch.delenv("LLM_BASE_URL")
    monkeypatch.delenv("LLM_SCORE_BATCH")
    assert main.score_batch_size() == 8


# --- provider-specific request fields ----------------------------------------

def test_extra_body_is_merged_but_cannot_override_the_request(local, monkeypatch):
    monkeypatch.setenv("LLM_EXTRA_BODY", json.dumps({
        "reasoning": {"effort": "low"},
        "provider": {"require_parameters": True},
        "model": "ignored/model",
    }))
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return _completion({"windows": []})

    _serve(handler, monkeypatch)
    llm_backend.generate_json("prompt", gemini_worker.ScoreResponse)

    assert seen["body"]["reasoning"] == {"effort": "low"}
    assert seen["body"]["provider"] == {"require_parameters": True}
    assert seen["body"]["model"] == "qwen2.5:14b"


def test_extra_body_that_is_not_an_object_fails_loudly(local, monkeypatch):
    monkeypatch.setenv("LLM_EXTRA_BODY", "[1, 2]")
    _serve(lambda request: _completion({"windows": []}), monkeypatch)

    with pytest.raises(ValueError, match="JSON object"):
        llm_backend.generate_json("prompt", gemini_worker.ScoreResponse)


# --- vision: images, video and model selection --------------------------------

def test_media_parts_keep_order_and_put_the_prompt_last(local, monkeypatch):
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return _completion({"windows": []})

    _serve(handler, monkeypatch)
    llm_backend.generate_json("the prompt", gemini_worker.ScoreResponse,
                              media=["Shot 0:", b"\xff\xd8jpeg", llm_backend.Video(b"mp4")])

    parts = seen["body"]["messages"][1]["content"]
    assert [p["type"] for p in parts] == ["text", "image_url", "video_url", "text"]
    assert parts[0]["text"] == "Shot 0:"
    assert parts[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert parts[2]["video_url"]["url"].startswith("data:video/mp4;base64,")
    assert parts[3]["text"] == "the prompt"


def test_text_only_requests_keep_a_plain_string_content(local, monkeypatch):
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return _completion({"windows": []})

    _serve(handler, monkeypatch)
    llm_backend.generate_json("just text", gemini_worker.ScoreResponse)

    assert seen["body"]["messages"][1]["content"] == "just text"


def test_unsupported_media_part_is_rejected():
    with pytest.raises(TypeError):
        llm_backend._content("p", [123])


def test_gateway_cost_is_reported_when_the_server_bills(local, monkeypatch):
    _serve(lambda request: _completion(
        {"windows": []},
        usage={"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.0012,
               "completion_tokens_details": {"reasoning_tokens": 3}}), monkeypatch)

    _, cost = llm_backend.generate_json("p", gemini_worker.ScoreResponse)

    assert cost["total_cost"] == pytest.approx(0.0012)
    assert cost["thinking_tokens"] == 3
    assert cost["local"] is False


def test_vision_and_video_models_fall_back_to_the_text_model(local, monkeypatch):
    monkeypatch.delenv("LLM_VISION_MODEL", raising=False)
    monkeypatch.delenv("LLM_VIDEO_MODEL", raising=False)
    assert llm_backend.vision_model() == "qwen2.5:14b"
    assert llm_backend.video_model() == "qwen2.5:14b"

    monkeypatch.setenv("LLM_VISION_MODEL", "meta/muse-ember")
    assert llm_backend.video_model() == "meta/muse-ember"
    monkeypatch.setenv("LLM_VIDEO_MODEL", "openai/gpt-6-luna")
    assert llm_backend.video_model() == "openai/gpt-6-luna"


def test_video_input_mode_is_validated(monkeypatch):
    monkeypatch.delenv("LLM_VIDEO_INPUT", raising=False)
    assert llm_backend.video_input_mode() == "video"
    monkeypatch.setenv("LLM_VIDEO_INPUT", "frames")
    assert llm_backend.video_input_mode() == "frames"
    monkeypatch.setenv("LLM_VIDEO_INPUT", "gif")
    with pytest.raises(ValueError):
        llm_backend.video_input_mode()


def test_layout_picker_asks_the_vision_model_without_a_gemini_key(local, monkeypatch):
    import layout_picker

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("LLM_VISION_MODEL", "meta/muse-ember")
    monkeypatch.setattr(layout_picker, "ENABLED", True)
    monkeypatch.setattr(layout_picker, "sample_frames", lambda path: [b"a", b"b"])
    asked = {}

    def fake_generate(prompt, schema, model=None, media=None):
        asked.update(model=model, media=media, schema=schema)
        return {"layout": sorted(layout_picker.VALID - {"none"})[0], "why": "x",
                "confidence": 0.9}, None

    monkeypatch.setattr(llm_backend, "generate_json", fake_generate)

    decision = layout_picker.pick("video.mp4", 60)

    assert decision in layout_picker.VALID and decision != "none"
    assert asked["model"] == "meta/muse-ember"
    assert asked["media"] == [b"a", b"b"]
    assert asked["schema"] is gemini_worker.LayoutChoice


def test_layout_picker_falls_back_to_none_when_the_vision_call_fails(local, monkeypatch):
    import layout_picker

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(layout_picker, "ENABLED", True)
    monkeypatch.setattr(layout_picker, "sample_frames", lambda path: [b"a"])

    def boom(*args, **kwargs):
        raise RuntimeError("gateway down")

    monkeypatch.setattr(llm_backend, "generate_json", boom)

    assert layout_picker.pick("video.mp4", 60) == "none"
