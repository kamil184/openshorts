"""The model facade (llm_backend.py) and its OpenAI-compatible provider.

Every stage asks for a task ("text", "vision", "video") and passes media;
which model answers is configuration, and what it can take is looked up, not
inferred from its name. These tests pin that contract and the provider's
request shape: schema validation, the response_format ladder, inline media,
gateway cost and the capability lookup.
"""
import json

import httpx
import pytest

import llm_backend
import llm_openai
import prompts

MODELS = {"data": [
    {"id": "meta/muse-ember",
     "architecture": {"input_modalities": ["text", "image", "video", "file", "audio"]}},
    {"id": "openai/gpt-6-luna", "architecture": {"input_modalities": ["file", "image", "text"]}},
    {"id": "qwen2.5:14b"},
]}


@pytest.fixture
def local(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://llm.test/v1")
    monkeypatch.setenv("LLM_MODEL", "qwen2.5:14b")
    for name in ("LLM_PROVIDER", "LLM_VISION_MODEL", "LLM_VIDEO_MODEL",
                 "LLM_INPUT_MODALITIES", "LLM_EXTRA_BODY", "GEMINI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(llm_openai, "_modalities_cache", None)


def _serve(handler, monkeypatch):
    """Route the provider's httpx client through an in-process handler that
    also answers the model list."""
    def route(request):
        if request.method == "GET" and request.url.path.endswith("/models"):
            return httpx.Response(200, json=MODELS)
        return handler(request)

    transport = httpx.MockTransport(route)
    monkeypatch.setattr(llm_openai, "_client",
                        lambda **kw: httpx.Client(transport=transport, **kw))


def _completion(payload, usage=None):
    return httpx.Response(200, json={
        "choices": [{"message": {"role": "assistant", "content": json.dumps(payload)}}],
        "usage": usage or {"prompt_tokens": 120, "completion_tokens": 40},
    })


def _capture(seen, payload=None):
    def handler(request):
        seen["body"] = json.loads(request.content)
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        return _completion(payload or {"windows": []})
    return handler


# --- provider selection -------------------------------------------------------

def test_nothing_configured_means_no_provider(monkeypatch):
    for name in ("LLM_BASE_URL", "LLM_PROVIDER", "GEMINI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    assert llm_backend.available() is False
    assert llm_backend.describe() is None
    with pytest.raises(RuntimeError, match="No model configured"):
        llm_backend.generate_json("p", prompts.ScoreResponse)


def test_base_url_selects_the_openai_compatible_provider(local, monkeypatch):
    monkeypatch.setenv("LLM_VISION_MODEL", "meta/muse-ember")
    assert llm_backend.provider_name() == "openai"
    assert llm_backend.describe() == {
        "provider": "openai", "model": "qwen2.5:14b", "baseUrl": "http://llm.test/v1",
        "visionModel": "meta/muse-ember", "videoModel": "meta/muse-ember"}


def test_a_gemini_key_alone_selects_gemini(monkeypatch):
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    assert llm_backend.provider_name() == "gemini"
    assert llm_backend.model_for(llm_backend.TEXT) == "gemini-3.1-flash-lite"
    assert llm_backend.supports("gemini-3.1-flash-lite", "video")


def test_explicit_provider_wins_over_the_other_configured_one(local, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    assert llm_backend.provider_name() == "openai"
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    assert llm_backend.provider_name() == "gemini"


# --- models per task ------------------------------------------------------------

def test_vision_and_video_models_fall_back_to_the_text_model(local, monkeypatch):
    assert llm_backend.model_for(llm_backend.VISION) == "qwen2.5:14b"
    assert llm_backend.model_for(llm_backend.VIDEO) == "qwen2.5:14b"

    monkeypatch.setenv("LLM_VISION_MODEL", "meta/muse-ember")
    assert llm_backend.model_for(llm_backend.VIDEO) == "meta/muse-ember"
    monkeypatch.setenv("LLM_VIDEO_MODEL", "openai/gpt-6-luna")
    assert llm_backend.model_for(llm_backend.VIDEO) == "openai/gpt-6-luna"
    with pytest.raises(ValueError):
        llm_backend.model_for("audio")


def test_task_picks_the_model_that_is_sent(local, monkeypatch):
    monkeypatch.setenv("LLM_VISION_MODEL", "meta/muse-ember")
    seen = {}
    _serve(_capture(seen), monkeypatch)
    llm_backend.generate_json("p", prompts.ScoreResponse, task=llm_backend.VISION, media=[b"jpg"])
    assert seen["body"]["model"] == "meta/muse-ember"


# --- capabilities ---------------------------------------------------------------

def test_capabilities_come_from_the_server_model_list(local, monkeypatch):
    _serve(lambda r: _completion({}), monkeypatch)
    assert llm_backend.supports("meta/muse-ember", "video")
    assert llm_backend.supports("openai/gpt-6-luna", "image")
    assert not llm_backend.supports("openai/gpt-6-luna", "video")
    # Listed without modalities, or not listed at all: text only.
    assert llm_backend.input_modalities("qwen2.5:14b") == {"text"}
    assert llm_backend.input_modalities("unknown/model") == {"text"}


def test_declared_modalities_cover_servers_that_do_not_publish_them(local, monkeypatch):
    monkeypatch.setenv("LLM_INPUT_MODALITIES", json.dumps({"llava:13b": ["image"]}))
    _serve(lambda r: _completion({}), monkeypatch)
    assert llm_backend.supports("llava:13b", "image")
    assert llm_backend.supports("llava:13b", "text")


def test_an_unreachable_model_list_means_text_only(local, monkeypatch):
    def boom(request):
        raise httpx.ConnectError("refused")
    monkeypatch.setattr(llm_openai, "_client",
                        lambda **kw: httpx.Client(transport=httpx.MockTransport(boom), **kw))
    assert llm_backend.input_modalities("meta/muse-ember") == {"text"}


def test_image_model_is_none_when_the_vision_model_cannot_see(local, monkeypatch):
    _serve(lambda r: _completion({}), monkeypatch)
    assert llm_backend.image_model() is None
    monkeypatch.setenv("LLM_VISION_MODEL", "openai/gpt-6-luna")
    assert llm_backend.image_model() == "openai/gpt-6-luna"


def test_media_the_model_cannot_take_is_refused_before_the_call(local, monkeypatch, tmp_path):
    calls = []
    _serve(lambda r: calls.append(r) or _completion({"windows": []}), monkeypatch)
    clip = tmp_path / "c.mp4"
    clip.write_bytes(b"mp4")
    with pytest.raises(ValueError, match="does not take video"):
        llm_backend.generate_json("p", prompts.ScoreResponse, model="openai/gpt-6-luna",
                                  media=[llm_backend.Video(str(clip))])
    assert calls == []


# --- the OpenAI-compatible request -------------------------------------------

def test_returns_validated_payload_and_zero_cost(local, monkeypatch):
    seen = {}
    _serve(_capture(seen, {"windows": [
        {"id": "w0", "start": 0, "end": 90, "score": 88, "reason": "hook"}]}), monkeypatch)
    parsed, cost = llm_backend.generate_json("prompt", prompts.ScoreResponse)

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
        fmt = (json.loads(request.content).get("response_format") or {}).get("type")
        formats.append(fmt)
        if fmt == "json_schema":
            return httpx.Response(400, json={"error": {"message": "response_format json_schema not supported"}})
        return _completion({"windows": []})

    _serve(handler, monkeypatch)
    parsed, _ = llm_backend.generate_json("prompt", prompts.ScoreResponse)
    assert formats == ["json_schema", "json_object"]
    assert parsed == {"windows": []}


def test_temperature_is_dropped_when_no_endpoint_accepts_it(local, monkeypatch):
    bodies = []

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        if "temperature" in body:
            return httpx.Response(404, json={"error": {
                "message": "No endpoints found that can handle the requested parameters."}})
        return _completion({"windows": []})

    _serve(handler, monkeypatch)
    llm_backend.generate_json("p", prompts.ScoreResponse)

    assert "temperature" in bodies[0]
    assert "temperature" not in bodies[1]
    assert bodies[1]["response_format"]["type"] == "json_schema"


def test_code_fenced_json_is_accepted(local, monkeypatch):
    """Small models wrap the object in ```json fences even when told not to."""
    text = "```json\n" + json.dumps({"windows": []}) + "\n```"
    _serve(lambda r: httpx.Response(200, json={"choices": [{"message": {"content": text}}],
                                               "usage": {}}), monkeypatch)
    parsed, _ = llm_backend.generate_json("prompt", prompts.ScoreResponse)
    assert parsed == {"windows": []}


def test_schema_violation_raises_instead_of_leaking_into_the_pipeline(local, monkeypatch):
    _serve(lambda r: _completion({"windows": [{"id": "w0"}]}), monkeypatch)
    with pytest.raises(Exception) as exc:
        llm_backend.generate_json("prompt", prompts.ScoreResponse)
    assert "validation error" in str(exc.value)


def test_server_errors_surface_with_status_and_url(local, monkeypatch):
    _serve(lambda r: httpx.Response(503, text="loading model"), monkeypatch)
    with pytest.raises(RuntimeError) as exc:
        llm_backend.generate_json("prompt", prompts.ScoreResponse)
    assert "503" in str(exc.value) and "loading model" in str(exc.value)


def test_extra_body_is_merged_but_cannot_override_the_request(local, monkeypatch):
    monkeypatch.setenv("LLM_EXTRA_BODY", json.dumps({
        "reasoning": {"effort": "low"},
        "provider": {"require_parameters": True},
        "model": "ignored/model",
    }))
    seen = {}
    _serve(_capture(seen), monkeypatch)
    llm_backend.generate_json("prompt", prompts.ScoreResponse)

    assert seen["body"]["reasoning"] == {"effort": "low"}
    assert seen["body"]["provider"] == {"require_parameters": True}
    assert seen["body"]["model"] == "qwen2.5:14b"


def test_extra_body_that_is_not_an_object_fails_loudly(local, monkeypatch):
    monkeypatch.setenv("LLM_EXTRA_BODY", "[1, 2]")
    _serve(lambda request: _completion({"windows": []}), monkeypatch)
    with pytest.raises(ValueError, match="JSON object"):
        llm_backend.generate_json("prompt", prompts.ScoreResponse)


def test_media_parts_keep_order_and_put_the_prompt_last(local, monkeypatch, tmp_path):
    clip = tmp_path / "proxy.mp4"
    clip.write_bytes(b"mp4")
    seen = {}
    _serve(_capture(seen), monkeypatch)
    llm_backend.generate_json("the prompt", prompts.ScoreResponse, model="meta/muse-ember",
                              media=["Shot 0:", b"\xff\xd8jpeg", llm_backend.Video(str(clip))])

    parts = seen["body"]["messages"][1]["content"]
    assert [p["type"] for p in parts] == ["text", "image_url", "video_url", "text"]
    assert parts[0]["text"] == "Shot 0:"
    assert parts[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert parts[2]["video_url"]["url"] == "data:video/mp4;base64,bXA0"
    assert parts[3]["text"] == "the prompt"


def test_text_only_requests_keep_a_plain_string_content(local, monkeypatch):
    seen = {}
    _serve(_capture(seen), monkeypatch)
    llm_backend.generate_json("just text", prompts.ScoreResponse)
    assert seen["body"]["messages"][1]["content"] == "just text"


def test_unsupported_media_part_is_rejected():
    with pytest.raises(TypeError):
        llm_openai._content("p", [123])


def test_gateway_cost_is_reported_when_the_server_bills(local, monkeypatch):
    _serve(lambda request: _completion(
        {"windows": []},
        usage={"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.0012,
               "completion_tokens_details": {"reasoning_tokens": 3}}), monkeypatch)

    _, cost = llm_backend.generate_json("p", prompts.ScoreResponse)

    assert cost["total_cost"] == pytest.approx(0.0012)
    assert cost["thinking_tokens"] == 3
    assert cost["local"] is False


# --- stages that look at frames -------------------------------------------------

def test_layout_picker_asks_the_vision_model(local, monkeypatch):
    import layout_picker

    monkeypatch.setenv("LLM_VISION_MODEL", "meta/muse-ember")
    monkeypatch.setattr(layout_picker, "ENABLED", True)
    monkeypatch.setattr(layout_picker, "sample_frames", lambda path: [b"a", b"b"])
    _serve(lambda r: _completion({}), monkeypatch)
    asked = {}

    def fake_generate(prompt, schema, model=None, media=None, **kw):
        asked.update(model=model, media=media, schema=schema)
        return {"layout": sorted(layout_picker.VALID - {"none"})[0], "why": "x",
                "confidence": 0.9}, None

    monkeypatch.setattr(llm_backend, "generate_json", fake_generate)

    decision = layout_picker.pick("video.mp4", 60)

    assert decision in layout_picker.VALID and decision != "none"
    assert asked == {"model": "meta/muse-ember", "media": [b"a", b"b"],
                     "schema": prompts.LayoutChoice}


def test_layout_picker_skips_a_model_that_cannot_see(local, monkeypatch):
    import layout_picker

    monkeypatch.setattr(layout_picker, "ENABLED", True)
    _serve(lambda r: _completion({}), monkeypatch)
    monkeypatch.setattr(llm_backend, "generate_json",
                        lambda *a, **k: pytest.fail("a text-only model must not get frames"))
    assert layout_picker.pick("video.mp4", 60) == "none"


def test_layout_picker_falls_back_to_none_when_the_call_fails(local, monkeypatch):
    import layout_picker

    monkeypatch.setenv("LLM_VISION_MODEL", "meta/muse-ember")
    monkeypatch.setattr(layout_picker, "ENABLED", True)
    monkeypatch.setattr(layout_picker, "sample_frames", lambda path: [b"a"])
    _serve(lambda r: _completion({}), monkeypatch)

    def boom(*args, **kwargs):
        raise RuntimeError("gateway down")

    monkeypatch.setattr(llm_backend, "generate_json", boom)
    assert layout_picker.pick("video.mp4", 60) == "none"


# --- main.py (needs the heavy deps; skipped on minimal CI) ----------------------

def test_stage_retries_transient_errors(local, monkeypatch):
    main = pytest.importorskip("main")
    monkeypatch.setattr(main.time, "sleep", lambda *_: None)
    calls = {"n": 0}

    def fake_generate(prompt, schema, model=None, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("connection refused")
        return {"windows": []}, {"total_cost": 0.0}

    monkeypatch.setattr(llm_backend, "generate_json", fake_generate)
    parsed, cost = main._run_model_stage("qwen2.5:14b", "prompt", prompts.ScoreResponse)
    assert calls["n"] == 2
    assert parsed == {"windows": []} and cost["total_cost"] == 0.0


def test_score_batch_follows_the_provider(local, monkeypatch):
    main = pytest.importorskip("main")
    monkeypatch.delenv("LLM_SCORE_BATCH", raising=False)
    assert main.score_batch_size() == 3
    monkeypatch.setenv("LLM_SCORE_BATCH", "5")
    assert main.score_batch_size() == 5
    monkeypatch.delenv("LLM_BASE_URL")
    monkeypatch.delenv("LLM_SCORE_BATCH")
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    assert main.score_batch_size() == 8


@pytest.mark.parametrize("model, mode", [
    ("meta/muse-ember", "video"), ("openai/gpt-6-luna", "frames"), ("qwen2.5:14b", None)])
def test_silent_footage_input_follows_what_the_model_takes(local, monkeypatch, model, mode):
    main = pytest.importorskip("main")
    monkeypatch.delenv("VISION_SEGMENT_SECONDS", raising=False)
    monkeypatch.delenv("VISION_MAX_SEGMENTS", raising=False)
    _serve(lambda r: _completion({}), monkeypatch)
    plan = main.vision_plan(model)
    if mode is None:
        assert plan is None
        return
    got_mode, segment, max_segments = plan
    assert got_mode == mode
    if mode == "frames":
        assert segment == main.VISION_FRAME_INTERVAL * main.VISION_MAX_FRAMES
    else:
        assert segment == llm_openai.VIDEO_SEGMENT_SECONDS
    # The same total span is watched whatever the stretch size.
    assert segment * max_segments >= main.VISION_MAX_SECONDS
