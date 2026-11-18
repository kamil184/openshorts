"""Transient failures retry inside the scoring/detail stage; policy blocks do not."""
import httpx
import pytest

import llm_backend
import prompts

main = pytest.importorskip("main")


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(main.time, "sleep", lambda *_: None)


def _stage(monkeypatch, blips, error=None):
    calls = []

    def generate(prompt, schema, *, model):
        calls.append((prompt, schema, model))
        if len(calls) <= blips:
            raise error or ValueError("The model returned an empty response body.")
        return {"windows": [{"id": "w0", "start": 0, "end": 90,
                              "score": 90, "reason": "hook"}]}, None

    monkeypatch.setattr(llm_backend, "generate_json", generate)
    return calls


@pytest.mark.parametrize("blips", [1, 2])
def test_recovers_from_empty_bodies(monkeypatch, blips):
    calls = _stage(monkeypatch, blips)
    parsed, cost = main._run_model_stage("m", "prompt", prompts.ScoreResponse)
    assert len(calls) == blips + 1
    assert parsed["windows"][0]["score"] == 90
    assert cost is None
    assert all(call == ("prompt", prompts.ScoreResponse, "m") for call in calls)


def test_gives_up_after_three_attempts(monkeypatch):
    calls = _stage(monkeypatch, 99)
    with pytest.raises(ValueError, match="empty response body"):
        main._run_model_stage("m", "prompt", prompts.ScoreResponse)
    assert len(calls) == 3


def test_non_transient_errors_are_not_retried(monkeypatch):
    calls = _stage(monkeypatch, 99, ValueError("400 INVALID_ARGUMENT: bad request"))
    with pytest.raises(ValueError, match="400 INVALID_ARGUMENT"):
        main._run_model_stage("m", "prompt", prompts.ScoreResponse)
    assert len(calls) == 1


def test_succeeds_on_the_first_call(monkeypatch):
    calls = _stage(monkeypatch, 0)
    parsed, _ = main._run_model_stage("m", "prompt", prompts.ScoreResponse)
    assert parsed["windows"][0]["score"] == 90
    assert len(calls) == 1


def test_policy_block_fails_fast(monkeypatch):
    calls = _stage(monkeypatch, 99, llm_backend.BlockedError("PROHIBITED_CONTENT"))
    with pytest.raises(llm_backend.BlockedError, match="PROHIBITED_CONTENT"):
        main._run_model_stage("m", "prompt", prompts.ScoreResponse)
    assert len(calls) == 1


@pytest.mark.parametrize("error", [httpx.ConnectError("refused"),
                                  httpx.ReadTimeout("timed out"),
                                  RuntimeError("503 UNAVAILABLE"),
                                  RuntimeError("429 RESOURCE_EXHAUSTED"),
                                  ValueError("Failed to parse the model JSON response"),
                                  ValueError("1 validation error for ScoreResponse")])
def test_recovers_from_transport_failures(monkeypatch, error):
    calls = _stage(monkeypatch, 1, error)
    parsed, _ = main._run_model_stage("m", "prompt", prompts.ScoreResponse)
    assert parsed["windows"][0]["score"] == 90
    assert len(calls) == 2
