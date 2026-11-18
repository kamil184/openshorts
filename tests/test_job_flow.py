"""The real ``main.py`` job, end to end on a tiny source (see job_harness)."""
import json
import signal

import pytest

import job_harness as jh
import presets

pytest.importorskip("main")
pytestmark = pytest.mark.skipif(not jh.have_tools(), reason="needs ffmpeg")

PRESET = presets.validate_preset({"captions": {
    "required_tags": ["#tag1", "#tag2"], "mention_one_of": ["CreatorOne"]}})


@pytest.fixture()
def model():
    m = jh.FakeModel([jh.detail_clip()])
    yield m
    m.close()


def _lines(proc):
    for raw in iter(proc.stdout.readline, b""):
        yield raw.decode("utf-8", "replace").rstrip()


def test_required_tags_are_in_the_first_metadata_the_job_writes(model, tmp_path):
    # The dashboard reads metadata.json while clips render; a job that dies
    # mid-render must not leave descriptions without the campaign's tags.
    env = jh.job_env(model, CLIP_PRESET_JSON=presets.preset_to_json(PRESET))
    proc, out = jh.start_job(tmp_path, env)
    seen = []
    try:
        for line in _lines(proc):
            seen.append(line)
            if line.startswith("   Saved metadata to"):
                proc.send_signal(signal.SIGKILL)
                break
    finally:
        proc.wait(timeout=60)
    assert any(l.startswith("   Saved metadata to") for l in seen), "\n".join(seen)
    clip = jh.metadata(out)["shorts"][0]
    assert clip["video_description_for_tiktok"] == "what a round #tag1 #tag2"
    assert clip["video_description_for_instagram"] == "what a round #tag1 #tag2"
    assert clip["caption_issues"]


def test_a_finished_job_keeps_tags_and_flags_missing_mentions(model, tmp_path):
    env = jh.job_env(model, CLIP_PRESET_JSON=presets.preset_to_json(PRESET))
    proc, out = jh.start_job(tmp_path, env)
    log = "\n".join(_lines(proc))
    assert proc.wait(timeout=300) == 0, log
    clip = jh.metadata(out)["shorts"][0]
    assert clip["video_description_for_tiktok"] == "what a round #tag1 #tag2"
    assert any("video_title_for_youtube_short" in i for i in clip["caption_issues"])
    assert (out / "Source_clip_1.mp4").exists()
    assert json.dumps(clip).count("#tag1") == 2


def test_the_job_records_its_clip_format_for_later_renders(model, tmp_path):
    env = jh.job_env(model, CLIP_FORMAT="gaming_stack", GAMING_FACECAM_BOX="0,0.5,0.3,0.4")
    proc, out = jh.start_job(tmp_path, env)
    log = "\n".join(_lines(proc))
    assert proc.wait(timeout=300) == 0, log
    meta = jh.metadata(out)
    assert meta["clip_format"] == "gaming_stack"
    assert meta["facecam_box"] == [0.0, 0.5, 0.3, 0.4]


def test_hook_grounding_on_a_gaming_stack_follows_the_preset(tmp_path):
    preset = presets.validate_preset({"captions": {
        "language": "en", "required_tags": ["#tag1"], "mention_one_of": ["CreatorOne"]}})
    model = jh.FakeModel(
        [jh.detail_clip(end=13.0, video_title_for_youtube_short="CreatorOne big round")],
        grounded={"on_screen": "a storm circle closing",
                  "viral_hook_text": "Der Sturm schliesst sich",
                  "video_title_for_youtube_short": "Der letzte Kreis"})
    try:
        env = jh.job_env(model, CLIP_FORMAT="gaming_stack", GAMING_FACECAM_BOX="0,0.5,0.3,0.4",
                         CLIP_PRESET_JSON=presets.preset_to_json(preset))
        proc, out = jh.start_job(tmp_path, env, output_format="vertical")
        log = "\n".join(_lines(proc))
        assert proc.wait(timeout=600) == 0, log
    finally:
        model.close()
    grounding = [p for p in model.prompts if "`on_screen`" in p]
    assert grounding, log
    assert "MUST be written in en" in grounding[0]
    assert "CreatorOne" in grounding[0]
    clip = jh.metadata(out)["shorts"][0]
    assert clip["viral_hook_text"] == "Der Sturm schliesst sich"
    # The rewritten title dropped the required mention: the old one stays.
    assert clip["video_title_for_youtube_short"] == "CreatorOne big round"
    assert "caption_issues" not in clip or not any(
        "video_title_for_youtube_short" in i for i in clip["caption_issues"])
