"""Unit and HTTP tests for campaign presets (presets.py)."""
import asyncio
import os
import re

import httpx
import pytest

import presets
import prompts
import remote_worker

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _write_preset(tmp_path, name, content):
    path = tmp_path / f"{name}.yaml"
    path.write_text(content, encoding="utf-8")
    return path


class TestLoadAndValidate:
    def test_loads_example_preset(self):
        example_dir = os.path.join(REPO_ROOT, "presets.example")
        loaded = presets.load_preset("stream-gaming", directory=example_dir)
        assert loaded["clip_format"] == "gaming_stack"
        assert loaded["layouts"] == ["screencast"]
        assert loaded["target_clips"] == 6
        assert loaded["clip_seconds"] == [20, 60]
        assert loaded["auto_hook"] is True
        assert loaded["captions"]["language"] == "en"
        assert loaded["captions"]["required_tags"] == ["#tag1", "#tag2"]
        assert loaded["captions"]["mention_one_of"] == ["CreatorOne", "CreatorTwo"]

    def test_loads_yml_extension_and_normalizes_tags(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PRESETS_DIR", str(tmp_path))
        (tmp_path / "custom.yml").write_text(
            "captions:\n"
            "  required_tags: [tagA, '#TAGA', '#tagB']\n"
            "  mention_one_of: [Alpha, alpha, Beta]\n",
            encoding="utf-8",
        )
        loaded = presets.load_preset("custom")
        assert loaded["captions"]["required_tags"] == ["#tagA", "#tagB"]
        assert loaded["captions"]["mention_one_of"] == ["Alpha", "Beta"]

    def test_missing_dir_or_unknown_preset_raises(self, tmp_path, monkeypatch):
        monkeypatch.delenv("PRESETS_DIR", raising=False)
        with pytest.raises(presets.PresetNotFoundError):
            presets.load_preset("missing")

        monkeypatch.setenv("PRESETS_DIR", str(tmp_path))
        with pytest.raises(presets.PresetNotFoundError, match="Unknown preset"):
            presets.load_preset("missing")

    @pytest.mark.parametrize("bad_name", ["", "   ", "../secret", "a/b", "a\\b", ".hidden"])
    def test_rejects_unsafe_preset_names(self, tmp_path, bad_name):
        with pytest.raises(presets.PresetValidationError):
            presets.load_preset(bad_name, directory=str(tmp_path))

    @pytest.mark.parametrize(
        "yaml_body, expected_msg",
        [
            ("not: [valid", "Invalid YAML"),
            ("- item1\n- item2\n", "YAML mapping"),
            ("unknown_key: 1\n", "Unknown preset field"),
            ("captions:\n  unknown_subkey: 1\n", "Unknown preset 'captions'"),
            ("target_clips: 0\n", "target_clips"),
            ("target_clips: 20\n", "target_clips"),
            ("target_clips: true\n", "target_clips"),
            ("clip_seconds: [20]\n", "2-item"),
            ("clip_seconds: [60, 20]\n", "at least 5s"),
            ("clip_seconds: [2, 30]\n", "within"),
            ("clip_seconds: [true, 30]\n", "finite number"),
            ("auto_hook: 'yes'\n", "boolean"),
            ("layouts: screencast\n", "list"),
            ("clip_format: ''\n", "non-empty string"),
            ("facecam_box: [0, 0.3, 0.2]\n", "4-item"),
            ("facecam_box: [true, 0.3, 0.2, 0.3]\n", "finite number"),
            ("facecam_box: [-0.1, 0.3, 0.2, 0.3]\n", "normalised 0-1"),
            ("facecam_box: [0, 0.3, 0, 0.3]\n", "normalised 0-1"),
            ("facecam_box: [0.8, 0.3, 0.3, 0.3]\n", "exceed 1.0"),
            ("captions:\n  language: ''\n", "language"),
            ("captions:\n  required_tags: ['#bad tag']\n", "whitespace"),
            ("captions:\n  required_tags: ['#']\n", "empty '#'"),
            ("captions:\n  mention_one_of: ['']\n", "mention_one_of"),
        ],
    )
    def test_bad_preset_gives_clear_error(self, tmp_path, yaml_body, expected_msg):
        _write_preset(tmp_path, "bad", yaml_body)
        with pytest.raises(presets.PresetValidationError, match=expected_msg):
            presets.load_preset("bad", directory=str(tmp_path))

    def test_load_active_preset_from_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PRESETS_DIR", str(tmp_path))
        monkeypatch.delenv("CLIP_PRESET", raising=False)
        assert presets.load_active_preset() is None

        _write_preset(tmp_path, "demo", "target_clips: 4\nfacecam_box: [0, 0.377, 0.273, 0.305]\n")
        monkeypatch.setenv("CLIP_PRESET", "demo")
        assert presets.load_active_preset() == {
            "target_clips": 4,
            "facecam_box": [0.0, 0.377, 0.273, 0.305],
        }


class TestExpandJobDefaults:
    def test_expands_all_fields_and_explicit_overrides_win(self):
        preset = {
            "clip_format": "gaming_stack",
            "facecam_box": [0.0, 0.377, 0.273, 0.305],
            "layouts": ["screencast"],
            "target_clips": 6,
            "clip_seconds": [20, 60],
            "auto_hook": True,
        }
        expanded = presets.expand_job_defaults(preset, {
            "clip_format": None,
            "facecam_box": None,
            "layouts": None,
            "target_clips": "",
            "clip_min_seconds": None,
            "clip_max_seconds": None,
            "auto_hook": None,
        })
        assert expanded == {
            "clip_format": "gaming_stack",
            "facecam_box": [0.0, 0.377, 0.273, 0.305],
            "layouts": ["screencast"],
            "target_clips": 6,
            "clip_min_seconds": 20,
            "clip_max_seconds": 60,
            "auto_hook": True,
        }

        overridden = presets.expand_job_defaults(preset, {
            "clip_format": "custom_format",
            "facecam_box": [0.1, 0.1, 0.2, 0.2],
            "layouts": [],
            "target_clips": 3,
            "clip_min_seconds": 12,
            "clip_max_seconds": None,
            "auto_hook": False,
        })
        assert overridden == {
            "clip_format": "custom_format",
            "facecam_box": [0.1, 0.1, 0.2, 0.2],
            "layouts": [],
            "target_clips": 3,
            "clip_min_seconds": 12,
            "clip_max_seconds": 60,
            "auto_hook": False,
        }


class TestPromptFormatting:
    def test_without_preset_is_byte_identical(self):
        detail_base = prompts.DETAIL_PROMPT_TEMPLATE.format(
            video_duration=180,
            language="es",
            min_clips=3,
            max_clips=6,
            min_secs=15.0,
            max_secs=60.0,
            windows_json="[]",
        )
        detail_new = prompts.format_detail_prompt(
            video_duration=180,
            language="es",
            min_clips=3,
            max_clips=6,
            min_secs=15.0,
            max_secs=60.0,
            windows_json="[]",
            caption_rules="",
        )
        assert detail_new == detail_base

        visual_base = prompts.VISUAL_PROMPT_TEMPLATE.format(
            video_duration=180,
            language="es",
            min_clips=3,
            max_clips=6,
            min_secs=15.0,
            max_secs=60.0,
        )
        visual_new = prompts.format_visual_prompt(
            video_duration=180,
            language="es",
            min_clips=3,
            max_clips=6,
            min_secs=15.0,
            max_secs=60.0,
            caption_rules="",
            caption_language="",
        )
        assert visual_new == visual_base

    def test_with_preset_inserts_caption_rules(self):
        preset = {
            "captions": {
                "language": "en",
                "required_tags": ["#tag1", "#tag2"],
                "mention_one_of": ["CreatorOne", "CreatorTwo"],
                "extra_instructions": "Keep it positive.",
            }
        }
        rules = presets.caption_rules_prompt(preset)
        detail = prompts.format_detail_prompt(
            video_duration=180,
            language="es",
            min_clips=3,
            max_clips=6,
            min_secs=20.0,
            max_secs=60.0,
            windows_json="[]",
            caption_rules=rules,
        )
        assert "CAMPAIGN CAPTION RULES (mandatory):" in detail
        assert "#tag1 #tag2" in detail
        assert "CreatorOne, CreatorTwo" in detail
        assert "Keep it positive." in detail
        assert "written in en" in detail
        assert detail.index("CAMPAIGN CAPTION RULES") < detail.index("TRANSCRIPT_LANGUAGE:")
        assert re.findall(r"\{[a-z_]+\}", detail) == []

        visual = prompts.format_visual_prompt(
            video_duration=180,
            language="es",
            min_clips=3,
            max_clips=6,
            min_secs=20.0,
            max_secs=60.0,
            caption_rules=rules,
            caption_language=presets.caption_language(preset),
        )
        assert "catchy copy in en" in visual
        assert "CAMPAIGN CAPTION RULES (mandatory):" in visual
        assert "#tag1 #tag2" in visual


class TestDeterministicEnforcement:
    PRESET = {
        "captions": {
            "language": "en",
            "required_tags": ["#tag1", "#tag2"],
            "mention_one_of": ["CreatorOne", "CreatorTwo"],
        }
    }

    def test_appends_missing_tags_without_duplicating_or_touching_title(self):
        clip = {
            "video_description_for_tiktok": "CreatorOne clutched the round #TAG1",
            "video_description_for_instagram": "Wild finish with CreatorTwo #tag10",
            "video_title_for_youtube_short": "CreatorOne wins the match",
        }
        presets.enforce_clip(clip, self.PRESET)
        assert clip["video_description_for_tiktok"] == "CreatorOne clutched the round #TAG1 #tag2"
        assert clip["video_description_for_instagram"] == "Wild finish with CreatorTwo #tag10 #tag1 #tag2"
        assert clip["video_title_for_youtube_short"] == "CreatorOne wins the match"
        assert "caption_issues" not in clip

    def test_compliant_caption_is_left_untouched(self):
        original_tt = "CreatorOne wins it!  #tag2 #tag1  "
        original_ig = "creatorone clutched this #Tag1 #Tag2"
        original_yt = "CreatorTwo hits a wild shot"
        clip = {
            "video_description_for_tiktok": original_tt,
            "video_description_for_instagram": original_ig,
            "video_title_for_youtube_short": original_yt,
            "caption_issues": ["stale"],
        }
        presets.enforce_clip(clip, self.PRESET)
        assert clip["video_description_for_tiktok"] == original_tt
        assert clip["video_description_for_instagram"] == original_ig
        assert clip["video_title_for_youtube_short"] == original_yt
        assert "caption_issues" not in clip

    def test_missing_mention_is_flagged_without_inventing_text(self):
        clip = {
            "video_description_for_tiktok": "Insane clutch #tag1 #tag2",
            "video_description_for_instagram": "CreatorOne did it #tag1 #tag2",
            "video_title_for_youtube_short": "Crazy endgame moment",
        }
        presets.enforce_clip(clip, self.PRESET)
        assert clip["video_description_for_tiktok"] == "Insane clutch #tag1 #tag2"
        assert clip["video_title_for_youtube_short"] == "Crazy endgame moment"
        issues = clip["caption_issues"]
        assert len(issues) == 2
        assert any("video_description_for_tiktok" in item for item in issues)
        assert any("video_title_for_youtube_short" in item for item in issues)


class TestApiProcessIntegration:
    @pytest.fixture()
    def app_env(self, tmp_path, monkeypatch):
        app_module = pytest.importorskip("app")
        out_root = tmp_path / "output"
        up_root = tmp_path / "uploads"
        presets_dir = tmp_path / "presets"
        out_root.mkdir()
        up_root.mkdir()
        presets_dir.mkdir()
        monkeypatch.setattr(app_module, "OUTPUT_DIR", str(out_root))
        monkeypatch.setattr(app_module, "UPLOAD_DIR", str(up_root))
        monkeypatch.setenv("PRESETS_DIR", str(presets_dir))
        async def fake_put(item):
            pass
        monkeypatch.setattr(app_module.job_queue, "put", fake_put)
        _write_preset(
            presets_dir,
            "gaming",
            "clip_format: gaming_stack\n"
            "facecam_box: [0, 0.377, 0.273, 0.305]\n"
            "layouts: [screencast]\n"
            "target_clips: 6\n"
            "clip_seconds: [20, 60]\n"
            "auto_hook: true\n"
            "captions:\n"
            "  language: en\n"
            "  required_tags: ['#tag1', '#tag2']\n",
        )
        return app_module, presets_dir

    @staticmethod
    def _post(app_module, *, json_body=None, data=None, files=None):
        async def _do():
            transport = httpx.ASGITransport(app=app_module.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                return await client.post(
                    "/api/process",
                    json=json_body,
                    data=data,
                    files=files,
                    headers={"X-Gemini-Key": "test-key"},
                )
        return asyncio.run(_do())

    def test_unknown_preset_returns_400(self, app_env):
        app_module, _ = app_env
        resp = self._post(
            app_module,
            data={"acknowledged": "true", "preset": "does-not-exist"},
            files={"file": ("clip.mp4", b"x" * 100, "video/mp4")},
        )
        assert resp.status_code == 400
        assert "does-not-exist" in resp.json()["detail"]

    def test_preset_expands_defaults_and_sets_env(self, app_env):
        app_module, _ = app_env
        resp = self._post(
            app_module,
            data={"acknowledged": "true", "preset": "gaming"},
            files={"file": ("clip.mp4", b"x" * 100, "video/mp4")},
        )
        assert resp.status_code == 200, resp.text
        job = app_module.jobs.pop(resp.json()["job_id"])
        env = job["env"]
        assert env["CLIP_PRESET"] == "gaming"
        assert env["CLIP_FORMAT"] == "gaming_stack"
        assert env["GAMING_FACECAM_BOX"] == "0,0.377,0.273,0.305"
        assert env["SCREENCAST_LAYOUT"] == "1"
        assert env["CLIP_TARGET_MIN"] == "6" and env["CLIP_TARGET_MAX"] == "6"
        assert env["CLIP_MIN_SECONDS"] == "20.0" and env["CLIP_MAX_SECONDS"] == "60.0"
        assert env["AUTO_HOOK"] == "1"

    def test_explicit_fields_beat_preset_defaults(self, app_env, monkeypatch):
        app_module, _ = app_env
        slot = app_module.pending_uploads["u1"] = {
            "path": os.path.join(app_module.UPLOAD_DIR, "u1.mp4"),
            "filename": "u1.mp4",
            "complete": True,
            "user_id": None,
            "created_at": 10**12,
        }
        open(slot["path"], "wb").write(b"x" * 100)
        resp = self._post(
            app_module,
            json_body={
                "upload_id": "u1",
                "acknowledged": True,
                "preset": "gaming",
                "clip_format": "default",
                "layouts": [],
                "target_clips": 3,
                "clip_min_seconds": 10,
                "auto_hook": False,
            },
        )
        assert resp.status_code == 200, resp.text
        job = app_module.jobs.pop(resp.json()["job_id"])
        env = job["env"]
        assert env["CLIP_PRESET"] == "gaming"
        assert "CLIP_FORMAT" not in env
        assert "SCREENCAST_LAYOUT" not in env
        assert env["CLIP_TARGET_MIN"] == "3" and env["CLIP_TARGET_MAX"] == "3"
        assert env["CLIP_MIN_SECONDS"] == "10.0" and env["CLIP_MAX_SECONDS"] == "60.0"
        assert "AUTO_HOOK" not in env

    def test_only_max_below_the_preset_band_is_accepted(self, app_env):
        app_module, _ = app_env
        resp = self._post(
            app_module,
            data={"acknowledged": "true", "preset": "gaming", "clip_max_seconds": "12"},
            files={"file": ("clip.mp4", b"x" * 100, "video/mp4")},
        )
        assert resp.status_code == 200, resp.text
        env = app_module.jobs.pop(resp.json()["job_id"])["env"]
        assert env["CLIP_MAX_SECONDS"] == "12.0"
        assert float(env["CLIP_MIN_SECONDS"]) <= 7


def test_remote_worker_forwards_clip_preset():
    assert remote_worker.allowed_option("CLIP_PRESET", "stream-gaming")
    assert remote_worker.allowed_option("GAMING_FACECAM_BOX", "0,0.377,0.273,0.305")
    out = remote_worker.job_options(
        {
            "CLIP_PRESET": "stream-gaming",
            "GAMING_FACECAM_BOX": "0,0.377,0.273,0.305",
            "GEMINI_API_KEY": "secret",
        },
        {},
    )
    assert out == {
        "CLIP_PRESET": "stream-gaming",
        "GAMING_FACECAM_BOX": "0,0.377,0.273,0.305",
    }


class TestReviewEdgeCases:
    def test_one_explicit_clip_bound_stays_consistent_with_the_preset_band(self):
        preset = {"clip_seconds": [30, 60]}
        only_max = presets.expand_job_defaults(preset, {"clip_min_seconds": None, "clip_max_seconds": 20})
        assert only_max["clip_max_seconds"] == 20
        assert float(only_max["clip_min_seconds"]) <= 15
        only_min = presets.expand_job_defaults(preset, {"clip_min_seconds": 70, "clip_max_seconds": None})
        assert only_min["clip_min_seconds"] == 70
        assert float(only_min["clip_max_seconds"]) >= 75
        kept = presets.expand_job_defaults(preset, {"clip_min_seconds": 10, "clip_max_seconds": None})
        assert kept["clip_max_seconds"] == 60
        both = presets.expand_job_defaults(preset, {"clip_min_seconds": None, "clip_max_seconds": None})
        assert (both["clip_min_seconds"], both["clip_max_seconds"]) == (30, 60)

    @pytest.mark.parametrize("body", [
        "captions:\n  required_tags: #SypherPK #fortnite\n",
        "captions:\n  mention_one_of:\n",
        "captions:\n  language:\n",
    ])
    def test_a_caption_rule_that_parses_to_nothing_is_an_error(self, tmp_path, body):
        _write_preset(tmp_path, "broken", body)
        with pytest.raises(presets.PresetValidationError, match="empty"):
            presets.load_preset("broken", directory=str(tmp_path))

    def test_a_tag_glued_to_a_word_is_not_appended_twice(self):
        preset = {"captions": {"required_tags": ["#SypherPK"]}}
        clip = {"video_description_for_tiktok": "gg#SypherPK",
                "video_description_for_instagram": "nice #SypherPKfan"}
        presets.enforce_clip(clip, preset)
        assert clip["video_description_for_tiktok"] == "gg#SypherPK"
        assert clip["video_description_for_instagram"] == "nice #SypherPKfan #SypherPK"

    def test_an_absent_description_does_not_become_tag_only_text(self):
        preset = {"captions": {"required_tags": ["#tag1"]}}
        clip = {"video_description_for_tiktok": "hello"}
        presets.enforce_clip(clip, preset)
        assert clip["video_description_for_tiktok"] == "hello #tag1"
        assert "video_description_for_instagram" not in clip
