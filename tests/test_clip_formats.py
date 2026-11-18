"""Tests for clip_formats package, gaming_stack geometry, and API/main routing."""
import asyncio
import os

import httpx
import pytest

import clip_formats
from clip_formats import gaming_stack as gs
import gpu_render


class TestRegistry:
    def test_builtin_formats_registered(self):
        names = clip_formats.available()
        assert "default" in names
        assert "gaming_stack" in names

    def test_get_default_and_empty(self):
        assert clip_formats.get(None).name == "default"
        assert clip_formats.get("").name == "default"
        assert clip_formats.get("default").name == "default"
        assert clip_formats.get("GAMING_STACK").name == "gaming_stack"

    def test_unknown_format_raises(self):
        with pytest.raises(ValueError, match="Unknown clip format"):
            clip_formats.get("nonexistent_format")

    def test_from_env(self, monkeypatch):
        monkeypatch.delenv("CLIP_FORMAT", raising=False)
        assert clip_formats.from_env().name == "default"
        monkeypatch.setenv("CLIP_FORMAT", "gaming_stack")
        assert clip_formats.from_env().name == "gaming_stack"
        monkeypatch.setenv("CLIP_FORMAT", "bogus")
        with pytest.raises(ValueError):
            clip_formats.from_env()


class TestMainRouting:
    def test_default_env_uses_reframe_v2(self, monkeypatch, tmp_path):
        main = pytest.importorskip("main")
        reframe_v2 = pytest.importorskip("reframe_v2")
        monkeypatch.delenv("CLIP_FORMAT", raising=False)
        seen = {}
        monkeypatch.setattr(reframe_v2, "render",
                            lambda *a, **kw: seen.update(called=True, kw=kw) or True)
        ok = main.process_video_to_vertical(str(tmp_path / "in.mp4"), str(tmp_path / "out.mp4"),
                                            watermark=True)
        assert ok is True
        assert seen.get("called") is True

    def test_gaming_stack_env_hands_render_to_format(self, monkeypatch, tmp_path):
        main = pytest.importorskip("main")
        reframe_v2 = pytest.importorskip("reframe_v2")
        monkeypatch.setenv("CLIP_FORMAT", "gaming_stack")
        seen = {}
        fmt = clip_formats.get("gaming_stack")
        monkeypatch.setattr(fmt, "render",
                            lambda inp, out, ar, **kw: seen.update(
                                inp=inp, out=out, ar=ar, watermark=kw.get("watermark")) or True)
        monkeypatch.setattr(reframe_v2, "render",
                            lambda *a, **kw: pytest.fail("reframe_v2 should not run"))
        ok = main.process_video_to_vertical(str(tmp_path / "in.mp4"), str(tmp_path / "out.mp4"),
                                            aspect_ratio=9 / 16, watermark=True)
        assert ok is True
        assert seen["watermark"] is True
        assert seen["out"].endswith("out.mp4")

    @pytest.mark.parametrize("override_kw", [
        {"force_strategy": "WIDE"},
        {"crop_overrides": {0: 0.5}},
    ])
    def test_manual_editor_overrides_beat_clip_format(self, monkeypatch, tmp_path, override_kw):
        main = pytest.importorskip("main")
        reframe_v2 = pytest.importorskip("reframe_v2")
        monkeypatch.setenv("CLIP_FORMAT", "gaming_stack")
        seen = {}
        fmt = clip_formats.get("gaming_stack")
        monkeypatch.setattr(fmt, "render",
                            lambda *a, **kw: pytest.fail("CLIP_FORMAT must yield to editor override"))
        monkeypatch.setattr(reframe_v2, "render",
                            lambda *a, **kw: seen.update(kw) or True)
        ok = main.process_video_to_vertical(str(tmp_path / "in.mp4"), str(tmp_path / "out.mp4"),
                                            **override_kw)
        assert ok is True
        for k, v in override_kw.items():
            assert seen[k] == v


class TestAppClipFormat:
    def _post(self, app_module, json_body=None, data=None, files=None):
        async def _do():
            transport = httpx.ASGITransport(app=app_module.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
                return await c.post("/api/process", json=json_body, data=data, files=files,
                                    headers={"X-Gemini-Key": "test-key"})
        return asyncio.run(_do())

    def _stub_submit(self, app_module, monkeypatch, tmp_path):
        out_root = tmp_path / "output"
        up_root = tmp_path / "uploads"
        out_root.mkdir(exist_ok=True)
        up_root.mkdir(exist_ok=True)
        monkeypatch.setattr(app_module, "OUTPUT_DIR", str(out_root))
        monkeypatch.setattr(app_module, "UPLOAD_DIR", str(up_root))
        monkeypatch.setattr(app_module, "_media_duration_seconds", lambda p: 120.0)
        monkeypatch.setattr(app_module, "_enqueue_job", lambda jid, prio=0: None)

        async def _ok_url(url):
            return None

        async def _ok_probe(url):
            return {"max_height": 1080, "duration": 120}

        monkeypatch.setattr(app_module, "_validate_source_url", _ok_url)
        monkeypatch.setattr(app_module, "_probe_youtube_quality", _ok_probe)

    def test_unknown_clip_format_returns_400_json_and_form(self, monkeypatch, tmp_path):
        app_module = pytest.importorskip("app")
        self._stub_submit(app_module, monkeypatch, tmp_path)

        r_json = self._post(app_module, json_body={
            "url": "https://www.youtube.com/watch?v=abc",
            "acknowledged": True,
            "clip_format": "nope",
        })
        assert r_json.status_code == 400
        assert "clip_format" in r_json.json()["detail"]

        r_form = self._post(app_module, data={
            "url": "https://www.youtube.com/watch?v=abc",
            "acknowledged": "true",
            "clip_format": "nope",
        })
        assert r_form.status_code == 400
        assert "clip_format" in r_form.json()["detail"]

    def test_valid_clip_format_sets_child_env_json_and_form(self, monkeypatch, tmp_path):
        app_module = pytest.importorskip("app")
        self._stub_submit(app_module, monkeypatch, tmp_path)

        r_json = self._post(app_module, json_body={
            "url": "https://www.youtube.com/watch?v=abc",
            "acknowledged": True,
            "clip_format": "gaming_stack",
        })
        assert r_json.status_code == 200
        jid_json = r_json.json()["job_id"]
        assert app_module.jobs[jid_json]["env"]["CLIP_FORMAT"] == "gaming_stack"

        r_form = self._post(app_module, data={
            "url": "https://www.youtube.com/watch?v=def",
            "acknowledged": "true",
            "clip_format": "gaming_stack",
        })
        assert r_form.status_code == 200
        jid_form = r_form.json()["job_id"]
        assert app_module.jobs[jid_form]["env"]["CLIP_FORMAT"] == "gaming_stack"


class TestGamingStackGeometry:
    def test_band_heights_default_and_custom_ratio(self, monkeypatch):
        monkeypatch.delenv("GAMING_CAMERA_RATIO", raising=False)
        assert gs.band_heights(1920) == (576, 1344)

        monkeypatch.setenv("GAMING_CAMERA_RATIO", "0.25")
        assert gs.band_heights(1920) == (480, 1440)

        monkeypatch.setenv("GAMING_CAMERA_RATIO", "invalid")
        assert gs.band_heights(1920) == (576, 1344)

    def test_parse_facecam_box(self):
        assert gs.parse_facecam_box("0,0.377,0.273,0.305", 1920, 1080) == (0, 407, 524, 329)
        assert gs.parse_facecam_box("", 1920, 1080) is None
        assert gs.parse_facecam_box("0,0.5,0.2", 1920, 1080) is None
        assert gs.parse_facecam_box("-0.1,0,0.2,0.2", 1920, 1080) is None
        assert gs.parse_facecam_box("0,0,0,0.2", 1920, 1080) is None

    def test_camera_crop_known_bounds_never_stretches_or_spills(self):
        # 524x329 (aspect 1.59) into 1080x576 (aspect 1.875): stays inside 0..524, 407..736
        # and anchors to the top edge of the box so the top of the head is never cut.
        cx, cy, cw, ch = gs.camera_crop_box((0, 407, 524, 329), 1920, 1080, 1080, 576, known_bounds=True)
        assert cx == 0 and cx + cw <= 524
        assert 407 <= cy <= 408 and cy + ch <= 736
        assert cw % 2 == 0 and ch % 2 == 0 and cx % 2 == 0 and cy % 2 == 0
        assert (cw / ch) == pytest.approx(1080 / 576, rel=0.02)

    def test_camera_crop_detected_box_adds_headroom_without_stretch(self):
        # Detected box 346x195 at y=475 grows upward with CAMERA_HEADROOM (~68px)
        # before matching 1080x576 aspect so hair/head stays inside the camera band.
        cx, cy, cw, ch = gs.camera_crop_box((0, 475, 346, 195), 1920, 1080, 1080, 576, known_bounds=False)
        assert cy < 475
        assert cy <= 408
        assert cy + ch >= 475 + 195 - 4
        assert cw >= 346
        assert (cw / ch) == pytest.approx(1080 / 576, rel=0.02)
        assert 0 <= cx and cx + cw <= 1920
        assert 0 <= cy and cy + ch <= 1080

        # Already pinned to the top edge: headroom clamps cleanly at y=0.
        tcx, tcy, tcw, tch = gs.camera_crop_box((0, 0, 346, 195), 1920, 1080, 1080, 576, known_bounds=False)
        assert tcy == 0
        assert tcx == 0
        assert (tcw / tch) == pytest.approx(1080 / 576, rel=0.02)

    def test_gameplay_crop_centred_when_clear_and_excludes_overlap(self):
        # Non-overlapping left webcam (0..524) vs centred 868px gameplay (526..1394)
        gx, gy, gw, gh = gs.gameplay_crop_box(1920, 1080, 1080, 1344, exclude_box=(0, 407, 524, 329))
        assert (gx, gy, gw, gh) == (526, 0, 868, 1080)

        # Overlapping left webcam (0..640) -> shifts gameplay right past 640
        gx2, gy2, gw2, gh2 = gs.gameplay_crop_box(1920, 1080, 1080, 1344, exclude_box=(0, 400, 640, 340))
        assert gx2 >= 640 and gx2 + gw2 <= 1920
        assert not gs._overlaps((gx2, gy2, gw2, gh2), (0, 400, 640, 340))
        assert (gw2 / gh2) == pytest.approx(1080 / 1344, rel=0.01)

        # Overlapping right webcam (1280..1920) -> shifts gameplay left before 1280
        gx3, gy3, gw3, gh3 = gs.gameplay_crop_box(1920, 1080, 1080, 1344, exclude_box=(1280, 400, 640, 340))
        assert gx3 >= 0 and gx3 + gw3 <= 1280
        assert not gs._overlaps((gx3, gy3, gw3, gh3), (1280, 400, 640, 340))

    def test_compose_stacked_filtergraph_and_gpu_layers(self, monkeypatch):
        monkeypatch.delenv("GAMING_CAMERA_RATIO", raising=False)
        fmt = gs.GamingStackFormat()
        plan = fmt.compose({"box": (0, 407, 524, 329), "known_bounds": True}, 1920, 1080, 1080, 1920, 20.0)
        assert "[0:v]split=2[ca][ga];" in plan.filtergraph
        assert "scale=1080:576[cam];" in plan.filtergraph
        assert "scale=1080:1344[game];" in plan.filtergraph
        assert "[cam][game]vstack=inputs=2,setsar=1[v]" in plan.filtergraph
        assert plan.layout_ranges == [{"start": 0.0, "end": 20.0, "layout": "stack", "seam": 0.3}]

        layers = gpu_render.layers_from_graph(plan.filtergraph, 1920, 1080, 1080, 1920)
        assert len(layers) == 2
        assert layers[0]["dst"] == [0, 0, 1080, 576]
        assert layers[1]["dst"] == [0, 576, 1080, 1344]

    def test_compose_fallback_without_inset_never_uses_track(self):
        fmt = gs.GamingStackFormat()
        plan = fmt.compose({"box": None, "known_bounds": False}, 1920, 1080, 1080, 1920, 15.0)
        assert "vstack" not in plan.filtergraph
        assert "scale=1080:1920,setsar=1[v]" in plan.filtergraph
        assert "crop=w=608:h=1080:x=656:y=0" in plan.filtergraph
        assert plan.layout_ranges == [{"start": 0.0, "end": 15.0, "layout": "general"}]

        layers = gpu_render.layers_from_graph(plan.filtergraph, 1920, 1080, 1080, 1920)
        assert len(layers) == 1
        assert layers[0]["dst"] == [0, 0, 1080, 1920]

    def test_analyze_uses_env_box_and_skips_detection(self, monkeypatch):
        import camera_inset
        monkeypatch.setenv("GAMING_FACECAM_BOX", "0,0.377,0.273,0.305")
        monkeypatch.setattr(camera_inset, "detect",
                            lambda *a, **kw: pytest.fail("detect must be skipped when GAMING_FACECAM_BOX is set"))
        fmt = gs.GamingStackFormat()
        got = fmt.analyze("unused.mp4", 1920, 1080, 1080, 1920, 20.0)
        assert got == {"box": (0, 407, 524, 329), "known_bounds": True, "source": "env"}


class TestDetectedInsetGameplay:
    def test_gameplay_stays_centred_when_only_the_widened_cam_crop_overlaps(self, monkeypatch):
        monkeypatch.delenv("GAMING_CAMERA_RATIO", raising=False)
        fmt = gs.GamingStackFormat()
        box = (1440, 756, 422, 291)  # bottom-right webcam, clear of the centre crop
        plan = fmt.compose({"box": box, "known_bounds": False}, 1920, 1080, 1080, 1920, 10.0)
        gx, gy, gw, gh = plan.meta["game_crop"]
        assert (gw, gh) == (868, 1080)
        assert gx == 526

    def test_gameplay_still_avoids_a_detected_webcam_it_overlaps(self, monkeypatch):
        monkeypatch.delenv("GAMING_CAMERA_RATIO", raising=False)
        fmt = gs.GamingStackFormat()
        box = (800, 700, 320, 240)  # bottom-centre webcam inside the centre crop
        plan = fmt.compose({"box": box, "known_bounds": False}, 1920, 1080, 1080, 1920, 10.0)
        assert not gs._overlaps(plan.meta["game_crop"], box)
