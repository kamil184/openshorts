"""Dead-footage trimming of clip edges (dead_footage.py)."""
import shutil
import subprocess

import numpy as np
import pytest

import dead_footage as d
import presets

H, W = d.PROXY_H, d.PROXY_W
FPS = d.SAMPLE_FPS


def _gameplay(n, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(n, H, W), dtype=np.uint8)


def _loading(n, seed=1):
    """Dark screen with a small spinner moving around and a little noise."""
    rng = np.random.default_rng(seed)
    frames = np.full((n, H, W), 22, dtype=np.int16)
    frames += rng.integers(-1, 2, size=frames.shape).astype(np.int16) * (rng.random(frames.shape) < 0.05)
    for i in range(n):
        x = 70 + (i % 4) * 3
        frames[i, 40:46, x:x + 6] = 200
    return frames.clip(0, 255).astype(np.uint8)


def _frozen(n, seed=2):
    """A bright textured still with codec-like noise."""
    rng = np.random.default_rng(seed)
    base = rng.integers(60, 230, size=(H, W)).astype(np.int16)
    noise = rng.integers(-1, 2, size=(n, H, W)).astype(np.int16) * (rng.random((n, H, W)) < 0.05)
    return (base + noise).clip(0, 255).astype(np.uint8)


def _lobby(n, seed=3):
    """A bright menu where characters idle: a patch keeps changing a little."""
    rng = np.random.default_rng(seed)
    base = rng.integers(80, 240, size=(H, W)).astype(np.int16)
    frames = np.repeat(base[None], n, axis=0)
    frames[:, 30:70, 60:100] += rng.integers(-12, 13, size=(n, 40, 40)).astype(np.int16)
    return frames.clip(0, 255).astype(np.uint8)


def _flags(frames, mask=None):
    return d.dead_flags(d.frame_stats(frames, mask))


def _times(n, t0=0.0):
    return t0 + np.arange(n) / FPS


class TestClassify:
    def test_gameplay_is_alive(self):
        assert not _flags(_gameplay(20)).any()

    def test_loading_screen_is_dead(self):
        assert _flags(_loading(20)).all()

    def test_black_frames_are_dead(self):
        assert _flags(np.zeros((20, H, W), dtype=np.uint8)).all()

    def test_frozen_bright_frame_is_dead(self):
        assert _flags(_frozen(20)).all()

    def test_idle_menu_with_motion_is_alive(self):
        assert not _flags(_lobby(20)).any()

    def test_single_scene_cut_does_not_make_a_still_alive(self):
        frames = np.concatenate([_frozen(10, seed=4), _frozen(10, seed=5)])
        assert _flags(frames).all()

    def test_facecam_motion_is_masked_out(self):
        frames = _loading(20)
        mask = d.facecam_mask("0,0.377,0.273,0.305")
        rng = np.random.default_rng(9)
        frames[:, 34:62, 0:44] = rng.integers(0, 256, size=(20, 28, 44), dtype=np.uint8)
        assert not _flags(frames).all()
        assert _flags(frames, mask).all()


class TestFacecamMask:
    def test_box_is_excluded(self):
        mask = d.facecam_mask("0,0.5,0.5,0.5")
        assert not mask[H - 1, 0] and mask[0, W - 1] and mask[H - 1, W - 1]

    @pytest.mark.parametrize("raw", [None, "", "1,2,3", "a,b,c,d", "0,0,1,1", "0,0,0,0.5"])
    def test_invalid_or_full_box_means_no_mask(self, raw):
        assert d.facecam_mask(raw) is None


class TestEdgeRuns:
    def test_dead_tail_start(self):
        flags = [False] * 20 + [True] * 12
        assert d.dead_tail_start(_times(32, 100), flags) == pytest.approx(105.0)

    def test_dead_head_end(self):
        flags = [True] * 8 + [False] * 20
        assert d.dead_head_end(_times(28, 100), flags) == pytest.approx(102.0)

    def test_short_dead_stretch_is_ignored(self):
        flags = [False] * 20 + [True] * 5  # 1.25 s
        assert d.dead_tail_start(_times(25), flags) is None

    def test_live_island_inside_the_run_is_absorbed(self):
        flags = [False] * 20 + [True] * 10 + [False] * 4 + [True] * 4
        assert d.dead_tail_start(_times(38), flags) == pytest.approx(5.0)

    def test_long_live_stretch_ends_the_run(self):
        flags = [False] * 20 + [True] * 10 + [False] * 8 + [True] * 8
        assert d.dead_tail_start(_times(46), flags) == pytest.approx(38 / FPS)

    def test_live_edge_shorter_than_gap_is_trimmed_with_the_run(self):
        flags = [False] * 20 + [True] * 10 + [False] * 3
        assert d.dead_tail_start(_times(33), flags) == pytest.approx(5.0)

    def test_all_alive(self):
        assert d.dead_tail_start(_times(10), [False] * 10) is None
        assert d.dead_head_end(_times(10), [False] * 10) is None


def _w(*spans):
    return [{"s": s, "e": e} for s, e in spans]


class TestSpeechSafeCuts:
    def test_no_words_cuts_at_the_dead_edge(self):
        assert d.tail_cut(50.0, []) == 50.0
        assert d.head_cut(12.0, []) == 12.0

    def test_dead_start_in_a_pause_cuts_there(self):
        words = _w((40, 41), (41, 43.3), (45.1, 46))
        assert d.tail_cut(44.2, words) == pytest.approx(44.2)

    def test_dead_start_just_after_a_word_keeps_its_decay(self):
        words = _w((40, 43.3), (45.1, 46))
        assert d.tail_cut(43.32, words) == pytest.approx(43.4)

    def test_dead_start_inside_speech_moves_to_the_next_pause(self):
        words = _w((40, 41), (41.1, 44.0), (44.1, 45.0), (46.0, 47.0))
        # 41-41.1 and 44.0-44.1 are not pauses; the first one is 45.0-46.0.
        assert d.tail_cut(42.0, words) == pytest.approx(45.45)

    def test_no_word_is_split_by_any_cut(self):
        words = _w((40, 41), (41.5, 44.0), (44.6, 45.0), (46.0, 47.0))
        for ds in np.arange(39.0, 48.0, 0.05):
            c = d.tail_cut(float(ds), words)
            assert not any(w["s"] < c < w["e"] for w in words)
            h = d.head_cut(float(ds), words)
            assert not any(w["s"] < h < w["e"] for w in words)

    def test_head_dead_end_in_a_pause_cuts_there(self):
        words = _w((10, 11), (13.0, 14))
        assert d.head_cut(12.0, words) == pytest.approx(12.0)

    def test_head_dead_end_inside_speech_moves_back_to_its_start(self):
        words = _w((10, 11), (12.0, 15.0))
        assert d.head_cut(13.0, words) == pytest.approx(11.65)


class TestRefineRange:
    def test_trims_tail(self):
        assert d.refine_range(0, 50, [], 20, None, 42.0) == (0, 42.0)

    def test_trims_head_and_tail(self):
        assert d.refine_range(0, 50, [], 20, 3.0, 45.0) == (3.0, 45.0)

    def test_tiny_trim_is_skipped(self):
        assert d.refine_range(0, 50, [], 20, 0.5, 49.6) == (0, 50)

    def test_trim_beyond_max_is_skipped(self):
        assert d.refine_range(0, 50, [], 20, None, 30.0, max_trim=12) == (0, 50)

    def test_never_below_min_length(self):
        assert d.refine_range(0, 30, [], 25, None, 20.0) == (0, 30)

    def test_larger_single_trim_wins_when_both_do_not_fit(self):
        assert d.refine_range(0, 30, [], 22, 3.0, 24.0) == (0, 24.0)
        assert d.refine_range(0, 30, [], 24, 6.0, 28.0) == (6.0, 30)

    def test_speech_running_through_the_dead_tail_is_kept(self):
        words = _w((40, 50.5))
        assert d.refine_range(0, 50.5, words, 20, None, 44.0) == (0, 50.5)


class TestSwitch:
    def test_default_on(self):
        assert d.enabled(None, {}) is True

    def test_preset_turns_off(self):
        assert d.enabled({"trim_dead_footage": False}, {}) is False

    @pytest.mark.parametrize("raw,want", [("0", False), ("off", False), ("1", True), ("yes", True)])
    def test_env_wins_over_preset(self, raw, want):
        preset = {"trim_dead_footage": not want}
        assert d.enabled(preset, {"CLIP_DEAD_TRIM": raw}) is want

    def test_max_trim_env(self):
        assert d.max_trim_seconds({}) == d.DEFAULT_MAX_TRIM
        assert d.max_trim_seconds({"CLIP_DEAD_TRIM_MAX_SECONDS": "8"}) == 8.0
        assert d.max_trim_seconds({"CLIP_DEAD_TRIM_MAX_SECONDS": "-1"}) == d.DEFAULT_MAX_TRIM
        assert d.max_trim_seconds({"CLIP_DEAD_TRIM_MAX_SECONDS": "x"}) == d.DEFAULT_MAX_TRIM

    def test_preset_field_is_validated(self):
        assert presets.validate_preset({"trim_dead_footage": False}) == {"trim_dead_footage": False}
        with pytest.raises(presets.PresetValidationError):
            presets.validate_preset({"trim_dead_footage": "no"})


def _fake_source(segments):
    """A sample_frames stand-in over a timeline of (seconds, frame maker)."""
    parts, t = [], 0.0
    for seconds, make in segments:
        parts.append((t, t + seconds, make(int(round(seconds * FPS)))))
        t += seconds

    def sample(path, start, end, fps=FPS):
        out = []
        for k in range(int(round((end - start) * fps))):
            tk = start + k / fps
            for a, b, frames in parts:
                if a <= tk < b:
                    out.append(frames[min(int((tk - a) * fps), len(frames) - 1)])
                    break
        return np.array(out, dtype=np.uint8).reshape(-1, H, W)
    return sample


class TestRefineClips:
    def test_loading_screen_tail_is_trimmed_and_recorded(self, monkeypatch):
        monkeypatch.setattr(d, "sample_frames", _fake_source(
            [(42, _gameplay), (10, _loading)]))
        shorts = [{"start": 2.0, "end": 52.0}]
        transcript = {"segments": [{"words": [
            {"word": "go", "start": 30.0, "end": 41.6},
            {"word": "later", "start": 43.0, "end": 51.0}]}]}
        assert d.refine_clips("src.mp4", shorts, transcript, 20, env={}) == 1
        assert shorts[0]["start"] == 2.0
        assert shorts[0]["end"] == pytest.approx(42.0, abs=0.3)
        assert shorts[0]["dead_trim"]["from"] == [2.0, 52.0]

    def test_black_head_is_trimmed(self, monkeypatch):
        monkeypatch.setattr(d, "sample_frames", _fake_source(
            [(4, lambda n: np.zeros((n, H, W), np.uint8)), (60, _gameplay)]))
        shorts = [{"start": 0.0, "end": 40.0}]
        assert d.refine_clips("src.mp4", shorts, None, 20, env={}) == 1
        assert shorts[0]["start"] == pytest.approx(4.0, abs=0.3)
        assert shorts[0]["end"] == 40.0

    def test_switched_off_does_not_sample(self, monkeypatch):
        monkeypatch.setattr(d, "sample_frames", lambda *a, **k: pytest.fail("sampled"))
        shorts = [{"start": 0.0, "end": 40.0}]
        assert d.refine_clips("src.mp4", shorts, None, 20, env={"CLIP_DEAD_TRIM": "0"}) == 0

    def test_measuring_failure_keeps_the_clip(self, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("decode failed")
        monkeypatch.setattr(d, "sample_frames", boom)
        shorts = [{"start": 0.0, "end": 40.0}]
        assert d.refine_clips("src.mp4", shorts, None, 20, env={}) == 0
        assert shorts == [{"start": 0.0, "end": 40.0}]

    def test_clip_at_min_length_is_not_measured(self, monkeypatch):
        monkeypatch.setattr(d, "sample_frames", lambda *a, **k: pytest.fail("sampled"))
        shorts = [{"start": 0.0, "end": 20.5}]
        assert d.refine_clips("src.mp4", shorts, None, 20, env={}) == 0


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_real_encode_with_black_tail(tmp_path):
    src = tmp_path / "src.mp4"
    subprocess.run([
        "ffmpeg", "-v", "error", "-y",
        "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30:duration=26",
        "-f", "lavfi", "-i", "color=c=black:size=320x180:rate=30:duration=6",
        "-filter_complex", "[0][1]concat=n=2:v=1:a=0",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(src),
    ], check=True)
    assert d.refine_clip(str(src), 2.0, 32.0, [], 15.0) == pytest.approx((2.0, 26.0), abs=0.3)
