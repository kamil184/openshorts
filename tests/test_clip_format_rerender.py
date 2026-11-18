"""A clip re-rendered in the editor keeps the job's clip format.

The re-render runs in the API process, which does not have the job's
CLIP_FORMAT / GAMING_FACECAM_BOX: the job's metadata carries them instead.
"""
import asyncio
import json

import httpx
import pytest

import job_harness as jh

app_module = pytest.importorskip("app")
pytest.importorskip("main")
pytestmark = pytest.mark.skipif(not jh.have_tools(), reason="needs ffmpeg")

JOB_ID = "clip-format-rerender-job"


@pytest.fixture()
def stack_job(tmp_path, monkeypatch):
    for name in ("CLIP_FORMAT", "GAMING_FACECAM_BOX", "WATERMARK"):
        monkeypatch.delenv(name, raising=False)
    out_root, up_root = tmp_path / "output", tmp_path / "uploads"
    job_dir = out_root / JOB_ID
    job_dir.mkdir(parents=True)
    up_root.mkdir()
    monkeypatch.setattr(app_module, "OUTPUT_DIR", str(out_root))
    monkeypatch.setattr(app_module, "UPLOAD_DIR", str(up_root))
    jh.make_source(job_dir / "src.mp4")
    clip = {"start": 10.0, "end": 30.0, "video_title_for_youtube_short": "t",
            "video_url": f"/videos/{JOB_ID}/mytitle_clip_1.mp4"}
    meta = {"shorts": [clip], "transcript": {"language": "en", "segments": []},
            "source_video": "src.mp4", "output_format": "auto", "cost_analysis": {},
            "clip_format": "gaming_stack", "facecam_box": [0.0, 0.5, 0.3, 0.4]}
    (job_dir / "mytitle_metadata.json").write_text(json.dumps(meta))
    (job_dir / "mytitle_clip_1.mp4").write_bytes(b"canonical")
    app_module.jobs[JOB_ID] = {"status": "completed", "logs": [],
                               "result": {"clips": [dict(clip)], "cost_analysis": {}},
                               "user_id": None, "watermark": False}
    yield job_dir
    app_module.jobs.pop(JOB_ID, None)


def _post(path, body):
    async def _do():
        transport = httpx.ASGITransport(app=app_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
            return await c.post(path, json=body)
    return asyncio.run(_do())


def test_a_source_rerender_keeps_the_gaming_stack(stack_job):
    import layout_ranges
    resp = _post("/api/clip/rerender", {
        "job_id": JOB_ID, "clip_index": 0, "reapply_captions": False,
        "segments": [{"start": 2, "end": 8}]})
    assert resp.status_code == 200, resp.text
    assert resp.json()["render_path"] == "source"
    served = resp.json()["new_video_url"].rsplit("/", 1)[1]
    ranges = layout_ranges.read(str(stack_job / served))
    assert [r["layout"] for r in ranges] == ["stack"]
    assert ranges[0]["seam"] == pytest.approx(0.3, abs=0.01)
