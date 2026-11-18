"""The manifest scans must not block the event loop (volume: hundreds of jobs).

The API, its health check and the job queue share one event loop. With the
queue backed up there are thousands of job dirs, and reading every manifest
(open + json.load) inside the loop stalled it for as long as that took on
every Stripe upsert, every resume scan and every metering sweep. The scans
now read and rewrite on a worker thread; only the heap is touched on the loop.

The first read waits for the event loop to release it, proving that the loop
can progress during a scan without depending on machine scheduling speed.
"""
import asyncio
import itertools
import json
import threading

import pytest

app_module = pytest.importorskip("app")

N = 800                  # manifests


@pytest.fixture
def busy_disk(tmp_path, monkeypatch):
    root = tmp_path / "output"
    root.mkdir()
    for name in (".instance", ".queue_watch.json", ".queue_watch.lock"):
        (root / name).write_text("x")
    monkeypatch.setattr(app_module, "OUTPUT_DIR", str(root))
    monkeypatch.setattr(app_module, "INSTANCE_ID", "me")
    monkeypatch.setattr(app_module, "_running_jobs", set())
    monkeypatch.setattr(app_module, "_admitting", set())
    monkeypatch.setattr(app_module, "jobs", {})
    monkeypatch.setattr(app_module, "job_queue", asyncio.PriorityQueue())
    monkeypatch.setattr(app_module, "_job_seq", itertools.count())
    monkeypatch.setattr(app_module, "_seen_priority_gen", app_module._UNSEEN)
    for i in range(N):
        d = root / f"j{i}"
        d.mkdir()
        uid = f"u{i % 30}"
        (d / app_module._RESUME_FILE).write_text(json.dumps(
            {"priority": 2, "user_id": uid, "reservation_id": f"r{i}"}))
        app_module.jobs[f"j{i}"] = {"status": "queued", "user_id": uid,
                                    "reservation_id": f"r{i}", "logs": []}
        app_module._enqueue_job(f"j{i}", 2)
    real = app_module._read_manifest

    state = {"started": threading.Event(), "release": threading.Event(), "threads": []}

    def gated_read(job_id):
        state["threads"].append(threading.get_ident())
        state["started"].set()
        if not state["release"].wait(timeout=5):
            raise AssertionError("event loop did not release the manifest read")
        return real(job_id)
    monkeypatch.setattr(app_module, "_read_manifest", gated_read)
    return state


def _scan_with_live_loop(coro_fn, state):
    async def run():
        loop_thread = threading.get_ident()
        scan = asyncio.create_task(coro_fn())
        try:
            started = await asyncio.to_thread(state["started"].wait, 5)
            progressed = started and not scan.done()
        finally:
            state["release"].set()
        result = await scan
        assert progressed, "event loop could not progress during a manifest scan"
        assert len(state["threads"]) == N, "every manifest must be read"
        assert all(t != loop_thread for t in state["threads"]), "manifest read ran on the event loop"
        return result
    return asyncio.run(run())


def test_reprioritize_keeps_the_loop_responsive(busy_disk):
    moved = _scan_with_live_loop(lambda: app_module.reprioritize_user_jobs("u7", 1), busy_disk)
    assert moved == sum(1 for i in range(N) if i % 30 == 7)


def test_follow_keeps_the_loop_responsive(busy_disk):
    _scan_with_live_loop(app_module._follow_manifest_priorities, busy_disk)


def test_live_reservations_keep_the_loop_responsive(busy_disk):
    live = _scan_with_live_loop(app_module._live_reservation_ids, busy_disk)
    assert len(live) == N
