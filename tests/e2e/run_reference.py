"""End-to-end reference runner and comparator for the real pipeline.

Launches ``main.py`` in child subprocesses (matching ``app.py``) against a
deterministic local HTTP LLM server (``fake_llm.py``) on a fixture video, once
with the default format (TRACK) and once with ``CLIP_FORMAT=gaming_stack``.
Records or verifies clip count/boundaries, ``.layout.json`` sidecars, and
per-clip 1 fps FFmpeg SSIM.
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
for _p in (str(THIS_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fake_llm import DEFAULT_MODEL_ID, FakeLLMServer  # noqa: E402

DEFAULT_BOUNDARY_TOLERANCE = 0.1
DEFAULT_SIDECAR_TOLERANCE = 0.05
DEFAULT_MIN_SSIM = 0.999
DEFAULT_GAMING_FACECAM_BOX = "0,0.377,0.273,0.305"

# Keep child environment clean of server-side credentials, mirroring app.CHILD_ENV_DENYLIST.
_ENV_STRIP = (
    "GEMINI_API_KEY",
    "MANAGED_GEMINI_API_KEY",
    "CLIP_PRESET",
    "CLIP_FORMAT",
    "GAMING_FACECAM_BOX",
    "LLM_EXTRA_BODY",
    "LLM_PROVIDER",
    "WATERMARK",
    "PRO_MODE",
    "AUTO_LAYOUT",
    "MAX_SOURCE_MINUTES",
    "SOURCE_CAP_MINUTES",
    "SOURCE_START_SECONDS",
)

_SSIM_SUMMARY_RE = re.compile(r"SSIM\s+Y:[^\n]*\bAll:([0-9]+(?:\.[0-9]+)?)")
_SSIM_FRAME_RE = re.compile(r"\bAll:([0-9]+(?:\.[0-9]+)?)")


@dataclass(frozen=True)
class JobSpec:
    name: str
    env: Dict[str, str] = field(default_factory=dict)


DEFAULT_JOBS: Dict[str, JobSpec] = {
    "track": JobSpec(name="track", env={}),
    "gaming_stack": JobSpec(
        name="gaming_stack",
        env={
            "CLIP_FORMAT": "gaming_stack",
            "GAMING_FACECAM_BOX": DEFAULT_GAMING_FACECAM_BOX,
        },
    ),
}


def build_child_env(
    *,
    llm_base_url: str,
    llm_model: str,
    asr_lock_dir: str,
    extra_env: Optional[Dict[str, str]] = None,
    base_env: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Construct a child env matching ``app.py`` launching ``main.py``."""
    env = dict(os.environ if base_env is None else base_env)
    for key in _ENV_STRIP:
        env.pop(key, None)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env["LLM_BASE_URL"] = llm_base_url
    env["LLM_MODEL"] = llm_model
    env["LLM_API_KEY"] = "e2e"
    env["AUTO_HOOK"] = "1"
    env["AUTO_CAPTIONS"] = "1"
    env["ASR_LOCK_DIR"] = asr_lock_dir
    if extra_env:
        env.update(extra_env)
    return env


def parse_clip_ready_lines(stdout_text: str) -> Dict[int, str]:
    """Parse ``CLIP_READY <idx> <filename>`` markers emitted by ``main.py``."""
    announced: Dict[int, str] = {}
    for line in stdout_text.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) == 3 and parts[0] == "CLIP_READY":
            try:
                announced[int(parts[1])] = parts[2]
            except ValueError:
                continue
    return announced


def collect_job_outputs(
    job_dir: Path,
    fixture_stem: str,
    stdout_text: str,
) -> Dict[str, Any]:
    """Read metadata, final rendered clip files, and ``.layout.json`` sidecars."""
    meta_path = job_dir / f"{fixture_stem}_metadata.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing job metadata JSON: {meta_path}")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    shorts = metadata.get("shorts") or []
    if not shorts:
        raise RuntimeError(f"No shorts produced in {meta_path}")

    announced = parse_clip_ready_lines(stdout_text)
    clips: List[Dict[str, Any]] = []
    for idx, short in enumerate(shorts):
        clip_num = idx + 1
        canonical = job_dir / f"{fixture_stem}_clip_{clip_num}.mp4"
        sidecar_path = Path(f"{canonical}.layout.json")
        if not sidecar_path.is_file():
            raise FileNotFoundError(f"Missing layout sidecar: {sidecar_path}")
        delivered_name = announced.get(idx)
        delivered_path = (job_dir / delivered_name) if delivered_name else canonical
        if not delivered_path.is_file():
            raise FileNotFoundError(f"Missing rendered clip video: {delivered_path}")
        clips.append({
            "index": clip_num,
            "start": float(short["start"]),
            "end": float(short["end"]),
            "sidecar_path": sidecar_path,
            "sidecar": json.loads(sidecar_path.read_text(encoding="utf-8")),
            "video_path": delivered_path,
        })
    return {"clips": clips, "metadata": metadata}


def compute_1fps_ssim(
    ref_video: Path,
    actual_video: Path,
    *,
    ffmpeg_bin: str = "ffmpeg",
) -> tuple[float, float, int]:
    """Compare 1 fps frames of two videos via FFmpeg's ``ssim`` filter.

    Returns ``(mean_all_ssim, min_frame_ssim, frame_count)``.
    """
    with tempfile.NamedTemporaryFile(prefix="e2e-ssim-", suffix=".log") as stats_file:
        stats_path = stats_file.name
        filter_graph = (
            "[0:v]fps=1,setpts=PTS-STARTPTS[a];"
            "[1:v]fps=1,setpts=PTS-STARTPTS[b];"
            f"[a][b]ssim=shortest=1:stats_file={stats_path}"
        )
        cmd = [
            ffmpeg_bin,
            "-nostdin",
            "-v",
            "info",
            "-i",
            str(ref_video),
            "-i",
            str(actual_video),
            "-filter_complex",
            filter_graph,
            "-f",
            "null",
            "-",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if proc.returncode != 0:
            raise RuntimeError(
                f"ffmpeg ssim failed (exit {proc.returncode}): {proc.stderr[-500:]}"
            )
        stats_text = Path(stats_path).read_text(encoding="utf-8", errors="replace")

    frame_scores = [float(m.group(1)) for m in _SSIM_FRAME_RE.finditer(stats_text)]
    match = _SSIM_SUMMARY_RE.search(proc.stderr)
    if not frame_scores and not match:
        raise RuntimeError(f"Could not parse SSIM output:\n{proc.stderr[-500:]}")
    mean_ssim = float(match.group(1)) if match else (sum(frame_scores) / len(frame_scores))
    min_ssim = min(frame_scores) if frame_scores else mean_ssim
    return mean_ssim, min_ssim, len(frame_scores)


def compare_sidecars(
    ref_sidecar: Dict[str, Any],
    actual_sidecar: Dict[str, Any],
    *,
    tolerance: float = DEFAULT_SIDECAR_TOLERANCE,
) -> List[str]:
    """Return human-readable diff lines if two ``.layout.json`` sidecars diverge."""
    ref_ranges = ref_sidecar.get("ranges") or []
    act_ranges = actual_sidecar.get("ranges") or []
    errors: List[str] = []

    if len(ref_ranges) != len(act_ranges):
        errors.append(
            f"range count differs: expected {len(ref_ranges)}, got {len(act_ranges)}"
        )
    else:
        for i, (r_ref, r_act) in enumerate(zip(ref_ranges, act_ranges)):
            if set(r_ref.keys()) != set(r_act.keys()):
                errors.append(
                    f"range[{i}] keys differ: expected {sorted(r_ref.keys())}, "
                    f"got {sorted(r_act.keys())}"
                )
                continue
            for key in sorted(r_ref.keys()):
                v_ref, v_act = r_ref[key], r_act[key]
                if isinstance(v_ref, (int, float)) and isinstance(v_act, (int, float)):
                    if abs(float(v_ref) - float(v_act)) > tolerance:
                        errors.append(
                            f"range[{i}].{key} differs: expected {v_ref}, got {v_act}"
                        )
                elif v_ref != v_act:
                    errors.append(
                        f"range[{i}].{key} differs: expected {v_ref!r}, got {v_act!r}"
                    )

    extra_ref = {k: v for k, v in ref_sidecar.items() if k != "ranges"}
    extra_act = {k: v for k, v in actual_sidecar.items() if k != "ranges"}
    if extra_ref != extra_act:
        errors.append(f"top-level sidecar metadata differs: expected {extra_ref}, got {extra_act}")

    if errors:
        ref_txt = json.dumps(ref_sidecar, indent=2, sort_keys=True).splitlines()
        act_txt = json.dumps(actual_sidecar, indent=2, sort_keys=True).splitlines()
        diff = "\n".join(difflib.unified_diff(ref_txt, act_txt, fromfile="ref", tofile="actual", lineterm=""))
        if diff:
            errors.append(diff)
    return errors


def compare_job(
    job_name: str,
    actual: Dict[str, Any],
    ref_job_dir: Path,
    *,
    boundary_tolerance: float = DEFAULT_BOUNDARY_TOLERANCE,
    sidecar_tolerance: float = DEFAULT_SIDECAR_TOLERANCE,
    min_ssim: float = DEFAULT_MIN_SSIM,
    ssim_fn=compute_1fps_ssim,
) -> tuple[bool, List[str]]:
    """Compare one job's outputs against its recorded reference."""
    manifest_path = ref_job_dir / "manifest.json"
    if not manifest_path.is_file():
        return False, [f"[{job_name}] FAIL: missing reference manifest at {manifest_path}"]

    ref_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ref_clips = ref_manifest.get("clips") or []
    act_clips = actual.get("clips") or []
    lines: List[str] = []
    ok = True

    if len(act_clips) != len(ref_clips):
        lines.append(
            f"[{job_name}] FAIL clip count: expected {len(ref_clips)}, got {len(act_clips)}"
        )
        return False, lines

    for ref_c, act_c in zip(ref_clips, act_clips):
        idx = int(ref_c["index"])
        clip_ok = True
        clip_notes: List[str] = []

        d_start = abs(float(act_c["start"]) - float(ref_c["start"]))
        d_end = abs(float(act_c["end"]) - float(ref_c["end"]))
        if d_start > boundary_tolerance or d_end > boundary_tolerance:
            clip_ok = False
            clip_notes.append(
                f"boundaries mismatch: expected [{ref_c['start']:.3f}, {ref_c['end']:.3f}], "
                f"got [{act_c['start']:.3f}, {act_c['end']:.3f}] "
                f"(Δstart={d_start:.3f}s, Δend={d_end:.3f}s > {boundary_tolerance:.3f}s)"
            )
        else:
            clip_notes.append(f"bounds ok (Δstart={d_start:.3f}s, Δend={d_end:.3f}s)")

        ref_sidecar_path = ref_job_dir / ref_c["sidecar"]
        ref_sidecar = json.loads(ref_sidecar_path.read_text(encoding="utf-8"))
        sidecar_diffs = compare_sidecars(
            ref_sidecar, act_c["sidecar"], tolerance=sidecar_tolerance
        )
        if sidecar_diffs:
            clip_ok = False
            clip_notes.append("sidecar mismatch:\n  " + "\n  ".join(sidecar_diffs))
        else:
            clip_notes.append(f"sidecar ok ({len(ref_sidecar.get('ranges') or [])} ranges)")

        ref_video_path = ref_job_dir / ref_c["video"]
        mean_ssim, min_frame_ssim, n_frames = ssim_fn(ref_video_path, act_c["video_path"])
        if n_frames <= 0:
            clip_ok = False
            clip_notes.append("SSIM error: 0 frames compared")
        elif mean_ssim < min_ssim:
            clip_ok = False
            clip_notes.append(
                f"SSIM too low: mean={mean_ssim:.4f} (min_frame={min_frame_ssim:.4f}, "
                f"frames={n_frames}) < {min_ssim:.4f}"
            )
        else:
            clip_notes.append(
                f"ssim_1fps={mean_ssim:.4f} (min={min_frame_ssim:.4f}, frames={n_frames})"
            )

        status = "PASS" if clip_ok else "FAIL"
        if not clip_ok:
            ok = False
        lines.append(f"[{job_name} clip {idx}] {status}: " + "; ".join(clip_notes))

    return ok, lines


def write_reference(job_name: str, actual: Dict[str, Any], ref_job_dir: Path) -> None:
    """Persist one job's outputs atomically as the new reference."""
    stage_dir = ref_job_dir.parent / f".staging-{job_name}-{os.getpid()}"
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    clips_dir = stage_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    manifest_clips: List[Dict[str, Any]] = []
    for clip in actual["clips"]:
        idx = int(clip["index"])
        sidecar_rel = f"clips/clip_{idx}.layout.json"
        video_rel = f"clips/clip_{idx}.mp4"
        shutil.copy2(clip["sidecar_path"], stage_dir / sidecar_rel)
        shutil.copy2(clip["video_path"], stage_dir / video_rel)
        manifest_clips.append({
            "index": idx,
            "start": float(clip["start"]),
            "end": float(clip["end"]),
            "sidecar": sidecar_rel,
            "video": video_rel,
        })

    manifest = {
        "job": job_name,
        "clips": manifest_clips,
    }
    (stage_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    if ref_job_dir.exists():
        shutil.rmtree(ref_job_dir)
    ref_job_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(stage_dir), str(ref_job_dir))


def run_single_job(
    spec: JobSpec,
    fixture_path: Path,
    work_dir: Path,
    *,
    llm_base_url: str,
    llm_model: str,
    asr_lock_dir: Path,
) -> Dict[str, Any]:
    """Run ``main.py`` once cleanly and return collected output artifacts."""
    job_out = work_dir / f"job_{spec.name}"
    if job_out.exists():
        shutil.rmtree(job_out)
    job_out.mkdir(parents=True, exist_ok=True)

    env = build_child_env(
        llm_base_url=llm_base_url,
        llm_model=llm_model,
        asr_lock_dir=str(asr_lock_dir),
        extra_env=spec.env,
    )
    cmd = [
        sys.executable,
        "-u",
        str(REPO_ROOT / "main.py"),
        "-i",
        str(fixture_path),
        "-o",
        str(job_out),
    ]
    print(f"\n=== Running E2E job: {spec.name} ===", flush=True)
    t0 = time.monotonic()
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    elapsed = time.monotonic() - t0
    if proc.returncode != 0:
        tail = "\n".join(proc.stdout.splitlines()[-60:])
        raise RuntimeError(
            f"Job {spec.name!r} failed with exit code {proc.returncode} "
            f"after {elapsed:.1f}s:\n{tail}"
        )
    print(f"=== Finished E2E job: {spec.name} in {elapsed:.1f}s ===", flush=True)
    return {
        **collect_job_outputs(job_out, fixture_path.stem, proc.stdout),
        "elapsed_s": elapsed,
    }


def run_suite(
    *,
    fixture_path: Path,
    reference_dir: Path,
    record: bool = False,
    jobs: Optional[Sequence[str]] = None,
    min_ssim: float = DEFAULT_MIN_SSIM,
    boundary_tolerance: float = DEFAULT_BOUNDARY_TOLERANCE,
    sidecar_tolerance: float = DEFAULT_SIDECAR_TOLERANCE,
    work_dir: Optional[Path] = None,
) -> int:
    """Execute the configured E2E jobs sequentially and record or verify results."""
    if not fixture_path.is_file():
        print(f"FAIL: fixture video not found: {fixture_path}", flush=True)
        return 1

    selected = list(jobs or ("track", "gaming_stack"))
    for name in selected:
        if name not in DEFAULT_JOBS:
            print(f"FAIL: unknown job {name!r} (valid: {sorted(DEFAULT_JOBS)})", flush=True)
            return 1

    tmp_ctx = tempfile.TemporaryDirectory(prefix="openshorts-e2e-") if work_dir is None else None
    run_root = Path(tmp_ctx.name) if tmp_ctx is not None else work_dir
    assert run_root is not None
    asr_locks = run_root / "asr_locks"
    asr_locks.mkdir(parents=True, exist_ok=True)

    suite_t0 = time.monotonic()
    all_ok = True
    report_lines: List[str] = []

    try:
        with FakeLLMServer(model_id=DEFAULT_MODEL_ID) as llm:
            for name in selected:
                spec = DEFAULT_JOBS[name]
                actual = run_single_job(
                    spec,
                    fixture_path,
                    run_root,
                    llm_base_url=llm.base_url,
                    llm_model=llm.model_id,
                    asr_lock_dir=asr_locks,
                )
                ref_job_dir = reference_dir / name
                if record:
                    write_reference(name, actual, ref_job_dir)
                    report_lines.append(
                        f"[{name}] RECORDED: {len(actual['clips'])} clip(s) "
                        f"in {actual['elapsed_s']:.1f}s -> {ref_job_dir}"
                    )
                else:
                    ok, lines = compare_job(
                        name,
                        actual,
                        ref_job_dir,
                        boundary_tolerance=boundary_tolerance,
                        sidecar_tolerance=sidecar_tolerance,
                        min_ssim=min_ssim,
                    )
                    report_lines.extend(lines)
                    report_lines.append(
                        f"[{name}] wall_s={actual['elapsed_s']:.1f}s"
                    )
                    if not ok:
                        all_ok = False
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()

    total_s = time.monotonic() - suite_t0
    print("\n=== E2E Summary ===", flush=True)
    for line in report_lines:
        print(line, flush=True)
    status = "RECORDED" if record else ("PASS" if all_ok else "FAIL")
    print(f"RESULT: {status} (total_wall_s={total_s:.1f}s)", flush=True)
    return 0 if all_ok else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run and verify E2E pipeline reference.")
    parser.add_argument(
        "--fixture",
        default=os.environ.get("E2E_FIXTURE", ""),
        help="Path to input fixture video.",
    )
    parser.add_argument(
        "--reference",
        default=os.environ.get("E2E_REFERENCE_DIR", ""),
        help="Directory where reference outputs are stored/read.",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        default=os.environ.get("E2E_RECORD", "").lower() in ("1", "true", "yes"),
        help="Record a new reference instead of comparing.",
    )
    parser.add_argument(
        "--jobs",
        default="track,gaming_stack",
        help="Comma-separated jobs to run (default: track,gaming_stack).",
    )
    parser.add_argument(
        "--min-ssim",
        type=float,
        default=float(os.environ.get("E2E_MIN_SSIM", str(DEFAULT_MIN_SSIM))),
        help=f"Minimum acceptable 1 fps SSIM (default: {DEFAULT_MIN_SSIM}).",
    )
    parser.add_argument(
        "--work-dir",
        default="",
        help="Optional working directory (default: temporary directory).",
    )
    args = parser.parse_args(argv)

    if not args.fixture or not args.reference:
        parser.error("Both --fixture (or E2E_FIXTURE) and --reference (or E2E_REFERENCE_DIR) are required.")

    jobs = [j.strip() for j in args.jobs.split(",") if j.strip()]
    return run_suite(
        fixture_path=Path(args.fixture),
        reference_dir=Path(args.reference),
        record=args.record,
        jobs=jobs,
        min_ssim=args.min_ssim,
        work_dir=Path(args.work_dir) if args.work_dir else None,
    )


if __name__ == "__main__":
    raise SystemExit(main())
