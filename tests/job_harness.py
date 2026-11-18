"""Run the real ``main.py`` job on a tiny generated video.

The model is an in-process OpenAI-compatible server that answers the scoring
and detail passes with canned clips, and the transcript is precomputed, so a
job runs end to end in seconds without a GPU, an ASR model or a network.
"""
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DURATION = 40


def have_tools():
    return bool(shutil.which("ffmpeg"))


def make_source(path, seconds=DURATION):
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y",
         "-f", "lavfi", "-i", f"testsrc=size=640x360:rate=25:duration={seconds}",
         "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
         "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest", str(path)],
        check=True, timeout=120)
    return path


def make_transcript(path, seconds=DURATION):
    segments = []
    for i in range(0, seconds, 4):
        words = [{"word": f"word{i}_{k}", "start": i + k * 0.5, "end": i + k * 0.5 + 0.4}
                 for k in range(6)]
        segments.append({"start": float(i), "end": float(i + 4),
                         "text": " ".join(w["word"] for w in words), "words": words})
    with open(path, "w") as f:
        json.dump({"language": "en", "segments": segments}, f)
    return path


def detail_clip(**fields):
    clip = {
        "start": 5.0, "end": 25.0, "source_window_id": "window_001",
        "predicted_score": 90, "viral_hook_text": "watch this",
        "video_description_for_tiktok": "what a round",
        "video_description_for_instagram": "what a round",
        "video_title_for_youtube_short": "Big round",
        "why": "payoff",
    }
    clip.update(fields)
    return clip


class FakeModel:
    """OpenAI-compatible endpoint: 404 for the model list, canned JSON for
    the scoring and detail passes."""

    def __init__(self, shorts):
        self.shorts = shorts
        self.prompts = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, payload):
                body = json.dumps(payload).encode()
                self.send_response(code)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                self._send(404, {"error": {"message": "no model list"}})

            def do_POST(self):
                raw = self.rfile.read(int(self.headers.get("content-length") or 0))
                req = json.loads(raw or b"{}")
                text = ""
                for m in req.get("messages") or []:
                    c = m.get("content")
                    if isinstance(c, str):
                        text += c
                    else:
                        text += "".join(p.get("text", "") for p in c or [] if isinstance(p, dict))
                outer.prompts.append(text)
                if "CANDIDATE_WINDOWS_JSON:" in text:
                    content = {"shorts": outer.shorts}
                else:
                    content = {"windows": [{"id": "window_001", "start": 0, "end": DURATION,
                                            "score": 90, "reason": "r"}]}
                self._send(200, {
                    "choices": [{"message": {"role": "assistant",
                                             "content": json.dumps(content)}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                })

        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        self.server = ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.port}/v1"

    def close(self):
        self.server.shutdown()


def job_env(model, **extra):
    env = dict(os.environ)
    for name in ("GEMINI_API_KEY", "LLM_PROVIDER", "LLM_VISION_MODEL", "LLM_VIDEO_MODEL",
                 "CLIP_PRESET", "CLIP_PRESET_JSON", "CLIP_FORMAT", "GAMING_FACECAM_BOX",
                 "AUTO_HOOK", "WATERMARK", "PRESETS_DIR"):
        env.pop(name, None)
    env.update(LLM_BASE_URL=model.base_url, LLM_MODEL="fake-model", AUTO_CAPTIONS="0",
               PYTHONIOENCODING="utf-8", CLIP_WORKERS="1")
    env.update({k: str(v) for k, v in extra.items()})
    return env


def start_job(tmp_path, env, output_format="horizontal"):
    """Popen of ``main.py`` on a fresh source; returns (proc, output_dir)."""
    source = make_source(tmp_path / "Source.mp4")
    transcript = make_transcript(tmp_path / "transcript.json")
    out = tmp_path / "out"
    out.mkdir()
    proc = subprocess.Popen(
        [sys.executable, "-u", os.path.join(REPO, "main.py"), "-i", str(source),
         "-o", str(out), "--format", output_format, "--transcript", str(transcript)],
        cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return proc, out


def metadata(out):
    path = out / "Source_metadata.json"
    with open(path) as f:
        return json.load(f)
