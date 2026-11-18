"""Deterministic OpenAI-compatible HTTP server for end-to-end pipeline runs.

Advertises a single model supporting text, image and video inputs at
``GET /models`` and answers ``POST /chat/completions`` with canned structured JSON
selected by ``response_format.json_schema.name``.
"""
from __future__ import annotations

import argparse
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Sequence

DEFAULT_MODEL_ID = "e2e-reference-model"

# Two fixed clips inside a ~3 minute source fixture: deterministic timestamps
# aligned cleanly with conversational stretches so word-snapping, captions,
# hooks and layout stages all exercise real media.
DEFAULT_CLIPS: List[Dict[str, Any]] = [
    {
        "start": 20.0,
        "end": 45.0,
        "source_window_id": "window_001",
        "predicted_score": 94,
        "video_description_for_tiktok": "Quick turnaround right from the opening seconds. #gaming #shorts",
        "video_description_for_instagram": "Quick turnaround right from the opening seconds. #gaming #reels",
        "video_title_for_youtube_short": "Opening Turnaround Moment",
        "viral_hook_text": "Wait until the five thousand drop",
        "why": "Clear speech hook followed by immediate payoff.",
    },
    {
        "start": 52.0,
        "end": 78.0,
        "source_window_id": "window_001",
        "predicted_score": 89,
        "video_description_for_tiktok": "Mistakes compound fast when moving at full speed. #gaming #shorts",
        "video_description_for_instagram": "Mistakes compound fast when moving at full speed. #gaming #reels",
        "video_title_for_youtube_short": "Buying Four By Mistake",
        "viral_hook_text": "Bought four on accident right here",
        "why": "Fast escalation with reaction and clean comedic payoff.",
    },
]


def _extract_user_text_and_images(messages: Sequence[Dict[str, Any]]) -> tuple[str, int]:
    """Return concatenated text content and image-part count from user messages."""
    texts: List[str] = []
    image_count = 0
    for msg in messages or []:
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                ptype = part.get("type")
                if ptype == "text":
                    texts.append(str(part.get("text") or ""))
                elif ptype == "image_url":
                    image_count += 1
    return "\n".join(texts), image_count


def _extract_windows_json(prompt_text: str, marker: str) -> List[Dict[str, Any]]:
    """Parse the JSON window list emitted after ``marker`` in scoring/detail prompts."""
    idx = prompt_text.find(marker)
    if idx == -1:
        return []
    tail = prompt_text[idx + len(marker):].lstrip()
    decoder = json.JSONDecoder()
    try:
        parsed, _ = decoder.raw_decode(tail)
    except json.JSONDecodeError:
        match = re.search(r"\[[\s\S]*\]", tail)
        if not match:
            return []
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []
    return [w for w in parsed if isinstance(w, dict)] if isinstance(parsed, list) else []


def canned_response_for_request(
    body: Dict[str, Any],
    *,
    clips: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build a deterministic structured-output payload for one chat completion call."""
    rf = body.get("response_format") or {}
    schema_name = str(((rf.get("json_schema") or {}).get("name") or "")).strip().lower()
    prompt_text, n_images = _extract_user_text_and_images(body.get("messages") or [])
    chosen_clips = [dict(c) for c in (clips if clips is not None else DEFAULT_CLIPS)]

    if schema_name == "scoreresponse":
        windows = _extract_windows_json(prompt_text, "WINDOWS_JSON:")
        if not windows:
            windows = [{"id": "window_001", "start": 0.0, "end": 90.0}]
        scored = []
        for idx, win in enumerate(windows):
            scored.append({
                "id": str(win.get("id") or f"window_{idx + 1:03d}"),
                "start": float(win.get("start", 0.0)),
                "end": float(win.get("end", 90.0)),
                "score": max(10, 95 - idx * 5),
                "reason": "Consistent pacing and clear spoken hook.",
            })
        return {"windows": scored}

    if schema_name == "detailresponse":
        candidates = _extract_windows_json(prompt_text, "CANDIDATE_WINDOWS_JSON:")
        candidate_ids = [str(w.get("id")) for w in candidates if w.get("id")]
        # Only emit the canned picks on the primary shortlist call (which includes
        # the top-ranked window). A subsequent detail-floor pass over unused spare
        # windows gets an empty list so clips are never duplicated.
        if candidates and candidate_ids and candidate_ids[0] != "window_001":
            return {"shorts": []}
        default_win = candidate_ids[0] if candidate_ids else "window_001"
        out = []
        for clip in chosen_clips:
            item = dict(clip)
            if item.get("source_window_id") not in candidate_ids and candidate_ids:
                item["source_window_id"] = default_win
            out.append(item)
        return {"shorts": out}

    if schema_name == "visualresponse":
        visual_clips = []
        for clip in chosen_clips:
            visual_clips.append({
                "start": clip["start"],
                "end": clip["end"],
                "predicted_score": clip["predicted_score"],
                "video_description_for_tiktok": clip["video_description_for_tiktok"],
                "video_description_for_instagram": clip["video_description_for_instagram"],
                "video_title_for_youtube_short": clip["video_title_for_youtube_short"],
                "viral_hook_text": clip["viral_hook_text"],
            })
        return {"shorts": visual_clips}

    if schema_name == "layoutchoice":
        return {
            "layout": "none",
            "confidence": 0.99,
            "why": "Standard camera and action framing.",
        }

    if schema_name == "subtitleanswer":
        count = max(1, n_images)
        return {"frames": [{"frame": i, "subtitle_top": None} for i in range(count)]}

    if schema_name == "groundedhook":
        return {
            "on_screen": "Live gameplay action with camera overlay",
            "viral_hook_text": "Wait until the five thousand drop",
            "video_title_for_youtube_short": "Opening Turnaround Moment",
        }

    if schema_name == "shotcontentresponse":
        return {"shots": []}

    raise ValueError(f"Unsupported schema name requested from fake LLM: {schema_name!r}")


class FakeLLMServer:
    """In-process deterministic OpenAI-compatible HTTP server."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        model_id: str = DEFAULT_MODEL_ID,
        clips: Optional[List[Dict[str, Any]]] = None,
    ):
        self.host = host
        self.port = port
        self.model_id = model_id
        self.clips = [dict(c) for c in clips] if clips is not None else None
        self.requests: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    @property
    def base_url(self) -> str:
        if self._httpd is None:
            raise RuntimeError("FakeLLMServer is not started")
        host, port = self._httpd.server_address[:2]
        return f"http://{host}:{port}"

    def start(self) -> "FakeLLMServer":
        parent = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:
                return

            def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
                raw = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self) -> None:
                path = self.path.split("?", 1)[0].rstrip("/")
                if path in ("/models", "/v1/models"):
                    self._send_json(200, {
                        "object": "list",
                        "data": [
                            {
                                "id": parent.model_id,
                                "object": "model",
                                "architecture": {
                                    "input_modalities": ["text", "image", "video"],
                                },
                            }
                        ],
                    })
                    return
                self._send_json(404, {"error": f"Unknown path {self.path}"})

            def do_POST(self) -> None:
                path = self.path.split("?", 1)[0].rstrip("/")
                if path not in ("/chat/completions", "/v1/chat/completions"):
                    self._send_json(404, {"error": f"Unknown path {self.path}"})
                    return
                length = int(self.headers.get("Content-Length") or "0")
                raw = self.rfile.read(length) if length > 0 else b"{}"
                try:
                    body = json.loads(raw.decode("utf-8"))
                except json.JSONDecodeError as exc:
                    self._send_json(400, {"error": f"Invalid JSON: {exc}"})
                    return

                rf = body.get("response_format") or {}
                schema_name = ((rf.get("json_schema") or {}).get("name") or "").lower()
                with parent._lock:
                    parent.requests.append({
                        "schema": schema_name,
                        "model": body.get("model"),
                    })

                try:
                    answer = canned_response_for_request(body, clips=parent.clips)
                except ValueError as exc:
                    self._send_json(400, {"error": str(exc)})
                    return

                self._send_json(200, {
                    "id": "e2e-completion",
                    "object": "chat.completion",
                    "model": body.get("model") or parent.model_id,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": json.dumps(answer),
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 50,
                        "total_tokens": 150,
                    },
                })

        self._httpd = ThreadingHTTPServer((self.host, self.port), Handler)
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, daemon=True, name="fake-llm-server"
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def __enter__(self) -> "FakeLLMServer":
        return self.start()

    def __exit__(self, *args: Any) -> None:
        self.stop()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run deterministic fake LLM server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    args = parser.parse_args(argv)

    server = FakeLLMServer(host=args.host, port=args.port, model_id=args.model).start()
    print(f"Fake LLM listening on {server.base_url} (model={server.model_id})", flush=True)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
