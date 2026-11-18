# OpenShorts

Open-source (MIT) tool that turns long videos into 9:16 Shorts, TikToks and Reels.
It also generates UGC marketing videos with AI actors and YouTube titles, thumbnails and descriptions.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://docs.docker.com/compose/)

![A podcast and the vertical clip OpenShorts makes of it: both speakers stacked, captions on the seam](screenshots/split-before-after.gif)

<sub>Real OpenShorts output. Source: [Turn the Tables with Dan and Shoshana Jordan](https://www.youtube.com/watch?v=-KbQj_vboOU), Heritage of Faith, licensed CC BY.</sub>

A hosted version runs at [openshorts.app](https://www.openshorts.app/); this repository is the self-hosted app.

## Features

### Clip Generator

Paste a YouTube link or upload a file and get finished 9:16 clips with captions, a hook line, a virality score and per-platform titles.

- **Moment picking**: the transcript and scene boundaries are scored by Gemini (`gemini-3.1-flash-lite` by default) or any OpenAI-compatible model ([local LLM](#run-without-a-google-key)); 15-60 s clips by default.
- **Reframing per scene**: TRACK (MediaPipe + YOLOv8 face tracking), GENERAL (blurred background), SPLIT (two speakers stacked, captions on the seam) and SCREENCAST (screen over presenter).
  The layout is picked per video or forced from the dashboard.
- **Gaming stack**: `clip_format=gaming_stack` renders the whole clip as webcam on top (30%) over a centred gameplay crop (70%); see [Presets and clip formats](#presets-and-clip-formats).
- **Dead-footage trimming**: clip edges move off loading screens, black and frozen frames, never mid-word.
- **Captions**: faster-whisper (or Parakeet on GPU) with word-level timestamps, burned in.
  Every clip ships with the `default` look (Poppins ExtraBold, sentence case, yellow active word); `classic` (Anton uppercase), `talking` (dimmed inactive words, orange accent) and other presets are available in the editor and via `/api/subtitle`.
- **Hooks**: an opening line written from what is said in the clip, in the box-free `headline` style by default (also `pill`, `classic`, `dark`, `yellow`, `red`, `outline`, `outline_yellow`).
- **Clip editor**: source and program monitors with the transcript; click a word to cut, add segments, change framing and re-render.
- **Dubbing** (ElevenLabs, 30+ languages) and Gemini-generated FFmpeg effects.

![A clip with its viral score, titles and edit actions](dashboard/public/screens/app-clipcard.webp)

### AI Shorts (UGC videos)

Describe a product or paste a URL: script (Gemini), actor (Flux 2 Pro or the gallery), voice (ElevenLabs), talking head (Hailuo 2.3 Fast + VEED Lipsync, or Kling Avatar v2 in Premium), b-roll with Ken Burns, FFmpeg composite with subtitles.
Roughly $0.65 per video in Low Cost mode and $2 in Premium, in fal.ai and ElevenLabs usage.

### YouTube Studio

Ten title options with a refinement chat, thumbnails with your face over a background, descriptions with chapter timestamps, and publishing to YouTube.

### Publishing

Post or schedule clips to TikTok, Instagram, YouTube, Facebook, LinkedIn, Threads and Bluesky through [Upload-Post](https://www.upload-post.com) with your own API key.

## Getting started

Requirements: Docker with Compose, 8 GB+ RAM.
API keys are entered in the dashboard's Settings and stored encrypted in the browser:

| Key | Needed for |
|---|---|
| Google Gemini ([get one](https://aistudio.google.com/app/apikey)) | Clipping (unless `LLM_BASE_URL` is set), AI Shorts scripts, YouTube Studio |
| fal.ai | AI Shorts |
| ElevenLabs | Voiceover and dubbing |
| Upload-Post | Social posting |

```bash
git clone https://github.com/mutonby/openshorts.git
cd openshorts
cp .env.example .env   # optional: S3 backup, encoder, models
docker compose up --build
```

Open `http://localhost:5175`.
The API listens on `http://localhost:8000` with OpenAPI docs at `/docs`.

### GPU acceleration (NVIDIA, optional)

The default image is CPU-only: an 8-minute video takes 5 to 8 minutes, about one minute with an NVENC-capable card.

Install the NVIDIA driver and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) on the host:

```bash
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi   # sanity check
```

On Windows use Docker Desktop with the WSL2 backend and the Windows NVIDIA driver.

Create `docker-compose.override.yml` next to `docker-compose.yml`.
`GPU: "1"` adds cuBLAS/cuDNN and onnxruntime-gpu (~2 GB); the `video` capability is required for NVENC.

```yaml
services:
  backend:
    build:
      context: .
      args:
        GPU: "1"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu, video]
```

Then in `.env`:

```
WHISPER_MODEL=large-v3-turbo
WHISPER_DEVICE=cuda
WHISPER_COMPUTE=float16
FFMPEG_ENCODER=auto           # probes h264_nvenc at startup, falls back to x264
TRANSCRIBE_BACKEND=parakeet   # optional: ~2x faster, 25 European languages, falls back to whisper
ASR_GPU_CONCURRENCY=1
```

Verify:

```bash
docker compose up --build -d
docker exec openshorts-backend nvidia-smi -L
docker exec openshorts-backend ffmpeg -hide_banner -f lavfi -i testsrc=size=256x256:rate=1 -frames:v 1 -c:v h264_nvenc -f null -
```

The first job's log reports the chosen encoder and transcription device.
A CUDA error in whisper retries once on CPU.
8 GB of VRAM fits `large-v3-turbo` fp16 plus the detection models.

### Run without a Google key

Set `LLM_BASE_URL` to an OpenAI-compatible `/v1` endpoint (Ollama, LM Studio, vLLM, llama.cpp server, LocalAI, OpenRouter) and clipping no longer needs a Gemini key.

```bash
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=qwen2.5:14b
# LLM_API_KEY=...              # only if the endpoint checks a bearer token
# LLM_VISION_MODEL=vision-model
# LLM_VIDEO_MODEL=video-model
# LLM_INPUT_MODALITIES={"vision-model":["text","image"]}
```

- `LLM_VISION_MODEL` defaults to `LLM_MODEL`, and `LLM_VIDEO_MODEL` to the vision model.
- Capabilities come from the endpoint's `/models` response (`architecture.input_modalities`); `LLM_INPUT_MODALITIES` overrides them, and unknown means text-only.
- Layout picking, screen detection, subtitle-band detection and hook grounding need an image-capable model and skip or fall back without one.
  Silent footage uses video input if supported, otherwise timestamped frames; a text-only model cannot clip it.
- Scoring sends three transcript windows per call (eight for Gemini; `LLM_SCORE_BATCH` overrides).
  With Ollama, raise the context (`OLLAMA_CONTEXT_LENGTH=16384` or `num_ctx`) to avoid silent truncation.
- `LLM_BASE_URL` takes precedence over `GEMINI_API_KEY`; `LLM_PROVIDER=openai|gemini` forces one.
  The editor, thumbnail and standalone generation features still use their own provider keys.

## Presets and clip formats

A preset is a YAML file of job defaults, named by the `preset` field of `POST /api/process`.
Presets are read from the directory in `PRESETS_DIR` (on the backend, so mount it into the container); keep real ones outside the repository.
See [`presets.example/stream-gaming.yaml`](presets.example/stream-gaming.yaml):

```yaml
clip_format: gaming_stack
layouts: [screencast]
target_clips: 6
clip_seconds: [20, 60]
auto_hook: true
trim_dead_footage: true
captions:
  language: en
  required_tags: ["#tag1", "#tag2"]
  mention_one_of: [CreatorOne, CreatorTwo]
  extra_instructions: "Keep captions upbeat and focused on the gameplay highlight."
```

Allowed keys: `clip_format`, `facecam_box`, `layouts`, `target_clips`, `clip_seconds`, `auto_hook`, `trim_dead_footage`, `captions`.
A field sent explicitly in the request overrides the preset's value.
The `captions` rules are added to the caption prompt, and required tags and mentions are enforced on the generated descriptions.
The validated preset is passed to the job, so later edits to the file do not affect running jobs.

`clip_format` (request field, preset key or server-wide `CLIP_FORMAT`) is `default` or `gaming_stack`.
`gaming_stack` puts the webcam on top (`GAMING_CAMERA_RATIO`, default 0.30) over a static centre gameplay crop; set `GAMING_FACECAM_BOX=x,y,w,h` (normalised 0-1), or `facecam_box` in a preset, to skip webcam detection.

Dead-footage trimming is on by default.
`CLIP_DEAD_TRIM=0` or `1` overrides it for the whole server; otherwise the preset's `trim_dead_footage` decides.
`CLIP_DEAD_TRIM_MAX_SECONDS` caps how much one edge may lose.

## API, MCP and CLI

### MCP server

The backend serves [MCP](https://modelcontextprotocol.io) at `/mcp`:

```bash
claude mcp add --transport http openshorts http://localhost:8000/mcp   # self-hosted, no key
claude mcp add openshorts -- python mcp_stdio.py                       # stdio, no web server needed
```

The hosted endpoint is `https://mcp.openshorts.app/mcp`, with an `osk_...` key as `Authorization: Bearer` or OAuth from claude.ai and ChatGPT connectors.

Tools: `process_video` (URL or `upload_id`; `captions: false` when the source already has subtitles, `auto_hook: false` to skip the hook), `create_upload` (PUT a local file, then process), `get_job_status`, `list_clips`, `get_quota`, `add_subtitles`, `recut_clip`, `publish_clip`.

### REST API and webhooks

```bash
curl -X POST http://localhost:8000/api/process \
  -H "Content-Type: application/json" \
  -d '{"url": "https://youtube.com/watch?v=...", "acknowledged": true,
       "preset": "stream-gaming",
       "webhook_url": "https://your-server.example/hooks/openshorts"}'
```

`acknowledged: true` confirms you have the rights to the source.
With `webhook_url` (and optional `webhook_secret`) the job sends exactly one `POST` when it finishes:

```json
{"event": "job.completed", "job_id": "…",
 "clips": [{"index": 0, "title": "…", "video_url": "…", "download_url": "…"}]}
```

`event` is `job.failed` on failure.
With a secret the body is signed: `X-OpenShorts-Signature: sha256=<hmac-sha256(body)>`.

### CLI

```bash
pip install openshorts   # or: uvx openshorts
export OPENSHORTS_API_URL=http://localhost:8000   # self-hosted; or OPENSHORTS_API_KEY=osk_... for hosted
openshorts process "https://youtube.com/watch?v=..." --wait
openshorts clips <job_id>
openshorts publish <job_id> 0 --platforms tiktok,youtube
```

See [`cli/README.md`](cli/README.md).

### Agent skill and n8n

[`skills/openshorts/`](skills/openshorts/) is an [Agent Skills](https://agentskills.io) skill: copy it into your agent's skills directory (for Claude Code, `~/.claude/skills/`).
An importable n8n workflow lives in [`examples/n8n/`](examples/n8n/).

## Server configuration

Set in `.env`; [`.env.example`](.env.example) documents the full list.

| Variable | Description |
|---|---|
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | S3 credentials (optional) |
| `AWS_REGION` | S3 region |
| `AWS_S3_BUCKET` | Private bucket for clip backup |
| `AWS_S3_PUBLIC_BUCKET` | Public bucket for the gallery and avatars |
| `MAX_CONCURRENT_JOBS` | Jobs processed at once (default 5) |
| `JOB_RETENTION_SECONDS` | How long finished jobs are kept (default 86400 self-hosted) |
| `PRESETS_DIR` | Directory of job presets |
| `CLIP_FORMAT` | Server-wide clip format: `default` or `gaming_stack` |
| `GAMING_CAMERA_RATIO`, `GAMING_FACECAM_BOX` | Gaming stack camera height and fixed webcam box |
| `CLIP_DEAD_TRIM`, `CLIP_DEAD_TRIM_MAX_SECONDS` | Dead-footage trimming switch and per-edge cap |
| `WHISPER_MODEL`, `WHISPER_DEVICE`, `WHISPER_COMPUTE` | Transcription model and device |
| `TRANSCRIBE_BACKEND` | `whisper` (default) or `parakeet` (GPU image) |
| `FFMPEG_ENCODER` | `x264` (default), `nvenc` or `auto` |
| `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY` | OpenAI-compatible endpoint for the clip pipeline |
| `LLM_PROVIDER` | Force `openai` or `gemini` |
| `LLM_VISION_MODEL`, `LLM_VIDEO_MODEL` | Image and silent-footage models |
| `LLM_INPUT_MODALITIES` | JSON map of model ID to input modalities |
| `LLM_EXTRA_BODY` | JSON of extra request fields for the compatible endpoint |
| `LLM_SCORE_BATCH` | Transcript windows per scoring call (default 3 compatible, 8 Gemini) |
| `LLM_TIMEOUT` | Compatible-endpoint timeout in seconds (default 600) |
| `VISION_FRAME_INTERVAL` | Seconds between stills for image input (default 2) |
| `VISION_MAX_FRAMES` | Stills per stretch (default 120) |
| `VISION_MAX_SECONDS` | Watched-duration budget (default 19200), rounded up to whole stretches |
| `VISION_SEGMENT_SECONDS` | Stretch size; default 2400 for Gemini video, 600 for compatible video, interval × frames for images |
| `VISION_MAX_SEGMENTS` | Stretch-count limit; otherwise derived from the budget |

Containers run as a non-root `appuser`, and uploads are capped at 2 GB.

## Upload-Post setup

1. Register at [app.upload-post.com](https://app.upload-post.com/login).
2. Create a profile under [Manage Users](https://app.upload-post.com/manage-users) and connect your social accounts.
3. Create a key under [API Keys](https://app.upload-post.com/api-keys) and paste it into Settings.

## License

MIT, except [`cloud/`](cloud/LICENSE) (billing and hosted-service code behind `BILLING_ENABLED`), which is source-available under the OpenShorts Commercial License: you may read, modify and self-host it for personal or internal use, but not offer it to third parties as a paid or hosted service.
The core app never requires that directory.
