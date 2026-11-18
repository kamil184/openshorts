# Clip selection

## How many clips a job returns (`clip_selection.py`, `main.get_viral_clips`)

The count is not a setting, it is derived.
`get_viral_clips` builds ~90 s scoring windows over the transcript, scores them in batches, shortlists the best `shortlist_target` (`min(10, duration//90 + 2)`, floor 3) and asks the detail pass for `clip_count_targets(len(shortlist))` clips.
The floor gives unused shortlist windows a second chance to produce clips.

Two invariants keep that chain from leaking:

- **The scoring pass ranks, it does not select.**
  The prompt scores **every** window and the shortlist is the global top N.
  A prompt that says "choose up to 3 windows from this batch" caps the shortlist at `3 * n_batches` instead of the target and starves every source under ~30 min.
  Batches are near-equal (`score_batches`): a trailing batch of one window would return that window whatever its score.
- **The clip floor is enforced in code, not just in the prompt.**
  When the detail pass comes back under `min_clips`, the shortlist windows that produced nothing get one more call for the difference.
  An empty answer is accepted: padding is worse than a short list.

`target_clips` on `/api/process` (dashboard: advanced options) pins both ends via `CLIP_TARGET_MIN` / `CLIP_TARGET_MAX` when the user wants an exact number.

## Silent footage: the vision fallback (`main.get_visual_clips`)

`transcribe_video` raising `NoAudioError`, **or** `speech_is_sparse()` returning true, both set `transcript = None` and route to `get_visual_clips`.
Sparse means under `MIN_SPEECH_WORDS` (8) in total or under `MIN_SPEECH_WORDS_PER_MIN` (5).
Music-only footage and mic-muted recordings transcribe to a handful of stray words; without the second test the picker scores those words and cuts around them.

The vision pass asks the configured video-task model for the same `{"shorts"}` shape (`prompts.VisualResponse`) in the same 15-60 s band, so every later stage is identical.
`CLIP_TARGET_MIN` / `CLIP_TARGET_MAX` apply directly (there are no scoring windows).
The transcript is stored as `{"language": "none", "segments": []}`, so the clips have no subtitles.

**This is the only clip-pipeline stage that sends video to a model.**
  `vision_plan` checks input capabilities: video input takes precedence, otherwise an image-capable model receives timestamped JPEG frames.
A text-only model cannot clip silent footage; the job reports that a model accepting video or images is needed.

Every stretch is re-encoded to a 360p / 1 fps proxy (`_vision_proxy`), including a source that fits in one stretch.
Re-encoding keeps timestamps accurate; stream copying could shift them by a GOP.
If the proxy's audio cannot be encoded, it is retried without audio.
Gemini uploads the proxy through its Files API; compatible endpoints receive it inline.
For frame input, stills are sampled every `VISION_FRAME_INTERVAL` seconds (default 2), capped at `VISION_MAX_FRAMES` (default 120); the actual interval increases if needed to respect the cap.

`VISION_SEGMENT_SECONDS` overrides the stretch size.
Without an override, video input uses the provider's size (2400 seconds for Gemini, 600 for compatible endpoints), while frame input uses interval × frame count (240 seconds by default).
Frame stretches are capped at that frame span; the shared `vision_segments` helper also enforces a 60-second minimum stretch size.
Long sources are split into near-equal stretches without a tiny tail, then picks are offset back to source time, de-duplicated and trimmed by score to `CLIP_TARGET_MAX`.
One failed stretch is logged and skipped; a policy block fails immediately.

`VISION_MAX_SECONDS` sets the approximate coverage budget (default 19200 seconds).
The stretch limit is the ceiling of this budget divided by the stretch size, so rounding can cover up to one extra stretch.
`VISION_MAX_SEGMENTS`, when set, overrides that count directly.
With the default Gemini settings this remains eight stretches of up to 40 minutes, about 5 h 20 total.
Sparse-speech thresholds remain eight words total or five words per minute.
The public `/gta-5-clips`, `/gta-6-clips`, `/wow-forever-clips` and `/platforms/twitch` pages therefore retain their current thresholds and stretch copy.

## Model configuration (`llm_backend.py`)

All converted stages call `llm_backend.generate_json`; stages never select a vendor or infer capabilities from a model name.
`LLM_MODEL` configures text tasks, `LLM_VISION_MODEL` configures frame tasks (defaulting to the text model), and `LLM_VIDEO_MODEL` configures silent footage (defaulting to the vision model).
The default provider remains Gemini when only `GEMINI_API_KEY` is configured, with `GEMINI_MODEL` retaining its default-model override.
`LLM_BASE_URL` selects an OpenAI-compatible endpoint ahead of Gemini; `LLM_PROVIDER` can explicitly select `openai` or `gemini`.
Adding a Gemini key alongside a compatible endpoint does not route frame stages to Gemini.

Compatible endpoints publish capabilities through `GET /models` → `architecture.input_modalities`.
Models without published capabilities are treated as text-only unless `LLM_INPUT_MODALITIES` declares an override, for example `{"vision-model": ["text", "image"]}`.
`LLM_EXTRA_BODY` merges optional JSON request fields, such as gateway routing or reasoning options, without overriding the model or messages.
Every response is validated against the same schemas in `prompts.py`.

`main.score_batch_size` defaults to three transcript windows for compatible endpoints and eight for Gemini; `LLM_SCORE_BATCH` overrides it.
Self-host `/api/process` accepts requests without a per-request Gemini key when a server model is configured, and `/api/config.localLlm` describes the models for the dashboard.
Cloud still requires its resolved Gemini key at the request gate and hides this self-host configuration from the dashboard.
Layout picking, subtitle detection, screen detection and hook grounding use an image-capable vision model, falling back on missing capabilities or provider failures.

## Campaign presets (`presets.py`)

A job submitted to `/api/process` can pass `preset=<name>` to load a YAML profile from the directory named by `PRESETS_DIR`.
Unknown preset names or invalid YAML files fail at request time with HTTP `400`.
The API loads and validates the preset once and passes its content to the job as `CLIP_PRESET_JSON` (forwarded to a remote worker too), so the job never reads the file again.
A job whose preset cannot be loaded fails with that error instead of reporting that no clips were found.
Preset fields expand into job defaults before request validation, and any field sent explicitly by the caller always wins.
When the caller sends only one clip-length bound, the preset's other bound moves just enough to keep the band valid.
A preset supports `clip_format`, `layouts`, `target_clips`, `clip_seconds` (`[min, max]`, mapped to `clip_min_seconds` and `clip_max_seconds`), `auto_hook`, `trim_dead_footage` (see below), and a `captions` block.
A `captions` field left empty is an error: in YAML an unquoted `#` starts a comment, so write hashtags as `'#tag'`.
Under `captions`, `language` overrides the copy language in the selection prompts, `required_tags` lists hashtags that must appear in TikTok and Instagram descriptions, `mention_one_of` lists creator names where at least one must appear, and `extra_instructions` appends free-form guidance to the prompt.
See `presets.example/stream-gaming.yaml` for a generic example profile.
Before `main.py` first writes the job metadata, and again after hook grounding, missing `required_tags` are appended to `video_description_for_tiktok` and `video_description_for_instagram` without modifying compliant text.
Missing `mention_one_of` names across those descriptions or `video_title_for_youtube_short` are never invented and are recorded in `clip['caption_issues']` instead.

## Dead footage at clip edges (`dead_footage.py`)

Clips are picked from the transcript, so an edge can land on a loading screen, a black transition, a frozen frame or a static menu while the speaker keeps talking.
Right after selection, before any clip is cut, `dead_footage.apply` measures the first and last `CLIP_DEAD_TRIM_MAX_SECONDS` (default 12) of every clip and moves the boundary inward past dead footage.
It decodes those stretches as 160x90 greyscale at 4 frames per second with ffmpeg; the facecam box (`GAMING_FACECAM_BOX`) is masked out, so a moving face over a loading screen does not count as action.
A sample is dead when the one-second median frame difference is below 0.15 (frozen), or when the frame is dark and flat (mean luma below 40, spread below 40) with a median difference below 2 (loading screens with a spinner, black frames, fades).
Only a dead stretch of at least 1.5 s touching the clip edge is trimmed; live islands of up to 1.5 s inside it (a scene cut, a second of a menu) do not break it.
The cut never falls inside a word: it lands in a pause of at least 0.3 s that reaches into the dead stretch, so every word removed was spoken entirely over dead footage, and speech running through the whole dead stretch is kept.
An edge moves by at least 1 s or not at all, and never so far that the clip drops under the minimum clip length; when both edges cannot be trimmed, the larger trim wins.
A trimmed clip keeps its earlier range in `dead_trim`.
`CLIP_DEAD_TRIM=0` turns it off and `CLIP_DEAD_TRIM=1` on; otherwise the preset's `trim_dead_footage` decides, and the default is on.
