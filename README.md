# ride-video-agent

An agent that turns raw ride footage (cycling, motorcycle, dashcam, etc.) — plus an
optional GPX telemetry track — into an edited highlight video from a single plain-English
prompt, e.g. *"30 second cinematic highlight reel"* or *"vertical social media reel"*.

## Problem it solves

Editing ride footage is slow and repetitive: scrubbing through long clips to find the
good parts, cutting them down, adding captions/telemetry overlays, and re-exporting for
different platforms (landscape for YouTube, vertical for Reels/Shorts, etc.). This project
automates that loop — it scores the raw footage for "interesting" moments, interprets a
short text prompt into concrete editing decisions, and renders the result.

## Architecture

The pipeline is a two-agent **perceive → plan → execute** system: one agent looks at the
footage and describes it, a second agent decides what to do with that description, and a
plain toolbox module carries out the decision.

```
INPUT: video.mp4  (+ optional ride.gpx)  (+ optional text prompt)
   │
   ▼
1. ANALYSIS — analysis.py (VideoAnalyzer)
   Scores the raw video using scene-change, motion, and audio signals only —
   no understanding of what's actually on screen yet.
   → raw moments: (start, end, score)
   │
   ▼
2. PERCEPTION AGENT — perception_agent.py (PerceptionAgent)
   Extracts a representative frame per moment and captions it with a BLIP
   vision-language model. First agent: turns raw signal into meaning.
   → enriched moments: (start, end, score, description)
   │
   ▼
3. PLANNING AGENT — agent.py (EditAgent)
   Combines the enriched moments with the text prompt to pick a style,
   select clips (favoring both high score and descriptive variety), and
   choose overlays/transitions/format. Second agent: turns meaning into
   editing decisions. Rule-based — no LLM calls.
   → edit_plan: {clips, overlays, transitions, format}
   │
   ▼
4. TOOLBOX — toolbox.py (VideoToolbox)
   Executes edit_plan: cut → overlay → transition → render.
   │
   ▼
OUTPUT: output/*.mp4
```

`main.py` orchestrates steps 1–4 end-to-end via the CLI.

## Project layout

- `main.py` — CLI entry point; wires analysis → Perception Agent → Planning Agent → toolbox together
- `analysis.py` — `VideoAnalyzer`: scene/motion/audio scoring of raw footage
- `perception_agent.py` — `PerceptionAgent`: BLIP-based scene captioning (the Perception Agent)
- `agent.py` — `EditAgent`: rule-based prompt interpretation → `edit_plan` (the Planning Agent)
- `toolbox.py` — `VideoToolbox`: MoviePy-based cut/overlay/transition/render operations
- `sample_data/` — sample input videos and GPX files for local testing
- `output/` — generated output videos

## Setup

```bash
pip install -r requirements.txt
```

**ffmpeg is required** and must be on your `PATH` — MoviePy, pydub, and OpenCV's video
I/O all shell out to it for decoding/encoding. Install it via your OS package manager
(e.g. `brew install ffmpeg`, `apt install ffmpeg`, or download a build for Windows) before
running the pipeline.

**First run downloads a captioning model.** The Perception Agent loads
`Salesforce/blip-image-captioning-base` from Hugging Face the first time it runs (a few
hundred MB) — this needs an internet connection and makes the first run noticeably slower;
subsequent runs reuse the local cache.

## Usage

```bash
# Basic run — prompts interactively if --prompt is omitted
python main.py sample_data/ride.mp4 --prompt "30 second cinematic highlight reel"

# Vertical cut for social media, with GPX-driven speed overlay
python main.py sample_data/ride.mp4 --gpx sample_data/ride.gpx --prompt "vertical social media reel"

# Custom output path
python main.py sample_data/ride.mp4 --prompt "technical telemetry version" --output output/telemetry_cut.mp4

# Dry run — print the edit_plan (chosen clips, overlays, format) without rendering,
# for fast iteration on prompts
python main.py sample_data/ride.mp4 --prompt "2 minute route summary" --dry-run
```

## How the agents interpret prompts

The **Perception Agent** (`PerceptionAgent`) runs first and doesn't look at the prompt at
all — for every moment `VideoAnalyzer` flagged, it grabs a representative frame and
captions it with BLIP, e.g. *"a car turning on a mountain road"*. That's purely content
understanding; it makes no editing decisions.

The **Planning Agent** (`EditAgent.plan()`) then takes the prompt plus those captioned
moments and decides what to do with them. It's **rule-based**, not model-driven — it never
calls an LLM. It works in four deterministic steps:

1. **Keyword matching.** The prompt is scanned against a small table of trigger words to
   pick a style preset — `cinematic`, `social`, `technical`, or `route_summary` — each with
   default duration, transition style, overlays, and output format. Explicit details in the
   prompt (e.g. `"45 seconds"`, `"vertical"`, `"no overlays"`) override the preset's defaults.

2. **Diversity-aware clip selection.** Moments are greedily packed into the target duration,
   but ranked by score *penalized for description similarity* to clips already picked
   (Maximal-Marginal-Relevance style) — so five near-identical "highway driving" moments
   don't crowd out one distinct "mountain switchback" moment. Selected clips are then
   re-ordered chronologically so the final cut still follows the ride's actual timeline.

3. **Overlay/format resolution.** Any remaining explicit keywords (e.g. `"scene caption"`,
   `"speed"`, `"clean"`) add to or clear the style preset's default overlays.
   
4. **Assembly.** The chosen clips (carrying their descriptions forward), overlays,
   transition, and output format are combined into a single `edit_plan` dict that
   `VideoToolbox` executes directly.

This keeps the system fast, deterministic, and fully explainable (see the "Edit Summary"
each run prints) — every decision maps back to a specific keyword, score, or description.

## Challenges encountered

- **Overlay/crop ordering.** Clips need to be resized/cropped to the target aspect ratio
  *before* burning in edge-positioned overlays (timestamp, captions), doing it only at
  final render time silently cropped those overlays out of frame.

- **Transition edge cases.** A fixed 1-second cross-fade could exceed a very short clip's
  own duration, and two caption overlays (scene description + prompt text) defaulting to
  the same screen position rendered on top of each other, both needed explicit handling.

- **Noisy scene detection.** PySceneDetect occasionally flagged sub-second false-positive
  cuts, so a minimum-scene-duration merge step was added to keep clip candidates meaningful.
  
- **GPX/video time alignment.** Reconciling GPX telemetry timestamps with the video's own
  timeline (for the speed overlay) proved harder than expected without embedded sync
  metadata, so it's currently a documented assumption rather than a solved problem.

## Known limitations

- **Keyword matching is brittle.** Phrasing the agent hasn't seen (e.g. "make it punchy and
  fast") won't match any preset and silently falls back to `cinematic` defaults.

- **`target_duration` is a soft ceiling, not a hard trim.** Clips are whole scenes from
  `VideoAnalyzer`'s scene-cut detection — never split or shortened to fit. If a video has
  few or no detected scene cuts (e.g. one continuous take), the only available moment may
  itself be longer than `target_duration`; `_select_clips` always includes at least one
  clip rather than returning an empty edit, so the final runtime can exceed the requested
  duration.

- **Scene descriptions aren't full semantic reasoning.** The Perception Agent can describe
  *what's in frame* ("a car turning on a mountain road"), and the Planning Agent uses that
  for diversity, but neither can act on instructions about content it hasn't scored highly
  ("focus on the descent, not the climb") — selection is still driven by score first.

- **Diversity penalty is a simple heuristic**, not true semantic similarity — it's word
  overlap (Jaccard) between BLIP captions. Paraphrased-but-distinct moments can still get
  penalized, and captions sharing common filler words can be over-penalized.

- **Analysis can be slow on long videos** — scene detection, full-frame audio RMS, and
  per-moment BLIP captioning all scale with video length; there's no caching between runs.

- **Timestamp overlays scale linearly with clip length.** `add_timestamp_overlay` composites
  one `TextClip` per second, so long uninterrupted clips (e.g. `technical`/`route_summary`
  styles, 60–90s target duration) can render noticeably slower than shorter styles.

- **GPX speed overlay assumes the video and GPX log started at the same instant.** Clip
  timestamps come from the video's own timeline, while GPX sample timestamps are offsets
  from the GPX track's first point — there's no reconciliation between the two clocks. If
  the camera and GPX logger were started at different times, the displayed speed will be
  wrong for a given clip without any error being raised.

- **Text overlays depend on system fonts.** MoviePy's `TextClip` needs a usable font;
  pass an explicit font path via `VideoToolbox(font=...)` if the default isn't found.

- **No automated test suite yet** for the pipeline or individual modules.

## Possible next steps

- Restructure around a reasoning-vs-doing split: an LLM-based agent (e.g. a
  vision-capable model like Claude) analyzes the video, captures what's happening, and
  suggests the editing decisions itself; a second, lighter agent takes that suggestion and
  executes it. This trades the current fully deterministic, offline pipeline for richer,
  content-aware editing suggestions, at the cost of API access, latency, and non-determinism.

- Swap the rule-based `EditAgent` for an LLM-based prompt parser (structured output →
  same `edit_plan` schema), enabling free-form prompts without new keyword rules.

- Correlate GPX telemetry with video timestamps to prioritize moments by real-world
  signals (speed spikes, elevation change) alongside the current motion/audio/scene score.

- Cache `VideoAnalyzer`/`PerceptionAgent` results per video file to avoid re-analyzing and
  re-captioning on every prompt.

- Reconcile GPX and video clocks (e.g. an explicit `--gpx-offset` flag, or auto-detecting
  from file metadata) so the speed overlay is trustworthy without matching start times.

- Add a proper test suite (unit tests per module, plus a golden-output integration test).

- Expose the style presets and scoring weights via `config.py` instead of hardcoding them.

- Support editing across multiple source video files in one edit (currently single-video
  input only).
