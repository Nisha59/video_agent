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

```
   ┌─────────────┐        ┌──────────────────┐
   │   INPUT     │        │                  │
   │ video.mp4   │───────▶│    ANALYSIS      │
   │ ride.gpx    │        │  (analysis.py)   │
   │ (optional)  │        │  VideoAnalyzer   │
   └─────────────┘        └────────┬─────────┘
                                    │ scored moments
                                    │ (start, end, score)
                                    ▼
   ┌─────────────┐        ┌──────────────────┐
   │ text prompt │───────▶│      AGENT       │
   │ "cinematic  │        │   (agent.py)     │
   │  reel..."   │        │    EditAgent     │
   └─────────────┘        └────────┬─────────┘
                                    │ edit_plan
                                    │ {clips, overlays,
                                    │  transitions, format}
                                    ▼
                           ┌──────────────────┐
                           │     TOOLBOX      │
                           │  (toolbox.py)    │
                           │  VideoToolbox    │
                           │ cut → overlay →  │
                           │ transition →     │
                           │ render           │
                           └────────┬─────────┘
                                    ▼
                           ┌──────────────────┐
                           │     OUTPUT       │
                           │ output/*.mp4     │
                           └──────────────────┘

   main.py orchestrates the pipeline above end-to-end via the CLI.
```

## Project layout

- `main.py` — CLI entry point; wires analysis → agent → toolbox together
- `agent.py` — `EditAgent`: rule-based prompt interpretation → `edit_plan`
- `analysis.py` — `VideoAnalyzer`: scene/motion/audio scoring of raw footage
- `toolbox.py` — `VideoToolbox`: MoviePy-based cut/overlay/transition/render operations
- `config.py` — configuration and default paths
- `utils/` — shared helper utilities
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

## How the agent interprets prompts

`EditAgent.plan()` is **rule-based**, not model-driven — it never calls an LLM. It works
in three deterministic steps:

1. **Keyword matching.** The prompt is scanned against a small table of trigger words to
   pick a style preset — `cinematic`, `social`, `technical`, or `route_summary` — each with
   default duration, transition style, overlays, and output format. Explicit details in the
   prompt (e.g. `"45 seconds"`, `"vertical"`, `"no overlays"`) override the preset's defaults.
2. **Score-based clip selection.** The `(start, end, score)` moments from `VideoAnalyzer`
   are greedily packed into the target duration, highest-scoring first, then re-ordered
   chronologically so the final cut still follows the ride's actual timeline.
3. **Assembly.** The chosen clips, overlays, transition, and output format are combined into
   a single `edit_plan` dict that `VideoToolbox` executes directly.

This keeps the system fast, deterministic, and fully explainable (see the "Edit Summary"
each run prints) — every decision maps back to a specific keyword or score.

## Known limitations

- **Keyword matching is brittle.** Phrasing the agent hasn't seen (e.g. "make it punchy and
  fast") won't match any preset and silently falls back to `cinematic` defaults.
- **No semantic understanding.** The agent can't reason about *content* ("focus on the
  descent, not the climb") — only score and duration.
- **Clip selection is score-greedy, not narrative-aware.** It doesn't consider pacing,
  variety, or avoiding near-duplicate consecutive clips.
- **Analysis can be slow on long videos** — scene detection and full-frame audio RMS both
  scale with video length; there's no caching between runs.
- **Text overlays depend on system fonts.** MoviePy's `TextClip` needs a usable font;
  pass an explicit font path via `VideoToolbox(font=...)` if the default isn't found.
- **No automated test suite yet** for the pipeline or individual modules.

## Possible next steps

- Swap the rule-based `EditAgent` for an LLM-based prompt parser (structured output →
  same `edit_plan` schema), enabling free-form prompts without new keyword rules.
- Correlate GPX telemetry with video timestamps to prioritize moments by real-world
  signals (speed spikes, elevation change) alongside the current motion/audio/scene score.
- Cache `VideoAnalyzer` results per video file to avoid re-analyzing on every prompt.
- Add a proper test suite (unit tests per module, plus a golden-output integration test).
- Expose the style presets and scoring weights via `config.py` instead of hardcoding them.
