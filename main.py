"""
main.py

Entry point for the ride-video-agent application. Wires the project's
two-agent pipeline together into a single command-line run:

    1. Analysis (analysis.VideoAnalyzer) - scores the raw video for
       interesting moments using numeric signals only (scene changes,
       motion, audio energy). No content understanding yet.
    2. Perception Agent (perception_agent.PerceptionAgent) - looks at a
       representative frame from each moment and captions it with a
       vision-language model, producing enriched moments (numeric score +
       text description).
    3. Planning Agent (agent.EditAgent) - takes the enriched moments plus
       an optional user prompt and decides editing actions: which clips to
       keep (favoring both high score and descriptive variety), which
       style/overlays/format to use, producing an edit_plan.
    4. Execution (toolbox.VideoToolbox) - runs the edit_plan: cut, overlay,
       transition, render.

    The result is saved to output/ and a summary of every decision is
    printed, including which agent made it.

Run `python main.py --help` for usage, or pass --dry-run to see the full
edit_plan (including descriptions) without actually rendering anything
(skips step 4 entirely).
"""

import argparse
import gc
import os
import sys
from typing import List, Tuple

import gpxpy
import torch
from moviepy import VideoFileClip

from agent import EditAgent
from analysis import VideoAnalyzer
from perception_agent import PerceptionAgent
from toolbox import VideoToolbox

SpeedSample = Tuple[float, float]  # (time_offset_seconds, speed_kmh)


def parse_args() -> argparse.Namespace:
    """Define and parse the command-line interface."""
    parser = argparse.ArgumentParser(
        description="Turn raw ride footage into an edited highlight video."
    )
    parser.add_argument("video", help="Path to the input video file.")
    parser.add_argument(
        "--gpx",
        help="Optional path to a GPX file with ride telemetry, used for speed overlays.",
    )
    parser.add_argument(
        "--prompt",
        help=(
            "Optional editing request, e.g. '30 second cinematic highlight reel'. "
            "If omitted, the agent runs automatically with a sensible default "
            "(prompting interactively only when run from a terminal)."
        ),
    )
    parser.add_argument(
        "--output",
        help="Output video path. Defaults to output/<video_name>_<style>.mp4",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Only compute and print the full edit_plan (including Perception "
            "Agent descriptions); skip cutting/rendering (fast testing)."
        ),
    )
    return parser.parse_args()


DEFAULT_PROMPT = "cinematic highlight reel"


def get_prompt(cli_prompt: str) -> str:
    """
    Use the --prompt flag if given. Otherwise, the prompt is optional by
    design (per the brief, a video alone should be enough to produce an
    automatic edit): if stdin is an interactive terminal, offer the user a
    chance to type one; if not (e.g. running non-interactively/scripted),
    skip straight to the default so the agent never blocks waiting for input.
    """
    if cli_prompt:
        return cli_prompt

    if not sys.stdin.isatty():
        print(f"No --prompt given - using default '{DEFAULT_PROMPT}'.")
        return DEFAULT_PROMPT

    prompt = input(
        "Describe the edit you want (e.g. '30 second cinematic highlight reel'), "
        "or press Enter to use the default: "
    ).strip()

    if not prompt:
        print(f"No prompt given - falling back to a default '{DEFAULT_PROMPT}'.")
        prompt = DEFAULT_PROMPT

    return prompt


def extract_speed_data(gpx_path: str) -> List[SpeedSample]:
    """
    Parse a GPX file into a list of (time_offset_seconds, speed_kmh) samples,
    suitable for VideoToolbox.add_speed_overlay(). Speed is derived from the
    distance and time between consecutive track points (there's no speed
    field in raw GPX). Returns [] if the file has no usable timed points.
    """
    with open(gpx_path, "r", encoding="utf-8") as gpx_file:
        gpx = gpxpy.parse(gpx_file)

    points = [
        point
        for track in gpx.tracks
        for segment in track.segments
        for point in segment.points
    ]

    if len(points) < 2 or points[0].time is None:
        return []

    start_time = points[0].time
    speed_data: List[SpeedSample] = []

    for prev_point, curr_point in zip(points, points[1:]):
        if prev_point.time is None or curr_point.time is None:
            continue

        elapsed = (curr_point.time - prev_point.time).total_seconds()
        if elapsed <= 0:
            continue

        distance_m = prev_point.distance_3d(curr_point) or prev_point.distance_2d(curr_point) or 0.0
        speed_kmh = (distance_m / elapsed) * 3.6
        time_offset = (curr_point.time - start_time).total_seconds()
        speed_data.append((time_offset, speed_kmh))

    return speed_data


def speed_data_for_clip(
    speed_data: List[SpeedSample], clip_start: float, clip_end: float
) -> List[SpeedSample]:
    """
    Slice the full-ride speed_data down to just the samples inside one clip,
    and shift their timestamps so 0 = the start of that clip (which is what
    add_speed_overlay expects, since it works on the clip's own timeline).
    """
    return [
        (time - clip_start, speed)
        for time, speed in speed_data
        if clip_start <= time <= clip_end
    ]


def build_output_path(cli_output: str, video_path: str, style: str) -> str:
    """Resolve the final output path, defaulting into output/ if not given."""
    if cli_output:
        return cli_output

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join("output", f"{video_name}_{style}.mp4")


def print_summary(edit_plan: dict) -> None:
    """
    Print a human-readable explanation of the decisions in edit_plan.

    Each clip's score and description are read straight off edit_plan
    itself (not looked up separately) since the Planning Agent's clips
    already carry both forward from the Perception Agent's output - that's
    what makes it visible here *why* each clip was chosen.
    """
    print("\n=== Edit Summary ===")
    print(f"Style:          {edit_plan['style']}")
    print(f"Target duration:{edit_plan['target_duration']:6.1f}s")
    print(f"Output format:  {edit_plan['output_format']}")
    print(f"Transitions:    {edit_plan['transitions']}")
    print(f"Overlays:       {', '.join(edit_plan['overlays']) or 'none'}")

    print(f"\nClips chosen ({len(edit_plan['clips'])}):")
    total_duration = 0.0
    for clip in edit_plan["clips"]:
        start, end = clip["start"], clip["end"]
        duration = end - start
        total_duration += duration

        score = clip.get("score")
        score_str = f"{score:.2f}" if score is not None else "n/a"
        description = clip.get("description") or "(no description)"

        print(
            f"  {start:7.2f}s - {end:7.2f}s  ({duration:5.2f}s)  "
            f"score={score_str}  \"{description}\""
        )

    print(f"\nTotal runtime: {total_duration:.2f}s (target was {edit_plan['target_duration']:.1f}s)")


def main() -> None:
    args = parse_args()

    # --- Step 0: validate inputs up front so we fail fast with a clear message ---
    if not os.path.exists(args.video):
        print(f"Error: video file not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    if args.gpx and not os.path.exists(args.gpx):
        print(f"Error: GPX file not found: {args.gpx}", file=sys.stderr)
        sys.exit(1)

    # --- Step 1: Analysis - score the raw video for interesting moments ---
    # Numeric signals only (scene changes, motion, audio energy); no
    # understanding of what's actually in frame yet.
    print(f"Analyzing {args.video} ...")
    try:
        analyzer = VideoAnalyzer(args.video)
        raw_moments = analyzer.analyze()
    except Exception as exc:
        print(f"Error: failed to analyze video ({exc}).", file=sys.stderr)
        print("Check that ffmpeg is installed and the file is a valid video.", file=sys.stderr)
        sys.exit(1)

    if not raw_moments:
        print("Error: no moments were detected in this video - nothing to edit.", file=sys.stderr)
        sys.exit(1)

    # --- Step 2: Perception Agent - describe what's actually happening in each moment ---
    print("Loading Perception Agent (captioning model) ...")
    try:
        perception_agent = PerceptionAgent()
        enriched_moments = perception_agent.enrich_moments(args.video, raw_moments)
    except Exception as exc:
        print(f"Error: Perception Agent failed to describe moments ({exc}).", file=sys.stderr)
        sys.exit(1)

    # The captioning model (~1GB+ with torch's runtime overhead) has done its
    # job - drop it before the memory-hungry MoviePy/ffmpeg render starts, so
    # it isn't sitting resident in RAM competing with encoding for the whole
    # video's duration.
    del perception_agent
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"[Perception Agent] Found {len(enriched_moments)} moments and described them")

    # --- Step 3: get the user's editing request, then plan the edit ---
    prompt = get_prompt(args.prompt)

    edit_agent = EditAgent()
    edit_plan = edit_agent.plan(prompt, enriched_moments)

    print(
        f"[Planning Agent] Selected {len(edit_plan['clips'])} clips based on "
        "scores + descriptions + user prompt"
    )

    if args.dry_run:
        # Fast path for testing: show the full plan (including descriptions)
        # and stop before touching MoviePy.
        print_summary(edit_plan)
        print("\n(--dry-run: skipping render)")
        return

    if not edit_plan["clips"]:
        print("Error: edit_plan has no clips to render.", file=sys.stderr)
        sys.exit(1)

    # --- Step 4: execute the edit plan with VideoToolbox ---
    speed_data: List[SpeedSample] = []
    if args.gpx and "speed" in edit_plan["overlays"]:
        try:
            speed_data = extract_speed_data(args.gpx)
        except Exception as exc:
            print(f"Warning: could not parse GPX file ({exc}); skipping speed overlay.")

    try:
        source_video = VideoFileClip(args.video)
    except Exception as exc:
        print(f"Error: could not open video file ({exc}).", file=sys.stderr)
        sys.exit(1)

    toolbox = VideoToolbox()

    print("\nBuilding edit ...")
    processed_clips = []
    for clip_info in edit_plan["clips"]:
        start, end = clip_info["start"], clip_info["end"]
        description = clip_info.get("description", "")

        clip = toolbox.cut_clip(source_video, start, end)
        # Fit to the target frame *before* overlays, so overlay positions
        # (e.g. "top", "bottom") land inside the region the final crop
        # actually keeps - see fit_to_format()'s docstring for why doing
        # this only at render time can crop overlays away entirely.
        clip = toolbox.fit_to_format(clip, edit_plan["output_format"])

        if "timestamp" in edit_plan["overlays"]:
            clip = toolbox.add_timestamp_overlay(clip)

        if "speed" in edit_plan["overlays"] and speed_data:
            clip = toolbox.add_speed_overlay(clip, speed_data_for_clip(speed_data, start, end))

        if "text" in edit_plan["overlays"]:
            clip = toolbox.add_text_overlay(clip, prompt, position="bottom")

        if "scene_caption" in edit_plan["overlays"]:
            clip = toolbox.add_scene_caption(clip, description)

        processed_clips.append(clip)

    # Joining the clips also applies the requested transition style
    # ("fade" cross-fades between clips, "none" is a hard cut).
    final_clip = toolbox.concatenate_clips(processed_clips, transition=edit_plan["transitions"])

    # --- Step 5: render and save to output/, then report what was done ---
    os.makedirs("output", exist_ok=True)
    output_path = build_output_path(args.output, args.video, edit_plan["style"])

    print(f"Rendering to {output_path} (format={edit_plan['output_format']}) ...")
    toolbox.render(final_clip, output_path, format=edit_plan["output_format"])

    source_video.close()

    print_summary(edit_plan)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
