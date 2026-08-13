"""
main.py

Entry point for the ride-video-agent application. Wires the other modules
together into a single command-line pipeline:

    1. Load a video (and optional GPX track) from the command line.
    2. analysis.VideoAnalyzer scores the video for interesting moments.
    3. A user prompt (flag or interactive) describes the desired edit.
    4. agent.EditAgent turns that prompt + the scored moments into an edit_plan.
    5. toolbox.VideoToolbox executes the edit_plan (cut / overlay / transition / render).
    6. The result is saved to output/ and a summary of the decisions is printed.

Run `python main.py --help` for usage, or pass --dry-run to see the edit_plan
without actually rendering anything (skips step 5 entirely).
"""

import argparse
import os
import sys
from typing import List, Tuple

import gpxpy
from moviepy import VideoFileClip

from agent import EditAgent
from analysis import VideoAnalyzer
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
        help="Only compute and print the edit_plan; skip cutting/rendering (fast testing).",
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


def print_summary(edit_plan: dict, analyzed_moments: list) -> None:
    """Print a human-readable explanation of the decisions in edit_plan."""
    # Look up each chosen clip's original interest score (by its start time)
    # so we can explain *why* it was picked, not just that it was.
    scores_by_start = {start: score for start, end, score in analyzed_moments}

    print("\n=== Edit Summary ===")
    print(f"Style:          {edit_plan['style']}")
    print(f"Target duration:{edit_plan['target_duration']:6.1f}s")
    print(f"Output format:  {edit_plan['output_format']}")
    print(f"Transitions:    {edit_plan['transitions']}")
    print(f"Overlays:       {', '.join(edit_plan['overlays']) or 'none'}")

    print(f"\nClips chosen ({len(edit_plan['clips'])}):")
    total_duration = 0.0
    for start, end in edit_plan["clips"]:
        duration = end - start
        total_duration += duration
        score = scores_by_start.get(start)
        reason = f"interest score {score:.2f}" if score is not None else "interest score n/a"
        print(f"  {start:7.2f}s - {end:7.2f}s  ({duration:5.2f}s)  {reason}")

    print(f"\nTotal runtime: {total_duration:.2f}s (target was {edit_plan['target_duration']:.1f}s)")


def main() -> None:
    args = parse_args()

    # --- Step 1: validate inputs up front so we fail fast with a clear message ---
    if not os.path.exists(args.video):
        print(f"Error: video file not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    if args.gpx and not os.path.exists(args.gpx):
        print(f"Error: GPX file not found: {args.gpx}", file=sys.stderr)
        sys.exit(1)

    # --- Step 2: analyze the video for interesting moments ---
    print(f"Analyzing {args.video} ...")
    try:
        analyzer = VideoAnalyzer(args.video)
        analyzed_moments = analyzer.analyze()
    except Exception as exc:
        print(f"Error: failed to analyze video ({exc}).", file=sys.stderr)
        print("Check that ffmpeg is installed and the file is a valid video.", file=sys.stderr)
        sys.exit(1)

    if not analyzed_moments:
        print("Error: no moments were detected in this video - nothing to edit.", file=sys.stderr)
        sys.exit(1)

    # --- Step 3: get the user's editing request ---
    prompt = get_prompt(args.prompt)

    # --- Step 4: turn the prompt + analyzed moments into a structured edit plan ---
    edit_agent = EditAgent()
    edit_plan = edit_agent.plan(prompt, analyzed_moments)

    if args.dry_run:
        # Fast path for testing: show the plan and stop before touching MoviePy.
        print_summary(edit_plan, analyzed_moments)
        print("\n(--dry-run: skipping render)")
        return

    if not edit_plan["clips"]:
        print("Error: edit_plan has no clips to render.", file=sys.stderr)
        sys.exit(1)

    # --- Step 5: execute the edit plan with VideoToolbox ---
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
    for start, end in edit_plan["clips"]:
        clip = toolbox.cut_clip(source_video, start, end)

        if "timestamp" in edit_plan["overlays"]:
            clip = toolbox.add_timestamp_overlay(clip)

        if "speed" in edit_plan["overlays"] and speed_data:
            clip = toolbox.add_speed_overlay(clip, speed_data_for_clip(speed_data, start, end))

        if "text" in edit_plan["overlays"]:
            clip = toolbox.add_text_overlay(clip, prompt, position="bottom")

        processed_clips.append(clip)

    # Joining the clips also applies the requested transition style
    # ("fade" cross-fades between clips, "none" is a hard cut).
    final_clip = toolbox.concatenate_clips(processed_clips, transition=edit_plan["transitions"])

    # --- Step 6: render and save to output/, then report what was done ---
    os.makedirs("output", exist_ok=True)
    output_path = build_output_path(args.output, args.video, edit_plan["style"])

    print(f"Rendering to {output_path} (format={edit_plan['output_format']}) ...")
    toolbox.render(final_clip, output_path, format=edit_plan["output_format"])

    source_video.close()

    print_summary(edit_plan, analyzed_moments)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
