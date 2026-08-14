"""
toolbox.py

Collection of reusable video-editing tools that the agent can invoke to
build the final ride video, via MoviePy.

Currently implements:
- VideoToolbox: cutting, concatenating, text/overlay burning (including
  Perception Agent-driven scene captions), transitions, and final
  rendering in multiple aspect ratios.

Still to come:
- Frame-level image processing via OpenCV (opencv-python)
- Audio manipulation helpers via pydub
- Wrapping scenedetect and gpxpy functionality into callable "tool" functions

Targets the MoviePy 2.x API (e.g. `subclipped`, `with_position`, `vfx.FadeIn`).
"""

from typing import List, Optional, Tuple

from moviepy import (
    CompositeVideoClip,
    TextClip,
    VideoClip,
    concatenate_videoclips,
)
from moviepy import vfx


class VideoToolbox:
    """
    A grab-bag of independent, reusable MoviePy operations. Each method takes
    a clip (or clips) in and returns a new clip out, so calls can be chained
    or composed by the agent in whatever order the workflow needs, e.g.:

        toolbox = VideoToolbox()
        clip = toolbox.cut_clip(video, 10, 40)
        clip = toolbox.add_timestamp_overlay(clip)
        clip = toolbox.add_fade_transition(clip)
        toolbox.render(clip, "output/highlight.mp4", format="vertical")
    """

    # Target (width, height) for each supported render format.
    FORMAT_SIZES = {
        "landscape": (1920, 1080),  # 16:9 - standard widescreen
        "vertical": (1080, 1920),   # 9:16 - Reels / Shorts / TikTok
        "square": (1080, 1080),     # 1:1  - feed posts
    }

    def __init__(self, font: Optional[str] = None):
        """
        Args:
            font: Optional path to a .ttf/.otf font file used for all text
                overlays. If None, MoviePy falls back to its own default
                font, which may not be available on every system - pass an
                explicit font path if you hit font-related errors.
        """
        self.font = font

    # ------------------------------------------------------------------
    # Cutting & joining
    # ------------------------------------------------------------------

    def cut_clip(self, video: VideoClip, start: float, end: float) -> VideoClip:
        """
        Extract the portion of `video` between `start` and `end` (seconds).

        `start`/`end` often come from an upstream analysis pass (e.g. scene
        detection via ffprobe) whose duration estimate can differ from
        MoviePy's own by a few milliseconds. Clamp against the clip's actual
        duration so those rounding differences don't raise a ValueError.
        """
        start = max(0.0, start)
        end = min(end, video.duration)
        return video.subclipped(start, end)

    def concatenate_clips(
        self, clips: List[VideoClip], transition: str = "none"
    ) -> VideoClip:
        """
        Join multiple clips into a single clip, in the order given.

        transition:
            "none" - clips are placed back-to-back with a hard cut between them.
            "fade" - each clip (after the first) briefly cross-fades in over
                     the tail of the previous one, for a smoother join.
        """
        if not clips:
            raise ValueError("concatenate_clips requires at least one clip")

        if len(clips) == 1:
            # Nothing to cut between or cross-fade into - running a single
            # clip through either transition's "compose" concatenation
            # machinery would still wrap it in mask-based compositing for
            # an effect that has no second clip to blend with, which is
            # pure per-frame overhead (mask arrays recomputed on every
            # frame) for zero visible difference.
            return clips[0]

        if transition == "none":
            return concatenate_videoclips(clips, method="compose")

        if transition == "fade":
            # Cross-fade overlap can't exceed any individual clip's own
            # duration - CrossFadeIn on a clip shorter than `overlap` fades
            # for longer than the clip is visible, and concatenate_videoclips
            # only accepts one uniform `padding` value for every pair below,
            # so the clamp has to be global rather than per-clip.
            overlap = min(1.0, min(clip.duration for clip in clips))
            faded_clips = [clips[0]]
            for clip in clips[1:]:
                faded_clips.append(clip.with_effects([vfx.CrossFadeIn(overlap)]))
            # Negative padding overlaps each clip with the previous one by
            # `overlap` seconds so the cross-fade has something to blend into.
            return concatenate_videoclips(
                faded_clips, method="compose", padding=-overlap
            )

        raise ValueError(f"Unknown transition type: {transition!r}")

    # ------------------------------------------------------------------
    # Overlays
    # ------------------------------------------------------------------

    def add_text_overlay(
        self, clip: VideoClip, text: str, position: str = "bottom"
    ) -> VideoClip:
        """
        Burn a static text caption onto `clip` for its entire duration.

        position: "top", "bottom", or "center" (horizontally always centered).
        """
        position_map = {
            "top": ("center", "top"),
            "bottom": ("center", "bottom"),
            "center": ("center", "center"),
        }
        pos = position_map.get(position, ("center", "bottom"))

        caption = (
            TextClip(
                font=self.font,
                text=text,
                font_size=48,
                color="white",
                stroke_color="black",
                stroke_width=2,
            )
            .with_position(pos)
            .with_duration(clip.duration)
        )

        return CompositeVideoClip([clip, caption])

    def add_timestamp_overlay(self, clip: VideoClip) -> VideoClip:
        """
        Burn a running mm:ss timestamp into the top-right corner, updating
        once per second for the full duration of the clip.

        Implemented as a stack of one-second TextClips (one per second)
        rather than a single dynamic clip, since it only needs MoviePy.
        """
        total_seconds = int(clip.duration) + 1
        timestamp_clips = []

        for second in range(total_seconds):
            minutes, seconds = divmod(second, 60)
            label = f"{minutes:02d}:{seconds:02d}"

            timestamp_clips.append(
                TextClip(font=self.font, text=label, font_size=36, color="white")
                .with_position(("right", "top"))
                .with_start(second)
                .with_duration(1)
            )

        return CompositeVideoClip([clip, *timestamp_clips])

    def add_speed_overlay(
        self,
        clip: VideoClip,
        speed_data: Optional[List[Tuple[float, float]]] = None,
    ) -> VideoClip:
        """
        Burn a "XX km/h" readout into the bottom-left corner, driven by ride
        telemetry.

        speed_data: optional list of (time_seconds, speed_kmh) samples, e.g.
            from a parsed GPX track. Each sample's speed is displayed from
            its timestamp until the next sample's timestamp (or the end of
            the clip for the last sample). If speed_data is None or empty,
            the clip is returned unchanged - this overlay is opt-in.
        """
        if not speed_data:
            return clip

        speed_data = sorted(speed_data, key=lambda sample: sample[0])
        speed_clips = []

        for i, (time, speed_kmh) in enumerate(speed_data):
            next_time = (
                speed_data[i + 1][0] if i + 1 < len(speed_data) else clip.duration
            )
            segment_duration = max(next_time - time, 0.1)
            label = f"{speed_kmh:.0f} km/h"

            speed_clips.append(
                TextClip(font=self.font, text=label, font_size=36, color="yellow")
                .with_position(("left", "bottom"))
                .with_start(time)
                .with_duration(segment_duration)
            )

        return CompositeVideoClip([clip, *speed_clips])

    def add_scene_caption(
        self, clip: VideoClip, description: str, duration: float = 2.0
    ) -> VideoClip:
        """
        Briefly overlay `description` as a caption at the very start of
        `clip`, fading out after `duration` seconds.

        `description` is meant to come from PerceptionAgent's per-moment
        captions (e.g. "highway driving at sunset"), giving each clip a
        quick scene-setting subtitle as it begins. This is deliberately
        different from add_text_overlay, which burns one caption for a
        clip's *entire* duration - this one introduces the scene, then
        fades out and gets out of the way.

        Positioned at the top (unlike add_text_overlay's bottom) so the two
        can be safely combined - the "social" style enables both by default,
        and stacking two captions at the same position would render them
        overlapping each other.

        If `description` is empty, the clip is returned unchanged - this
        overlay is opt-in and only useful when a description is available.
        """
        if not description:
            return clip

        display_duration = max(0.1, min(duration, clip.duration))
        fade_duration = min(0.5, display_duration)

        caption = (
            TextClip(
                font=self.font,
                text=description,
                font_size=40,
                color="white",
                stroke_color="black",
                stroke_width=2,
            )
            .with_position(("center", "top"))
            .with_start(0)
            .with_duration(display_duration)
            .with_effects([vfx.FadeOut(fade_duration)])
        )

        return CompositeVideoClip([clip, caption])

    # ------------------------------------------------------------------
    # Transitions
    # ------------------------------------------------------------------

    def add_fade_transition(self, clip: VideoClip, duration: float = 1.0) -> VideoClip:
        """
        Apply a fade-in from black at the start of `clip` and a fade-out to
        black at the end, `duration` seconds each.
        """
        return clip.with_effects([vfx.FadeIn(duration), vfx.FadeOut(duration)])

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def fit_to_format(self, clip: VideoClip, format: str = "landscape") -> VideoClip:
        """
        Resize/crop `clip` to match a named output format (see
        FORMAT_SIZES).

        Call this on each clip *before* burning in overlays (e.g. right
        after cut_clip), not only at render time - overlay positions like
        "top" or "bottom" are calculated against whatever frame they're
        drawn on, so if a clip is only fit-to-format afterwards (e.g. once
        on the final concatenated clip inside render()), an edge-positioned
        overlay can end up outside the region the crop keeps. This is
        especially visible when the source aspect ratio differs a lot from
        the target format - e.g. portrait ride footage (9:16) rendered to
        landscape (16:9) keeps only the middle ~32% of the frame's height,
        which crops away anything anchored to the top or bottom edge.

        format: "landscape" (1920x1080), "vertical" (1080x1920), or
            "square" (1080x1080). See FORMAT_SIZES.
        """
        if format not in self.FORMAT_SIZES:
            raise ValueError(
                f"Unknown format {format!r}. Choose from {list(self.FORMAT_SIZES)}"
            )

        target_w, target_h = self.FORMAT_SIZES[format]

        if clip.w == target_w and clip.h == target_h:
            # Already the right size (e.g. render()'s safety-net call on a
            # clip that main.py already fit per-clip before overlays) -
            # skip re-running the resize/crop transform. MoviePy composes
            # transforms lazily per frame, so doing this "no-op" fit twice
            # isn't actually free: it doubles the per-frame resize/crop
            # work (and its temporary array allocations) for the entire
            # render, which is wasteful at best and can exhaust memory on
            # long clips at worst.
            return clip

        return self._fit_to_frame(clip, target_w, target_h)

    def render(
        self, clip: VideoClip, output_path: str, format: str = "landscape"
    ) -> str:
        """
        Resize `clip` to match the target output format and write it to
        `output_path`.

        Safe to call even if `clip` was already fit to this format (e.g.
        via fit_to_format() on each source clip before overlays) - fitting
        an already-correctly-sized clip is a no-op, so this doubles as a
        safety net for callers that skip the earlier per-clip fit.
        """
        framed = self.fit_to_format(clip, format)
        framed.write_videofile(output_path)
        return output_path

    def _fit_to_frame(
        self, clip: VideoClip, target_w: int, target_h: int
    ) -> VideoClip:
        """
        Scale `clip` up to fully cover a target_w x target_h frame (like CSS
        `object-fit: cover`), then center-crop the overhang so the result is
        exactly target_w x target_h without distorting the image.
        """
        target_ratio = target_w / target_h
        clip_ratio = clip.w / clip.h

        if clip_ratio > target_ratio:
            # Clip is relatively wider than the target: match heights, crop the sides.
            resized = clip.resized(height=target_h)
        else:
            # Clip is relatively taller than the target: match widths, crop top/bottom.
            resized = clip.resized(width=target_w)

        return resized.cropped(
            x_center=resized.w / 2,
            y_center=resized.h / 2,
            width=target_w,
            height=target_h,
        )
