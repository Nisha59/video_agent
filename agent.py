"""
agent.py

Defines the agent responsible for turning a user's plain-text request into a
concrete, executable edit plan for the ride video.

Reasoning approach
-------------------
This is a *rule-based* agent, not a model-driven one: it never calls an LLM
or does anything probabilistic. Instead, `EditAgent.plan()` follows three
deterministic steps:

1. Keyword matching - the prompt is lowercased and scanned for known trigger
   words (see STYLE_PRESETS and the _*_KEYWORDS tables below) to decide the
   editing `style`, `target_duration`, `output_format`, and `overlays`. Each
   style preset supplies sensible defaults for anything the prompt doesn't
   mention explicitly, and explicit keywords in the prompt can override
   those defaults (e.g. a "technical" prompt that also says "vertical"
   still renders vertically).

2. Score-based clip selection - `analyzed_moments` (the
   (start, end, score) tuples produced by analysis.VideoAnalyzer) are
   greedily packed into the available `target_duration`, most interesting
   first, then re-ordered chronologically so the final edit still follows
   the ride's natural timeline.

3. Assembly - the selected clips, overlays, transition style, and output
   format are combined into a single `edit_plan` dict that agent.py's
   caller can hand straight to toolbox.VideoToolbox to render.

Prompt keyword -> editing decision mapping
-------------------------------------------
    "cinematic" / "highlight" / "epic" / "movie"
        -> style="cinematic": longer landscape cut, fade transitions, no
           telemetry overlays (clean, story-driven look).
    "social" / "reel" / "tiktok" / "instagram" / "shorts" / "vertical"
        -> style="social": short vertical cut with fades and a text overlay,
           tuned for social feeds.
    "technical" / "telemetry" / "data" / "stats" / "analysis"
        -> style="technical": longer landscape cut, hard cuts (no fades),
           timestamp + speed overlays.
    "route" / "summary" / "map" / "journey" / "overview"
        -> style="route_summary": long landscape cut, hard cuts, timestamp +
           speed overlays, meant to cover the whole ride rather than just
           the highlights.

    "<N> second(s)/sec(s)" or "<N> minute(s)/min(s)"
        -> overrides the style's default target_duration.
    "vertical" / "reel" / "tiktok" / "shorts" / "story"
        -> overrides output_format to "vertical".
    "square"
        -> overrides output_format to "square".
    "landscape" / "widescreen" / "horizontal"
        -> overrides output_format to "landscape".
    "timestamp" / "clock"       -> adds a "timestamp" overlay.
    "speed" / "telemetry" / "gps" -> adds a "speed" overlay.
    "caption" / "text overlay" / "title" -> adds a "text" overlay.
    "no overlay" / "no text" / "clean" -> clears all overlays.

If nothing matches, the agent falls back to the "cinematic" style preset.
"""

import re
from typing import Any, Dict, List, Tuple

# A moment as produced by analysis.VideoAnalyzer.analyze(): (start, end, score).
Moment = Tuple[float, float, float]

# A selected clip in the final edit plan: (start, end) — no score, already chosen.
Clip = Tuple[float, float]

EditPlan = Dict[str, Any]


class EditAgent:
    """
    Turns a natural-language editing request into a structured edit plan.

    Usage:
        agent = EditAgent()
        edit_plan = agent.plan("30 second cinematic highlight reel", analyzed_moments)
    """

    # Default settings per editing style. `keywords` are the prompt trigger
    # words that select this style; everything else is a default that can
    # still be overridden by other keywords in the prompt (see module
    # docstring above).
    STYLE_PRESETS: Dict[str, Dict[str, Any]] = {
        "cinematic": {
            "keywords": ["cinematic", "highlight", "epic", "movie"],
            "target_duration": 30,
            "transitions": "fade",
            "overlays": [],
            "output_format": "landscape",
        },
        "social": {
            "keywords": ["social", "reel", "tiktok", "instagram", "shorts"],
            "target_duration": 15,
            "transitions": "fade",
            "overlays": ["text"],
            "output_format": "vertical",
        },
        "technical": {
            "keywords": ["technical", "telemetry", "data", "stats", "analysis"],
            "target_duration": 60,
            "transitions": "none",
            "overlays": ["timestamp", "speed"],
            "output_format": "landscape",
        },
        "route_summary": {
            "keywords": ["route", "summary", "map", "journey", "overview"],
            "target_duration": 90,
            "transitions": "none",
            "overlays": ["timestamp", "speed"],
            "output_format": "landscape",
        },
    }

    DEFAULT_STYLE = "cinematic"

    # Keywords that override output_format regardless of style preset.
    # Note: "reel" is deliberately excluded here (unlike in the style
    # keywords below) since "highlight reel" is a common cinematic phrase
    # too, not just a social-media one - it would incorrectly force
    # output_format="vertical" on a landscape cinematic request.
    _FORMAT_KEYWORDS: Dict[str, List[str]] = {
        "vertical": ["vertical", "tiktok", "shorts", "story"],
        "square": ["square"],
        "landscape": ["landscape", "widescreen", "horizontal"],
    }

    # Keywords that add an overlay regardless of style preset.
    _OVERLAY_KEYWORDS: Dict[str, List[str]] = {
        "timestamp": ["timestamp", "clock"],
        "speed": ["speed", "telemetry", "gps"],
        "text": ["caption", "text overlay", "title"],
    }

    # Matches things like "30 second", "45 sec", "1 minute", "2 mins".
    _DURATION_RE = re.compile(
        r"(\d+)\s*(seconds?|secs?|minutes?|mins?)\b", re.IGNORECASE
    )

    def plan(self, prompt: str, analyzed_moments: List[Moment]) -> EditPlan:
        """
        Build an edit plan from a text prompt and a list of scored moments.

        Args:
            prompt: Free-text editing request, e.g. "30 second cinematic
                highlight reel" or "vertical social media reel".
            analyzed_moments: (start, end, score) tuples from
                analysis.VideoAnalyzer.analyze().

        Returns:
            {
                "clips": [(start, end), ...],   # chronologically ordered
                "overlays": [...],
                "transitions": "fade" | "none",
                "output_format": "landscape" | "vertical" | "square",
                "style": "cinematic" | "social" | "technical" | "route_summary",
                "target_duration": float,
            }
        """
        prompt_lower = prompt.lower()

        style = self._detect_style(prompt_lower)
        preset = self.STYLE_PRESETS[style]

        target_duration = self._parse_duration(prompt_lower, preset["target_duration"])
        output_format = self._parse_output_format(prompt_lower, preset["output_format"])
        overlays = self._parse_overlays(prompt_lower, preset["overlays"])

        clips = self._select_clips(analyzed_moments, target_duration)

        return {
            "clips": clips,
            "overlays": overlays,
            "transitions": preset["transitions"],
            "output_format": output_format,
            "style": style,
            "target_duration": target_duration,
        }

    # ------------------------------------------------------------------
    # Prompt parsing
    # ------------------------------------------------------------------

    def _detect_style(self, prompt_lower: str) -> str:
        """Pick the first style preset whose keywords appear in the prompt."""
        for style_name, preset in self.STYLE_PRESETS.items():
            if any(keyword in prompt_lower for keyword in preset["keywords"]):
                return style_name
        return self.DEFAULT_STYLE

    def _parse_duration(self, prompt_lower: str, default: float) -> float:
        """Extract an explicit "<N> second(s)/minute(s)" duration, if present."""
        match = self._DURATION_RE.search(prompt_lower)
        if not match:
            return float(default)

        value = float(match.group(1))
        unit = match.group(2)
        return value * 60 if unit.startswith("min") else value

    def _parse_output_format(self, prompt_lower: str, default: str) -> str:
        """Let explicit format keywords override the style preset's default."""
        for fmt, keywords in self._FORMAT_KEYWORDS.items():
            if any(keyword in prompt_lower for keyword in keywords):
                return fmt
        return default

    def _parse_overlays(self, prompt_lower: str, preset_overlays: List[str]) -> List[str]:
        """Start from the style preset's overlays, then apply explicit requests."""
        if any(phrase in prompt_lower for phrase in ("no overlay", "no text", "clean")):
            return []

        overlays = set(preset_overlays)
        for overlay_name, keywords in self._OVERLAY_KEYWORDS.items():
            if any(keyword in prompt_lower for keyword in keywords):
                overlays.add(overlay_name)
        return sorted(overlays)

    # ------------------------------------------------------------------
    # Clip selection
    # ------------------------------------------------------------------

    def _select_clips(
        self, analyzed_moments: List[Moment], target_duration: float
    ) -> List[Clip]:
        """
        Greedily pack the highest-scoring moments into target_duration
        (best-fit by score, not by size), then return them in chronological
        order so playback follows the ride's actual timeline.

        Always includes at least one clip, even if the single
        highest-scoring moment alone exceeds target_duration.
        """
        if not analyzed_moments:
            return []

        ranked = sorted(analyzed_moments, key=lambda moment: moment[2], reverse=True)

        selected: List[Clip] = []
        total_duration = 0.0

        for start, end, _score in ranked:
            clip_duration = end - start

            if selected and total_duration + clip_duration > target_duration:
                # Would overflow the target runtime - skip and keep looking
                # for a smaller clip that still fits.
                continue

            selected.append((start, end))
            total_duration += clip_duration

            if total_duration >= target_duration:
                break

        selected.sort(key=lambda clip: clip[0])
        return selected
