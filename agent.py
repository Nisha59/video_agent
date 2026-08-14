"""
agent.py

Defines the second half of the project's two-agent architecture: the
"Planning Agent" (implemented as the `EditAgent` class) responsible for
turning a user's plain-text request, together with structured scene
descriptions from the Perception Agent, into a concrete, executable edit
plan for the ride video.

Two-agent architecture
-----------------------
1. Perception Agent (perception_agent.PerceptionAgent) - looks at the raw
   video. It extracts a representative frame per detected moment and
   captions it with a vision-language model, producing "enriched moments":
   (start, end, score, description) dicts. It makes no editing decisions.

2. Planning Agent (this module, EditAgent) - never touches raw video or
   frames. It receives the enriched moments produced by the Perception
   Agent plus a text prompt, and decides *editing actions*: which style to
   render in, which clips to keep, how to order them, and which overlays
   to apply. It reasons entirely over already-extracted structured data
   (scores + descriptions), not pixels.

This split mirrors a typical perceive -> plan pipeline: perception turns
raw signal into meaning, planning turns meaning into decisions.

Reasoning approach (Planning Agent)
-------------------------------------
`EditAgent.plan()` is *rule-based*, not model-driven: it never calls an LLM
or does anything probabilistic. It follows four deterministic steps:

1. Keyword matching - the prompt is lowercased and scanned for known trigger
   words (see STYLE_PRESETS and the _*_KEYWORDS tables below) to decide the
   editing `style`, `target_duration`, `output_format`, and `overlays`. Each
   style preset supplies sensible defaults for anything the prompt doesn't
   mention explicitly, and explicit keywords in the prompt can override
   those defaults (e.g. a "technical" prompt that also says "vertical"
   still renders vertically). This part is unchanged by the Perception
   Agent integration - style/duration/format/overlay decisions still come
   purely from the prompt.

2. Diversity-aware clip selection - `enriched_moments` (the
   (start, end, score, description) dicts produced by
   PerceptionAgent.enrich_moments()) are greedily packed into the available
   `target_duration`. Unlike pure score-ranking, each candidate's score is
   penalized by how similar its description is to already-selected clips'
   descriptions (via word-overlap), so that, say, five near-identical "a
   car driving on a highway" moments don't crowd out a single "a car
   turning on a mountain road" moment - the reel ends up covering more
   *distinct* content instead of just the loudest/busiest few seconds
   repeated. Selected clips are then re-ordered chronologically so the
   final edit still follows the ride's natural timeline.

3. Assembly - the selected clips (now carrying their descriptions forward,
   for use as auto-generated captions/subtitles later), overlays,
   transition style, and output format are combined into a single
   `edit_plan` dict that toolbox.VideoToolbox can render directly.

Prompt keyword -> editing decision mapping
-------------------------------------------
    "cinematic" / "highlight" / "epic" / "movie"
        -> style="cinematic": longer landscape cut, fade transitions, no
           telemetry overlays (clean, story-driven look) - just a brief
           scene_caption at the start of each clip.
    "social" / "reel" / "tiktok" / "instagram" / "shorts" / "vertical"
        -> style="social": short vertical cut with fades, a text overlay,
           and per-clip scene_caption, tuned for social feeds.
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
    "scene caption" / "scene description" / "describe scene"
        -> adds a "scene_caption" overlay (per-clip Perception Agent
           descriptions, shown briefly at the start of each clip).
    "no overlay" / "no text" / "clean" -> clears all overlays.

If nothing matches, the agent falls back to the "cinematic" style preset.
"""

import re
from typing import Any, Dict, List

# An enriched moment as produced by
# PerceptionAgent.enrich_moments(): {"start", "end", "score", "description"}.
EnrichedMoment = Dict[str, Any]

# A selected clip in the final edit plan - the enriched moment carried
# forward unchanged (start/end/score/description), now "chosen" rather
# than just "scored".
Clip = Dict[str, Any]

EditPlan = Dict[str, Any]


class EditAgent:
    """
    The Planning Agent: turns a natural-language editing request plus the
    Perception Agent's enriched moments into a structured edit plan.

    It never analyzes raw video or frames itself - all content
    understanding (what's actually happening on screen) has already been
    done upstream by PerceptionAgent. This class only reasons over the
    scores/descriptions it's handed, and over the prompt text.

    Usage:
        perception = PerceptionAgent()
        enriched_moments = perception.enrich_moments(video_path, raw_moments)

        planner = EditAgent()
        edit_plan = planner.plan("30 second cinematic highlight reel", enriched_moments)
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
            # No telemetry overlays (keeps the story-driven look clean), but
            # a brief Perception Agent scene caption at the start of each
            # clip fits the "cinematic" framing well - it reads like a
            # scene title card, not a data readout.
            "overlays": ["scene_caption"],
            "output_format": "landscape",
        },
        "social": {
            "keywords": ["social", "reel", "tiktok", "instagram", "shorts"],
            "target_duration": 15,
            "transitions": "fade",
            # "text" (the prompt as a caption) plus scene_caption (each
            # clip's own description) - social cuts benefit from both the
            # overall hook and per-clip context, since viewers skim fast.
            "overlays": ["text", "scene_caption"],
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
    #
    # Note: "scene_caption" and "text" both key off phrases containing
    # "caption" on purpose (they're complementary, not exclusive) - "text"
    # burns one caption for a clip's whole duration, "scene_caption" briefly
    # shows the Perception Agent's per-clip description at the start of
    # each clip. A prompt mentioning "caption" can reasonably want both.
    _OVERLAY_KEYWORDS: Dict[str, List[str]] = {
        "timestamp": ["timestamp", "clock"],
        "speed": ["speed", "telemetry", "gps"],
        "text": ["caption", "text overlay", "title"],
        "scene_caption": ["scene caption", "scene description", "describe scene"],
    }

    # Matches things like "30 second", "45 sec", "1 minute", "2 mins".
    _DURATION_RE = re.compile(
        r"(\d+)\s*(seconds?|secs?|minutes?|mins?)\b", re.IGNORECASE
    )

    # How strongly a candidate clip's score gets penalized for being
    # textually similar to an already-selected clip's description (0 =
    # ignore descriptions entirely and rank by score alone, 1 = diversity
    # dominates score). 0.3 was picked empirically to break near-duplicate
    # ties without letting a single odd/short description caption veto an
    # otherwise clearly-best moment.
    _DIVERSITY_PENALTY_WEIGHT = 0.3

    def plan(self, prompt: str, enriched_moments: List[EnrichedMoment]) -> EditPlan:
        """
        Build an edit plan from a text prompt and the Perception Agent's
        enriched moments.

        Args:
            prompt: Free-text editing request, e.g. "30 second cinematic
                highlight reel" or "vertical social media reel".
            enriched_moments: {"start", "end", "score", "description"}
                dicts from PerceptionAgent.enrich_moments().

        Returns:
            {
                "clips": [{"start", "end", "score", "description"}, ...],
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

        clips = self._select_clips(enriched_moments, target_duration)

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
        self, enriched_moments: List[EnrichedMoment], target_duration: float
    ) -> List[Clip]:
        """
        Greedily pack moments into target_duration, but rank candidates by
        score *adjusted for description similarity* to what's already been
        selected, rather than by raw score alone (Maximal-Marginal-Relevance
        style: prefer high score, penalize redundancy). This spreads the
        final reel across more distinct content instead of picking several
        near-identical high-scoring moments back to back.

        Always includes at least one clip, even if the single
        highest-scoring moment alone exceeds target_duration.
        """
        if not enriched_moments:
            return []

        remaining = list(enriched_moments)
        token_sets = {
            id(moment): self._tokenize(moment.get("description", ""))
            for moment in enriched_moments
        }

        selected: List[EnrichedMoment] = []
        total_duration = 0.0

        while remaining:
            best_moment = None
            best_adjusted_score = None

            for moment in remaining:
                clip_duration = moment["end"] - moment["start"]
                if selected and total_duration + clip_duration > target_duration:
                    # Would overflow the target runtime - not a candidate
                    # this round (a smaller clip might still fit later).
                    continue

                similarity = max(
                    (
                        self._description_similarity(
                            token_sets[id(moment)], token_sets[id(other)]
                        )
                        for other in selected
                    ),
                    default=0.0,
                )
                adjusted_score = moment["score"] - self._DIVERSITY_PENALTY_WEIGHT * similarity

                if best_adjusted_score is None or adjusted_score > best_adjusted_score:
                    best_moment = moment
                    best_adjusted_score = adjusted_score

            if best_moment is None:
                # Nothing left fits within the remaining runtime budget.
                break

            selected.append(best_moment)
            total_duration += best_moment["end"] - best_moment["start"]
            remaining.remove(best_moment)

            if total_duration >= target_duration:
                break

        selected.sort(key=lambda moment: moment["start"])
        return [dict(moment) for moment in selected]

    @staticmethod
    def _tokenize(description: str) -> set:
        """Lowercase word set for a description, used for similarity comparison."""
        return set(re.findall(r"\w+", description.lower()))

    @staticmethod
    def _description_similarity(tokens_a: set, tokens_b: set) -> float:
        """
        Jaccard similarity (intersection over union) between two
        descriptions' word sets - a simple, dependency-free stand-in for
        semantic similarity. 0.0 if either description is empty (no basis
        for comparison, so no diversity penalty is applied) or 1.0 for
        identical word sets.
        """
        if not tokens_a or not tokens_b:
            return 0.0
        union = tokens_a | tokens_b
        if not union:
            return 0.0
        return len(tokens_a & tokens_b) / len(union)
