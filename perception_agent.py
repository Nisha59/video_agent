"""
perception_agent.py

Implements the "Perception Agent" - the part of the system that actually
*looks at* the ride footage and describes what it sees, using a real
vision-language model instead of raw pixel/audio signal processing.

Scope, on purpose
------------------
This module's ONLY job is content understanding: given a video and a list
of already-detected interesting moments (from analysis.VideoAnalyzer), it
extracts one representative frame per moment and captions it with a
Hugging Face image captioning model. It does NOT decide which moments are
worth keeping, how long the final edit should be, which overlays to apply,
or how to render anything - that planning/editing logic belongs to
EditAgent (agent.py) and VideoToolbox (toolbox.py). PerceptionAgent hands
those downstream components richer, human-readable context (e.g. "a car
turning on a mountain road") to reason with, but makes no editing
decisions itself.

This is what turns the pipeline from pure signal-processing heuristics
(motion/audio/scene-cut counts) into something that actually understands
video content, in the "multimodal AI system" sense the brief asks for.
"""

import glob
import os
import sys
from typing import Any, Dict, List, Tuple

import cv2
import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

# A moment as produced by analysis.VideoAnalyzer.analyze(): (start, end, score).
Moment = Tuple[float, float, float]

DEFAULT_MODEL_NAME = "Salesforce/blip-image-captioning-base"


class PerceptionAgent:
    """
    Looks at ride footage and describes what's happening in it.

    Usage:
        perception = PerceptionAgent()
        enriched = perception.enrich_moments("sample_data/ride.mp4", moments)
        # -> [{"start": 12.3, "end": 18.1, "score": 0.82,
        #      "description": "a car driving on a highway"}, ...]
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        """
        Load the captioning model and processor once, up front, so that
        enrich_moments() can caption many frames without repeatedly paying
        the (relatively expensive) model-loading cost per frame.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(
            self.device
        )

    def enrich_moments(
        self, video_path: str, moments: List[Moment]
    ) -> List[Dict[str, Any]]:
        """
        For each (start, end, score) moment, extract a representative frame
        (the middle of the segment) and caption it, returning the moments
        enriched with a "description" field.

        Frames that can't be read (e.g. a moment that lands past the last
        decodable frame) are skipped rather than failing the whole batch -
        they're returned with an empty description instead of a caption.
        """
        if not moments:
            return []

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        enriched: List[Dict[str, Any]] = []
        for start, end, score in moments:
            frame = self._extract_middle_frame(cap, start, end, fps, frame_count)
            description = self._caption_frame(frame) if frame is not None else ""

            enriched.append(
                {
                    "start": start,
                    "end": end,
                    "score": score,
                    "description": description,
                }
            )

        cap.release()
        return enriched

    # ------------------------------------------------------------------
    # Frame extraction
    # ------------------------------------------------------------------

    def _extract_middle_frame(
        self,
        cap: cv2.VideoCapture,
        start: float,
        end: float,
        fps: float,
        frame_count: float,
    ):
        """
        Seek to and decode the frame at the midpoint of [start, end], as a
        single representative snapshot of that moment. Returns a BGR numpy
        frame, or None if the frame couldn't be read.
        """
        middle_time = (start + end) / 2.0
        frame_idx = int(middle_time * fps)

        if frame_count:
            frame_idx = min(frame_idx, max(int(frame_count) - 1, 0))
        frame_idx = max(frame_idx, 0)

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        return frame if ret else None

    # ------------------------------------------------------------------
    # Captioning
    # ------------------------------------------------------------------

    def _caption_frame(self, frame) -> str:
        """
        Run one BGR OpenCV frame through the BLIP image captioning model
        and return the generated caption as plain text.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_frame)

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        output_ids = self.model.generate(**inputs)

        return self.processor.decode(output_ids[0], skip_special_tokens=True)


if __name__ == "__main__":
    # Standalone smoke test: analyze a sample video for interesting moments
    # (reusing VideoAnalyzer from analysis.py), then caption each one and
    # print the enriched results.
    from analysis import VideoAnalyzer

    if len(sys.argv) > 1:
        sample_path = sys.argv[1]
    else:
        candidates = sorted(
            glob.glob(os.path.join("sample_data", "*.mp4"))
            + glob.glob(os.path.join("sample_data", "*.mov"))
        )
        sample_path = candidates[0] if candidates else None

    if not sample_path or not os.path.exists(sample_path):
        print("No sample video found.")
        print("Usage: python perception_agent.py <path_to_video>")
        print("(or place a .mp4/.mov file in sample_data/)")
        sys.exit(1)

    print(f"Analyzing {sample_path} ...")
    moments = VideoAnalyzer(sample_path).analyze()

    print("Loading captioning model (this can take a while on first run) ...")
    perception = PerceptionAgent()

    print(f"Captioning {len(moments)} moment(s) ...")
    enriched_moments = perception.enrich_moments(sample_path, moments)

    print("\nEnriched moments:")
    for moment in enriched_moments:
        print(
            f"  {moment['start']:7.2f}s - {moment['end']:7.2f}s  "
            f"score={moment['score']:.3f}  \"{moment['description']}\""
        )
