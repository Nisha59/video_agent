"""
analysis.py

Handles analysis of ride video and telemetry data.

Currently implements:
- VideoAnalyzer: detects "interesting moments" in a video by combining
  scene-change detection, motion intensity, and audio energy signals.

Still to come:
- Parsing and analyzing GPX telemetry data (via gpxpy) for speed, elevation,
  and route segments
- Correlating video timestamps with GPX telemetry data
- Producing structured analysis results for use by agent.py
"""

import os
import sys
import glob
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
from scenedetect import detect, ContentDetector

# A "moment" is a scored time range in the source video: (start_time, end_time, score).
# start_time/end_time are in seconds; score is a normalized interest value (roughly 0-1,
# higher = more interesting).
Moment = Tuple[float, float, float]


class VideoAnalyzer:
    """
    Analyzes a video file and scores segments of it by how "interesting" they are,
    using three independent signals: scene changes, motion, and audio energy.

    Usage:
        analyzer = VideoAnalyzer("sample_data/ride.mp4")
        moments = analyzer.analyze()
    """

    # Default weight given to each signal when combining into a single score.
    # These must sum to 1.0 for the combined score to stay roughly in [0, 1].
    DEFAULT_WEIGHTS: Dict[str, float] = {
        "motion": 0.4,
        "audio": 0.4,
        "scene": 0.2,
    }

    # Scenes shorter than this get merged into a neighboring scene rather than
    # standing alone as a clip candidate. Below ~1s, PySceneDetect cuts are
    # often false positives (a flash, a quick pan) rather than a real
    # standalone moment, and toolbox.VideoToolbox.concatenate_clips uses a
    # fixed 1-second cross-fade between clips - a clip shorter than that
    # can't be cleanly faded in and out. 1.5s leaves a bit of headroom above
    # that fade duration.
    MIN_SCENE_DURATION = 1.5

    def __init__(
        self,
        video_path: str,
        weights: Optional[Dict[str, float]] = None,
        motion_sample_every_n: int = 5,
        min_scene_duration: Optional[float] = None,
    ):
        """
        Args:
            video_path: Path to the input video file.
            weights: Optional override for how much each signal (motion, audio,
                scene) contributes to the final combined score.
            motion_sample_every_n: Only compute frame-differencing on every Nth
                frame within a segment, to keep motion analysis fast on long videos.
            min_scene_duration: Optional override for MIN_SCENE_DURATION.
        """
        self.video_path = video_path
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.motion_sample_every_n = max(1, motion_sample_every_n)
        self.min_scene_duration = (
            self.MIN_SCENE_DURATION if min_scene_duration is None else min_scene_duration
        )

    def analyze(self) -> List[Moment]:
        """
        Run the full analysis pipeline and return interesting moments in
        chronological order. Each moment's score is a weighted combination
        of the motion, audio, and scene signals described below.
        """
        # Step 1: segment the video using scene-change detection. Every other
        # signal is then measured *within* each of these segments, so scene
        # detection effectively decides where moment boundaries fall.
        scenes = self._detect_scenes()

        # Step 2: measure how much visual motion happens in each segment.
        motion_raw = self._compute_motion_scores(scenes)

        # Step 3: measure how loud/energetic the audio is in each segment.
        audio_raw = self._compute_audio_scores(scenes)

        # Step 4: shorter scenes tend to feel punchier/more eventful than long,
        # static ones, so we use inverse duration as the "scene" signal.
        duration_raw = [1.0 / max(end - start, 0.01) for start, end in scenes]

        # Normalize each raw signal onto a common [0, 1] scale so that no single
        # signal (e.g. audio RMS, which can be in the hundreds/thousands) dominates
        # the combined score just because of its raw units.
        motion_norm = self._normalize(motion_raw)
        audio_norm = self._normalize(audio_raw)
        scene_norm = self._normalize(duration_raw)

        # Combine the three normalized signals into one score per segment using
        # a simple weighted average. Weights are configurable via self.weights.
        moments: List[Moment] = []
        for i, (start, end) in enumerate(scenes):
            score = (
                self.weights["motion"] * motion_norm[i]
                + self.weights["audio"] * audio_norm[i]
                + self.weights["scene"] * scene_norm[i]
            )
            moments.append((start, end, score))

        return moments

    def _detect_scenes(self) -> List[Tuple[float, float]]:
        """
        Signal 1: Scene change detection (PySceneDetect).

        Uses PySceneDetect's ContentDetector, which flags a cut wherever the
        difference between consecutive frames' color/intensity histograms
        exceeds a threshold. This gives us natural segment boundaries (e.g.
        camera angle changes, hard cuts) instead of arbitrary fixed windows.

        Falls back to treating the entire video as a single scene if no cuts
        are detected (e.g. for a single continuous take).
        """
        scene_list = detect(self.video_path, ContentDetector())

        if not scene_list:
            cap = cv2.VideoCapture(self.video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            duration = (frame_count / fps) if fps else 0.0
            cap.release()
            return [(0.0, duration)]

        scenes = [(start.get_seconds(), end.get_seconds()) for start, end in scene_list]
        return self._merge_short_scenes(scenes, self.min_scene_duration)

    @staticmethod
    def _merge_short_scenes(
        scenes: List[Tuple[float, float]], min_duration: float
    ) -> List[Tuple[float, float]]:
        """
        Merge consecutive scenes forward until each merged scene reaches
        min_duration, so an overly-short scene-cut detection doesn't become a
        standalone clip candidate on its own (see MIN_SCENE_DURATION).
        """
        if not scenes:
            return scenes

        merged: List[Tuple[float, float]] = []
        acc_start, acc_end = scenes[0]
        for start, end in scenes[1:]:
            if (acc_end - acc_start) < min_duration:
                acc_end = end
            else:
                merged.append((acc_start, acc_end))
                acc_start, acc_end = start, end
        merged.append((acc_start, acc_end))

        # The last accumulated scene may still be short if the video ran out
        # before it reached min_duration - fold it into the previous one
        # instead of leaving a too-short clip candidate.
        if len(merged) > 1 and (merged[-1][1] - merged[-1][0]) < min_duration:
            prev_start, _ = merged[-2]
            _, last_end = merged[-1]
            merged[-2] = (prev_start, last_end)
            merged.pop()

        return merged

    def _compute_motion_scores(self, scenes: List[Tuple[float, float]]) -> List[float]:
        """
        Signal 2: Motion intensity (OpenCV frame differencing).

        For each scene, we walk through its frames and compute the absolute
        pixel-value difference between consecutive *sampled* grayscale frames
        (cv2.absdiff). The mean of these differences approximates how much
        movement/action is happening on screen - a mostly-static shot scores
        low, fast action scores high.

        We only convert/diff every `motion_sample_every_n` frames (still
        decoding every frame, since OpenCV must read frames sequentially) to
        keep this affordable on long videos.
        """
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        scores = []

        for start, end in scenes:
            start_frame = int(start * fps)
            end_frame = int(end * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            prev_gray = None
            diffs = []
            frame_idx = start_frame

            while frame_idx < end_frame:
                ret, frame = cap.read()
                if not ret:
                    break

                if (frame_idx - start_frame) % self.motion_sample_every_n == 0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    if prev_gray is not None:
                        diff = cv2.absdiff(gray, prev_gray)
                        diffs.append(float(np.mean(diff)))
                    prev_gray = gray

                frame_idx += 1

            scores.append(float(np.mean(diffs)) if diffs else 0.0)

        cap.release()
        return scores

    def _compute_audio_scores(self, scenes: List[Tuple[float, float]]) -> List[float]:
        """
        Signal 3: Audio energy peaks (pydub RMS volume).

        Loads the audio track and computes RMS (root-mean-square) volume for
        each 1-second chunk. RMS approximates perceived loudness, so spikes
        here tend to correspond to exciting moments (shouting, engine revs,
        music swells, etc.). For each scene, we average the per-second RMS
        values that fall within that scene's time range.

        Videos with no audio track (e.g. silent action-cam footage) raise an
        IndexError from pydub when it looks up the audio stream; treat that
        the same as "no audio signal" rather than failing the whole analysis.
        """
        try:
            audio = AudioSegment.from_file(self.video_path)
        except (IndexError, CouldntDecodeError):
            return [0.0] * len(scenes)

        chunk_ms = 1000

        rms_per_second = [
            audio[i : i + chunk_ms].rms for i in range(0, len(audio), chunk_ms)
        ]

        scores = []
        for start, end in scenes:
            start_idx = int(start)
            end_idx = max(int(end), start_idx + 1)
            window = rms_per_second[start_idx:end_idx]
            scores.append(float(np.mean(window)) if window else 0.0)

        return scores

    @staticmethod
    def _normalize(values: List[float]) -> List[float]:
        """
        Min-max normalize a list of raw values onto a [0, 1] scale so that
        signals with very different units (pixel-diff means vs. RMS volume vs.
        inverse duration) can be combined fairly with the weighted average
        in analyze(). If every value is identical (no variation), returns a
        neutral 0.5 for each instead of dividing by zero.
        """
        if not values:
            return []

        lo, hi = min(values), max(values)
        if hi == lo:
            return [0.5] * len(values)

        return [(v - lo) / (hi - lo) for v in values]


if __name__ == "__main__":
    # Simple manual test: analyze a sample video and print the top-scoring moments.
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
        print("Usage: python analysis.py <path_to_video>")
        print("(or place a .mp4/.mov file in sample_data/)")
        sys.exit(1)

    print(f"Analyzing {sample_path} ...")
    analyzer = VideoAnalyzer(sample_path)
    moments = analyzer.analyze()

    top_moments = sorted(moments, key=lambda m: m[2], reverse=True)[:5]
    print("\nTop interesting moments:")
    for start, end, score in top_moments:
        print(f"  {start:7.2f}s - {end:7.2f}s   score={score:.3f}")
