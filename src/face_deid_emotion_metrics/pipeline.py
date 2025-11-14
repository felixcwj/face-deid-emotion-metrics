from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from .config import PipelineConfig
from .models_emotion import EMOTION_LABELS, EmotionSimilarityEngine
from .models_face import FaceObservation, FaceSimilarityEngine

ProgressCallback = Optional[Callable[[str, int], None]]


@dataclass
class PersonAccumulator:
    facenet_values: List[float] = field(default_factory=list)
    style_values: List[float] = field(default_factory=list)
    fer_original: List[np.ndarray] = field(default_factory=list)
    fer_output: List[np.ndarray] = field(default_factory=list)
    deep_original: List[np.ndarray] = field(default_factory=list)
    deep_output: List[np.ndarray] = field(default_factory=list)

    def add(self, observation: FaceObservation, fer_original: np.ndarray, fer_output: np.ndarray, deep_original: np.ndarray, deep_output: np.ndarray) -> None:
        self.facenet_values.append(observation.facenet_percent)
        self.style_values.append(observation.style_percent)
        self.fer_original.append(fer_original)
        self.fer_output.append(fer_output)
        self.deep_original.append(deep_original)
        self.deep_output.append(deep_output)


class MetricsPipeline:
    def __init__(self, config: PipelineConfig, face_engine: FaceSimilarityEngine | None = None, emotion_engine: EmotionSimilarityEngine | None = None) -> None:
        self.config = config
        self.face_engine = face_engine or FaceSimilarityEngine(device=config.device, lpips_distance_max=config.lpips_distance_max)
        self.emotion_engine = emotion_engine or EmotionSimilarityEngine()

    def run(self, progress_callback: ProgressCallback = None) -> pd.DataFrame:
        records: List[Dict[str, float | str]] = []
        matched_pairs = self._matched_pairs()
        if self.config.max_files is not None:
            matched_pairs = matched_pairs[: self.config.max_files]
        if progress_callback:
            progress_callback("start", len(matched_pairs))
        for relative_path, input_file, output_file in matched_pairs:
            observations = self._analyze_file(input_file, output_file)
            metrics, person_count = self._aggregate_observations(observations)
            duration_label = self._video_duration_label(input_file) if input_file.suffix.lower() == ".mp4" else ""
            record = {
                "filename": relative_path,
                "facenet_percent": metrics["facenet_percent"],
                "lpips_percent": metrics["lpips_percent"],
                "final_percent": metrics["final_percent"],
                "fer_percent": metrics["fer_percent"],
                "deepface_percent": metrics["deepface_percent"],
                "person_count": person_count,
                "duration_label": duration_label,
            }
            records.append(record)
            if progress_callback:
                progress_callback("update", 1)
        columns = [
            "filename",
            "facenet_percent",
            "lpips_percent",
            "final_percent",
            "fer_percent",
            "deepface_percent",
            "person_count",
            "duration_label",
        ]
        if not records:
            return pd.DataFrame(columns=columns)
        return pd.DataFrame(records, columns=columns)

    def _matched_pairs(self) -> List[Tuple[str, Path, Path]]:
        files: List[Path] = []
        for candidate in self.config.input_dir.rglob("*"):
            if candidate.is_file() and candidate.suffix.lower() in self.config.file_extensions:
                files.append(candidate)
        files.sort(key=lambda path: path.relative_to(self.config.input_dir).as_posix())
        pairs: List[Tuple[str, Path, Path]] = []
        for input_path in files:
            relative = input_path.relative_to(self.config.input_dir)
            relative_str = relative.as_posix()
            output_path = self.config.output_dir / relative
            if not output_path.exists():
                logging.warning("Missing output file for %s", relative_str)
                continue
            pairs.append((relative_str, input_path, output_path))
        return pairs

    def _analyze_file(self, input_file: Path, output_file: Path) -> List[FaceObservation]:
        suffix = input_file.suffix.lower()
        if suffix == ".mp4":
            return self.face_engine.analyze_video_pair(input_file, output_file, self.config.max_frames_per_video)
        return self.face_engine.analyze_image_pair(input_file, output_file)

    def _aggregate_observations(self, observations: List[FaceObservation]) -> Tuple[Dict[str, float], int]:
        if not observations:
            return self._blank_metrics(), 0
        persons: Dict[str, PersonAccumulator] = defaultdict(PersonAccumulator)
        for observation in observations:
            fer_original, deep_original = self.emotion_engine.emotion_vectors(observation.original_face)
            fer_output, deep_output = self.emotion_engine.emotion_vectors(observation.deidentified_face)
            persons[observation.person_id].add(observation, fer_original, fer_output, deep_original, deep_output)
        facenet_scores: List[float] = []
        style_scores: List[float] = []
        final_scores: List[float] = []
        fer_scores: List[float] = []
        deep_scores: List[float] = []
        for accumulator in persons.values():
            facenet_mean = self._mean(accumulator.facenet_values)
            style_mean = self._mean(accumulator.style_values)
            final_score = facenet_mean if style_mean >= self.config.style_similarity_threshold else 0.3 * facenet_mean + 0.7 * style_mean
            fer_original_vector = self._mean_vector(accumulator.fer_original)
            fer_output_vector = self._mean_vector(accumulator.fer_output)
            deep_original_vector = self._mean_vector(accumulator.deep_original)
            deep_output_vector = self._mean_vector(accumulator.deep_output)
            fer_similarity = self.emotion_engine.similarity_percent(fer_original_vector, fer_output_vector)
            deep_similarity = self.emotion_engine.similarity_percent(deep_original_vector, deep_output_vector)
            facenet_scores.append(facenet_mean)
            style_scores.append(style_mean)
            final_scores.append(final_score)
            fer_scores.append(fer_similarity)
            deep_scores.append(deep_similarity)
        metrics = {
            "facenet_percent": self._mean(facenet_scores),
            "lpips_percent": self._mean(style_scores),
            "final_percent": self._mean(final_scores),
            "fer_percent": self._mean(fer_scores),
            "deepface_percent": self._mean(deep_scores),
        }
        return metrics, len(persons)

    def _mean(self, values: List[float]) -> float:
        if not values:
            return 0.0
        return float(fmean(values))

    def _mean_vector(self, vectors: List[np.ndarray]) -> np.ndarray:
        if not vectors:
            return np.full(len(EMOTION_LABELS), 1.0 / len(EMOTION_LABELS), dtype=np.float32)
        stacked = np.stack(vectors, axis=0)
        mean_vector = stacked.mean(axis=0)
        total = float(mean_vector.sum())
        if total == 0:
            return np.full(len(mean_vector), 1.0 / len(mean_vector), dtype=np.float32)
        return mean_vector / total

    def _blank_metrics(self) -> Dict[str, float]:
        return {
            "facenet_percent": 0.0,
            "lpips_percent": 0.0,
            "final_percent": 0.0,
            "fer_percent": 0.0,
            "deepface_percent": 0.0,
        }

    def _video_duration_label(self, path: Path) -> str:
        cap = cv2.VideoCapture(str(path))
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 0
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        finally:
            cap.release()
        if fps <= 0 or frame_count <= 0:
            return ""
        seconds = int(round(frame_count / fps))
        minutes, secs = divmod(seconds, 60)
        if minutes:
            return f"{minutes}m {secs}s"
        return f"{secs}s"
