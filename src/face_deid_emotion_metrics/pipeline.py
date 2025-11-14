from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean
from typing import Dict, List

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .models_emotion import EMOTION_LABELS, EmotionSimilarityEngine
from .models_face import FaceObservation, FaceSimilarityEngine


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

    def run(self) -> pd.DataFrame:
        records: List[Dict[str, float | str]] = []
        input_files = self._enumerate_inputs()
        if self.config.max_files is not None:
            input_files = input_files[: self.config.max_files]
        for input_file in input_files:
            relative_path = input_file.relative_to(self.config.input_dir).as_posix()
            output_file = self.config.output_dir / input_file.relative_to(self.config.input_dir)
            if not output_file.exists():
                logging.warning("Missing output file for %s", relative_path)
                continue
            observations = self._analyze_file(input_file, output_file)
            metrics = self._aggregate_observations(observations)
            record = {
                "filename": relative_path,
                "facenet_percent": metrics["facenet_percent"],
                "lpips_percent": metrics["lpips_percent"],
                "final_percent": metrics["final_percent"],
                "fer_percent": metrics["fer_percent"],
                "deepface_percent": metrics["deepface_percent"],
            }
            records.append(record)
        columns = [
            "filename",
            "facenet_percent",
            "lpips_percent",
            "final_percent",
            "fer_percent",
            "deepface_percent",
        ]
        if not records:
            return pd.DataFrame(columns=columns)
        return pd.DataFrame(records, columns=columns)

    def _enumerate_inputs(self) -> List[Path]:
        paths: List[Path] = []
        for candidate in self.config.input_dir.rglob("*"):
            if candidate.is_file() and candidate.suffix.lower() in self.config.file_extensions:
                paths.append(candidate)
        return sorted(paths)

    def _analyze_file(self, input_file: Path, output_file: Path) -> List[FaceObservation]:
        suffix = input_file.suffix.lower()
        if suffix == ".mp4":
            return self.face_engine.analyze_video_pair(input_file, output_file, self.config.max_frames_per_video)
        return self.face_engine.analyze_image_pair(input_file, output_file)

    def _aggregate_observations(self, observations: List[FaceObservation]) -> Dict[str, float]:
        if not observations:
            return self._blank_metrics()
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
        return {
            "facenet_percent": self._mean(facenet_scores),
            "lpips_percent": self._mean(style_scores),
            "final_percent": self._mean(final_scores),
            "fer_percent": self._mean(fer_scores),
            "deepface_percent": self._mean(deep_scores),
        }

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
