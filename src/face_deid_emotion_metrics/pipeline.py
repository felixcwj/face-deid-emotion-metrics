from __future__ import annotations

import logging
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from .config import PipelineConfig, final_score_percent
from .models_emotion import EmotionSimilarityEngine
from .models_face import FaceObservation, FaceSimilarityEngine
from .profiling import ProfileResult, StageProfiler

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv"}


class _FileProgress:
    def __init__(self, callback: ProgressCallback, relative_path: str) -> None:
        self.callback = callback
        self.relative_path = relative_path
        self.update_raw("file_progress_start", percent=0.0, message="queued")

    def update(self, percent: float, message: str) -> None:
        percent = max(0.0, min(100.0, percent))
        self.update_raw("file_progress", percent=percent, message=message)

    def range_update(self, start: float, end: float, fraction: float, message: str) -> None:
        span = max(0.0, min(1.0, fraction))
        percent = start + (end - start) * span
        self.update(percent, message)

    def finish(self) -> None:
        self.update_raw("file_progress_end", percent=100.0, message="done")

    def update_raw(self, event: str, percent: float, message: str) -> None:
        if self.callback:
            self.callback(event, {"relative_path": self.relative_path, "percent": percent, "message": message})

ProgressCallback = Optional[Callable[[str, object], None]]


@dataclass
class PersonAccumulator:
    facenet_values: List[float] = field(default_factory=list)
    style_values: List[float] = field(default_factory=list)
    emotion_original: List[np.ndarray] = field(default_factory=list)
    emotion_output: List[np.ndarray] = field(default_factory=list)

    def add(self, observation: FaceObservation, emotion_original: np.ndarray, emotion_output: np.ndarray) -> None:
        self.facenet_values.append(observation.facenet_percent)
        self.style_values.append(observation.style_percent)
        self.emotion_original.append(emotion_original)
        self.emotion_output.append(emotion_output)


class MetricsPipeline:
    def __init__(self, config: PipelineConfig, face_engine: FaceSimilarityEngine | None = None, emotion_engine: EmotionSimilarityEngine | None = None) -> None:
        self.config = config
        self.face_engine = face_engine or FaceSimilarityEngine(
            device=config.device,
            lpips_distance_max=config.lpips_distance_max,
            video_backend=config.video_backend,
            resize_to=config.resize_to,
            mtcnn_thresholds=config.mtcnn_thresholds,
            mtcnn_min_face_size=config.mtcnn_min_face_size,
            rotation_angles=config.detector_rotation_angles,
        )
        self.emotion_engine = emotion_engine or EmotionSimilarityEngine(device=config.device, model_name=config.emoti_model_name)
        self._pair_cache: List[Tuple[str, Path, Path]] | None = None

    def run(self, progress_callback: ProgressCallback = None) -> pd.DataFrame:
        records: List[Dict[str, float | str]] = []
        matched_pairs = self._matched_pairs()
        if self.config.max_files is not None:
            matched_pairs = matched_pairs[: self.config.max_files]
        total_files = len(matched_pairs)
        if progress_callback:
            progress_callback("start", total_files)
        for index, (relative_path, input_file, output_file) in enumerate(matched_pairs, start=1):
            file_progress = _FileProgress(progress_callback, relative_path)
            file_progress.update(5.0, "loading files")
            observations = self._analyze_file(input_file, output_file, file_progress)
            file_progress.update(75.0, "emotion inference")
            metrics, person_count = self._aggregate_observations(observations)
            file_progress.update(90.0, "aggregating metrics")
            duration_label = self._video_duration_label(input_file) if input_file.suffix.lower() in VIDEO_EXTENSIONS else ""
            record = self._build_record(relative_path, input_file, metrics, person_count, duration_label)
            records.append(record)
            if total_files:
                percent = (index / total_files) * 100.0
            else:
                percent = 100.0
            logging.info("Processed %s (%d/%d, %.2f%%)", relative_path, index, total_files, percent)
            file_progress.finish()
            if progress_callback:
                progress_callback("update", 1)
        columns = [
            "filename",
            "facenet_percent",
            "lpips_percent",
            "final_score_percent",
            "emoti_emotion_percent",
            "person_count",
            "duration_label",
        ]
        if not records:
            return pd.DataFrame(columns=columns)
        return pd.DataFrame(records, columns=columns)

    def run_profile(self, sample_count: int = 10, seed: int = 42, kind: str = "all") -> ProfileResult:
        matched_pairs = self._matched_pairs()
        matched_pairs = self._filter_pairs_by_kind(matched_pairs, kind)
        if self.config.max_files is not None:
            matched_pairs = matched_pairs[: self.config.max_files]
        if not matched_pairs:
            raise RuntimeError("No matched input/output pairs were found under the base directory")
        if sample_count <= 0:
            raise ValueError("sample_count must be positive")
        if len(matched_pairs) > sample_count:
            rng = random.Random(seed)
            selected = rng.sample(matched_pairs, sample_count)
        else:
            selected = matched_pairs
        profiler = StageProfiler()
        total_elapsed = 0.0
        for relative_path, input_file, output_file in selected:
            start = time.perf_counter()
            observations = self._analyze_file(input_file, output_file, None, profiler)
            self._aggregate_observations(observations, profiler=profiler)
            total_elapsed += time.perf_counter() - start
            logging.info("Profiled %s", relative_path)
        totals = profiler.totals()
        ordered = ("load", "resize", "facenet", "lpips", "emoti")
        allocated = sum(totals.get(stage, 0.0) for stage in ordered)
        totals["other"] = max(0.0, total_elapsed - allocated)
        totals["total"] = total_elapsed
        return ProfileResult(processed=len(selected), requested=sample_count, totals=totals, elapsed=total_elapsed)

    def matched_pairs(self) -> List[Tuple[str, Path, Path]]:
        return list(self._matched_pairs())

    def compute_records_for_paths(self, relative_paths: List[str], progress_callback: ProgressCallback = None) -> Dict[str, Dict[str, object]]:
        if not relative_paths:
            return {}
        lookup = {relative: (input_path, output_path) for relative, input_path, output_path in self._matched_pairs()}
        unique: List[str] = []
        seen: set[str] = set()
        for relative in relative_paths:
            if relative in seen:
                continue
            if relative not in lookup:
                logging.warning("Skipping missing pair for %s", relative)
                continue
            unique.append(relative)
            seen.add(relative)
        records: Dict[str, Dict[str, object]] = {}
        total = len(unique)
        if total == 0:
            return records
        if progress_callback:
            progress_callback("start", total)
        for relative in unique:
            input_file, output_file = lookup[relative]
            record = self._process_single_pair(relative, input_file, output_file)
            records[relative] = record
            if progress_callback:
                progress_callback("update", 1)
        return records

    def run_random_sample(self, seed: int = 42, image_count: int = 5, video_count: int = 5) -> List[Dict[str, object]]:
        matched_pairs = self._matched_pairs()
        images = [pair for pair in matched_pairs if pair[1].suffix.lower() in IMAGE_EXTENSIONS]
        videos = [pair for pair in matched_pairs if pair[1].suffix.lower() in VIDEO_EXTENSIONS]
        rng = random.Random(seed)

        def pick(source: List[Tuple[str, Path, Path]], count: int) -> List[Tuple[str, Path, Path]]:
            if not source or count <= 0:
                return []
            if len(source) <= count:
                return list(source)
            return rng.sample(source, count)

        selected = pick(images, image_count) + pick(videos, video_count)
        seen: set[str] = set()
        unique_pairs: List[Tuple[str, Path, Path]] = []
        for pair in selected:
            if pair[0] in seen:
                continue
            unique_pairs.append(pair)
            seen.add(pair[0])
        records: List[Dict[str, object]] = []
        for relative_path, input_file, output_file in unique_pairs:
            observations = self._analyze_file(input_file, output_file, None)
            metrics, person_count = self._aggregate_observations(observations)
            duration_label = self._video_duration_label(input_file) if input_file.suffix.lower() in VIDEO_EXTENSIONS else ""
            record = self._build_record(relative_path, input_file, metrics, person_count, duration_label)
            records.append(record)
        return records


    def _matched_pairs(self) -> List[Tuple[str, Path, Path]]:
        if self._pair_cache is not None:
            return list(self._pair_cache)
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
        self._pair_cache = pairs
        return list(pairs)

    def _process_single_pair(self, relative_path: str, input_file: Path, output_file: Path) -> Dict[str, object]:
        record: Dict[str, object] = {
            "filename": relative_path,
            "facenet_percent": None,
            "lpips_percent": None,
            "final_score_percent": None,
            "emoti_emotion_percent": None,
            "person_count": 0,
            "duration_label": self._video_duration_label(input_file) if input_file.suffix.lower() in VIDEO_EXTENSIONS else "",
        }
        def log_callback(event: str, payload) -> None:
            if event == "file_progress":
                message = payload.get("message", "")
                percent = payload.get("percent", 0.0)
                logging.info("Processing %s: %s (%.1f%%)", relative_path, message, percent)
        file_progress = _FileProgress(log_callback, relative_path)
        try:
            file_progress.update(5.0, "loading files")
            observations = self._analyze_file(input_file, output_file, file_progress)
            file_progress.update(75.0, "emotion inference")
            metrics, person_count = self._aggregate_observations(observations)
            file_progress.update(90.0, "aggregating metrics")
            record["facenet_percent"] = metrics.get("facenet_percent")
            record["lpips_percent"] = metrics.get("lpips_percent")
            record["final_score_percent"] = metrics.get("final_score_percent")
            record["emoti_emotion_percent"] = metrics.get("emoti_emotion_percent")
            record["person_count"] = person_count
        except Exception:
            logging.exception("Processing failed for %s", relative_path)
            raise
        finally:
            file_progress.finish()
        return record

    def _analyze_file(self, input_file: Path, output_file: Path, file_progress: _FileProgress | None, profiler: StageProfiler | None = None) -> List[FaceObservation]:
        suffix = input_file.suffix.lower()
        if suffix == ".mp4":
            return self.face_engine.analyze_video_pair(
                input_file,
                output_file,
                self.config.max_frames_per_video,
                progress_fn=lambda frac, msg: file_progress.range_update(5.0, 75.0, frac, msg) if file_progress else None,
                profiler=profiler,
            )
        return self.face_engine.analyze_image_pair(
            input_file,
            output_file,
            progress_fn=lambda frac, msg: file_progress.range_update(5.0, 60.0, frac, msg) if file_progress else None,
            profiler=profiler,
        )

    def _aggregate_observations(self, observations: List[FaceObservation], profiler: StageProfiler | None = None) -> Tuple[Dict[str, float], int]:
        if not observations:
            return self._blank_metrics(), 0
        persons: Dict[str, PersonAccumulator] = defaultdict(PersonAccumulator)
        original_faces = [obs.original_face for obs in observations]
        deidentified_faces = [obs.deidentified_face for obs in observations]
        combined_faces = original_faces + deidentified_faces
        emotion_start = time.perf_counter()
        combined_vectors = self.emotion_engine.emotion_vectors(combined_faces)
        if profiler:
            profiler.add("emoti", time.perf_counter() - emotion_start)
        split_index = len(original_faces)
        original_vectors = combined_vectors[:split_index]
        output_vectors = combined_vectors[split_index:]
        for observation, original_vector, output_vector in zip(observations, original_vectors, output_vectors):
            persons[observation.person_id].add(observation, original_vector, output_vector)
        facenet_scores: List[float] = []
        style_scores: List[float] = []
        final_scores: List[float] = []
        emotion_scores: List[float] = []
        for accumulator in persons.values():
            facenet_mean = self._mean(accumulator.facenet_values)
            style_mean = self._mean(accumulator.style_values)
            if facenet_mean is None or style_mean is None:
                continue
            final_score = final_score_percent(facenet_mean, style_mean)
            emotion_original_vector = self._mean_vector(accumulator.emotion_original)
            emotion_output_vector = self._mean_vector(accumulator.emotion_output)
            emotion_similarity = self.emotion_engine.similarity_percent(emotion_original_vector, emotion_output_vector)
            facenet_scores.append(facenet_mean)
            style_scores.append(style_mean)
            final_scores.append(final_score)
            emotion_scores.append(emotion_similarity)
        metrics = {
            "facenet_percent": self._mean(facenet_scores),
            "lpips_percent": self._mean(style_scores),
            "final_score_percent": self._mean(final_scores),
            "emoti_emotion_percent": self._mean(emotion_scores),
        }
        return metrics, len(persons)

    def _build_record(self, relative_path: str, input_file: Path, metrics: Dict[str, float | None], person_count: int, duration_label: str) -> Dict[str, object]:
        facenet_percent = metrics.get("facenet_percent")
        lpips_percent = metrics.get("lpips_percent")
        final_percent = metrics.get("final_score_percent")
        emoti_percent = metrics.get("emoti_emotion_percent")
        if facenet_percent is not None and lpips_percent is not None and final_percent is None:
            final_percent = final_score_percent(facenet_percent, lpips_percent)
        record = {
            "filename": relative_path,
            "facenet_percent": facenet_percent,
            "lpips_percent": lpips_percent,
            "final_score_percent": final_percent,
            "emoti_emotion_percent": emoti_percent,
            "person_count": person_count,
            "duration_label": duration_label,
        }
        return record

    def _mean(self, values: List[float]) -> float | None:
        if not values:
            return None
        return float(fmean(values))

    def _mean_vector(self, vectors: List[np.ndarray]) -> np.ndarray:
        if not vectors:
            return self.emotion_engine.uniform_vector()
        stacked = np.stack(vectors, axis=0)
        mean_vector = stacked.mean(axis=0)
        total = float(mean_vector.sum())
        if total == 0:
            return self.emotion_engine.uniform_vector()
        return mean_vector / total

    def _blank_metrics(self) -> Dict[str, float | None]:
        return {
            "facenet_percent": None,
            "lpips_percent": None,
            "final_score_percent": None,
            "emoti_emotion_percent": None,
        }

    def _filter_pairs_by_kind(self, pairs: List[Tuple[str, Path, Path]], kind: str) -> List[Tuple[str, Path, Path]]:
        if kind == "image":
            return [pair for pair in pairs if pair[1].suffix.lower() in IMAGE_EXTENSIONS]
        if kind == "video":
            return [pair for pair in pairs if pair[1].suffix.lower() in VIDEO_EXTENSIONS]
        return pairs

    def _video_duration_label(self, path: Path) -> str:
        cap = cv2.VideoCapture(str(path))
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 0
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        finally:
            cap.release()
        if fps <= 0 or frame_count <= 0:
            return ""
        seconds = int(max(0, round(frame_count / fps)))
        return f"{seconds}s"
