from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image
from deepface import DeepFace
from fer.fer import FER

EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


class EmotionSimilarityEngine:
    def __init__(self) -> None:
        self.fer_model = FER(mtcnn=False)
        self.deepface_model = DeepFace.build_model("Emotion")

    def emotion_vectors(self, image: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
        fer_vector = self._fer_vector(image)
        deepface_vector = self._deepface_vector(image)
        return fer_vector, deepface_vector

    def similarity_percent(self, vector_a: np.ndarray, vector_b: np.ndarray) -> float:
        distance = float(np.sum(np.abs(vector_a - vector_b)))
        percent = (1.0 - distance / 2.0) * 100.0
        return float(max(0.0, min(100.0, percent)))

    def _fer_vector(self, image: Image.Image) -> np.ndarray:
        array = np.array(image.convert("RGB"))
        detections = self.fer_model.detect_emotions(array)
        if detections:
            emotions = detections[0].get("emotions", {})
            vector = np.array([float(emotions.get(label, 0.0)) for label in EMOTION_LABELS], dtype=np.float32)
        else:
            vector = np.full(len(EMOTION_LABELS), 1.0 / len(EMOTION_LABELS), dtype=np.float32)
        return self._normalize(vector)

    def _deepface_vector(self, image: Image.Image) -> np.ndarray:
        array = np.array(image.convert("RGB"))
        analysis = DeepFace.analyze(
            img_path=array,
            actions=["emotion"],
            enforce_detection=False,
            detector_backend="skip",
            models={"emotion": self.deepface_model},
            prog_bar=False,
        )
        result = analysis[0] if isinstance(analysis, list) else analysis
        emotions = result.get("emotion", {})
        vector = np.array([float(emotions.get(label, 0.0)) for label in EMOTION_LABELS], dtype=np.float32)
        if vector.sum() == 0:
            vector = np.full(len(EMOTION_LABELS), 1.0 / len(EMOTION_LABELS), dtype=np.float32)
        return self._normalize(vector)

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        total = float(vector.sum())
        if total == 0:
            return np.full(len(vector), 1.0 / len(vector), dtype=np.float32)
        return vector / total
