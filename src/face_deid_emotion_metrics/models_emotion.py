from __future__ import annotations

import logging
from typing import List, Sequence

import numpy as np
import torch
from PIL import Image

from emotiefflib.facial_analysis import EmotiEffLibRecognizer

logger = logging.getLogger(__name__)


class EmotionSimilarityEngine:
    def __init__(self, device: torch.device, model_name: str) -> None:
        self.device = device
        self.model_name = model_name
        self.recognizer = EmotiEffLibRecognizer(engine="torch", model_name=model_name, device=str(device))
        self.label_order = self._resolve_label_order()

    def emotion_vectors(self, images: Sequence[Image.Image]) -> List[np.ndarray]:
        if not images:
            return []
        bgr_faces = [self._pil_to_bgr(image) for image in images]
        batch = self._batched_probabilities(bgr_faces)
        return [row.copy() for row in batch]

    def emotion_vector(self, image: Image.Image) -> np.ndarray:
        vectors = self.emotion_vectors([image])
        return vectors[0] if vectors else self.uniform_vector()

    def similarity_percent(self, vector_a: np.ndarray, vector_b: np.ndarray) -> float:
        distance = float(np.sum(np.abs(vector_a - vector_b)))
        percent = (1.0 - distance / 2.0) * 100.0
        return float(max(0.0, min(100.0, percent)))

    def uniform_vector(self) -> np.ndarray:
        length = len(self.label_order)
        return np.full(length, 1.0 / length, dtype=np.float32)

    def _batched_probabilities(self, faces: Sequence[np.ndarray]) -> np.ndarray:
        if not faces:
            return np.empty((0, len(self.label_order)), dtype=np.float32)
        uniform = self.uniform_vector()
        try:
            _, scores = self.recognizer.predict_emotions(list(faces), logits=False)
        except Exception as error:  # pragma: no cover - defensive fallback
            logger.exception("EmotiEffLib inference failed: %s", error)
            return np.stack([uniform for _ in faces], axis=0)
        probabilities = self._ensure_2d(scores).astype(np.float32, copy=False)
        row_sums = probabilities.sum(axis=1, keepdims=True)
        zero_mask = row_sums <= 0
        if np.any(zero_mask):
            probabilities[zero_mask] = uniform
            row_sums = probabilities.sum(axis=1, keepdims=True)
        probabilities /= row_sums
        return probabilities

    def _pil_to_bgr(self, image: Image.Image) -> np.ndarray:
        array = np.array(image.convert("RGB"), dtype=np.uint8)
        return array[..., ::-1]

    def _ensure_2d(self, scores: np.ndarray) -> np.ndarray:
        if scores.ndim == 1:
            return scores[np.newaxis, :]
        return scores

    def _resolve_label_order(self) -> List[str]:
        ordered = [self.recognizer.idx_to_emotion_class[idx].lower() for idx in sorted(self.recognizer.idx_to_emotion_class)]
        return ordered
