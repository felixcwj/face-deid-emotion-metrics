from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import lpips
import numpy as np
import torch
from PIL import Image
from facenet_pytorch import InceptionResnetV1, MTCNN, fixed_image_standardization
from torchvision import transforms


@dataclass
class FaceDescriptor:
    image: Image.Image
    embedding: np.ndarray
    bbox: Tuple[int, int, int, int]


@dataclass
class FaceObservation:
    person_id: str
    original_face: Image.Image
    deidentified_face: Image.Image
    facenet_percent: float
    style_percent: float


@dataclass
class FaceTrack:
    track_id: str
    embedding: np.ndarray


class FaceTrackManager:
    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.tracks: List[FaceTrack] = []
        self.next_id = 0

    def assign(self, descriptor: FaceDescriptor) -> str:
        best_track: FaceTrack | None = None
        best_score = self.threshold
        for track in self.tracks:
            score = float(np.dot(track.embedding, descriptor.embedding))
            if score > best_score:
                best_score = score
                best_track = track
        if best_track is None:
            track_id = f"person_{self.next_id}"
            self.next_id += 1
            track = FaceTrack(track_id, descriptor.embedding.copy())
            self.tracks.append(track)
            return track_id
        best_track.embedding = self._normalize(best_track.embedding + descriptor.embedding)
        return best_track.track_id

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm


class FaceSimilarityEngine:
    def __init__(self, lpips_distance_max: float = 1.0, track_threshold: float = 0.5, device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.detector = MTCNN(keep_all=True, device=self.device)
        self.embedder = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
        self.lpips_model = lpips.LPIPS(net="vgg").to(self.device)
        self.lpips_distance_max = lpips_distance_max
        self.track_threshold = track_threshold
        self.embedding_transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            fixed_image_standardization,
        ])
        self.lpips_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

    def analyze_image_pair(self, original_path: Path, deidentified_path: Path) -> List[FaceObservation]:
        original_image = self._load_image(original_path)
        deidentified_image = self._load_image(deidentified_path)
        descriptors_a = self._detect_faces(original_image)
        descriptors_b = self._detect_faces(deidentified_image)
        pairs = self._match_pairs(descriptors_a, descriptors_b)
        observations: List[FaceObservation] = []
        for index, (i, j, similarity) in enumerate(pairs):
            descriptor_a = descriptors_a[i]
            descriptor_b = descriptors_b[j]
            facenet_percent = self._cosine_percent(similarity)
            style_percent = self._style_percent(descriptor_a.image, descriptor_b.image)
            observations.append(
                FaceObservation(
                    person_id=f"person_{index}",
                    original_face=descriptor_a.image,
                    deidentified_face=descriptor_b.image,
                    facenet_percent=facenet_percent,
                    style_percent=style_percent,
                )
            )
        return observations

    def analyze_video_pair(self, original_path: Path, deidentified_path: Path, max_frames: int) -> List[FaceObservation]:
        frames_a, frames_b = self._paired_video_frames(original_path, deidentified_path, max_frames)
        observations: List[FaceObservation] = []
        track_manager = FaceTrackManager(threshold=self.track_threshold)
        for frame_a, frame_b in zip(frames_a, frames_b):
            descriptors_a = self._detect_faces(frame_a)
            descriptors_b = self._detect_faces(frame_b)
            if not descriptors_a or not descriptors_b:
                continue
            track_ids: Dict[int, str] = {}
            for idx, descriptor in enumerate(descriptors_a):
                track_ids[idx] = track_manager.assign(descriptor)
            pairs = self._match_pairs(descriptors_a, descriptors_b)
            for i, j, similarity in pairs:
                descriptor_a = descriptors_a[i]
                descriptor_b = descriptors_b[j]
                person_id = track_ids.get(i)
                if person_id is None:
                    continue
                facenet_percent = self._cosine_percent(similarity)
                style_percent = self._style_percent(descriptor_a.image, descriptor_b.image)
                observations.append(
                    FaceObservation(
                        person_id=person_id,
                        original_face=descriptor_a.image,
                        deidentified_face=descriptor_b.image,
                        facenet_percent=facenet_percent,
                        style_percent=style_percent,
                    )
                )
        return observations

    def _load_image(self, path: Path) -> Image.Image:
        image = Image.open(path)
        return image.convert("RGB")

    def _detect_faces(self, image: Image.Image) -> List[FaceDescriptor]:
        boxes, _ = self.detector.detect(image)
        descriptors: List[FaceDescriptor] = []
        if boxes is None:
            return descriptors
        width, height = image.size
        for box in boxes:
            x1, y1, x2, y2 = [int(max(0, value)) for value in box]
            x1 = min(x1, width)
            y1 = min(y1, height)
            x2 = min(max(x2, x1 + 1), width)
            y2 = min(max(y2, y1 + 1), height)
            crop = image.crop((x1, y1, x2, y2))
            embedding = self._embed_face(crop)
            descriptors.append(FaceDescriptor(crop, embedding, (x1, y1, x2, y2)))
        return descriptors

    def _embed_face(self, image: Image.Image) -> np.ndarray:
        tensor = self.embedding_transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.embedder(tensor)
        vector = embedding.cpu().numpy()[0]
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    def _match_pairs(self, faces_a: Sequence[FaceDescriptor], faces_b: Sequence[FaceDescriptor]) -> List[Tuple[int, int, float]]:
        if not faces_a or not faces_b:
            return []
        matrix = np.zeros((len(faces_a), len(faces_b)), dtype=np.float32)
        for i, face_a in enumerate(faces_a):
            for j, face_b in enumerate(faces_b):
                matrix[i, j] = float(np.dot(face_a.embedding, face_b.embedding))
        flat: List[Tuple[int, int, float]] = []
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                flat.append((i, j, matrix[i, j]))
        flat.sort(key=lambda item: item[2], reverse=True)
        used_a: set[int] = set()
        used_b: set[int] = set()
        pairs: List[Tuple[int, int, float]] = []
        for i, j, value in flat:
            if i in used_a or j in used_b:
                continue
            used_a.add(i)
            used_b.add(j)
            pairs.append((i, j, value))
        return pairs

    def _cosine_percent(self, value: float) -> float:
        percent = (value + 1.0) * 50.0
        return float(max(0.0, min(100.0, percent)))

    def _style_percent(self, image_a: Image.Image, image_b: Image.Image) -> float:
        tensor_a = self._lpips_tensor(image_a)
        tensor_b = self._lpips_tensor(image_b)
        with torch.no_grad():
            distance = float(self.lpips_model(tensor_a, tensor_b).item())
        clamped = min(max(distance, 0.0), self.lpips_distance_max)
        similarity = (1.0 - clamped / self.lpips_distance_max) * 100.0
        return float(max(0.0, min(100.0, similarity)))

    def _lpips_tensor(self, image: Image.Image) -> torch.Tensor:
        tensor = self.lpips_transform(image).unsqueeze(0).to(self.device)
        return tensor * 2.0 - 1.0

    def _paired_video_frames(self, path_a: Path, path_b: Path, max_frames: int) -> Tuple[List[Image.Image], List[Image.Image]]:
        cap_a = cv2.VideoCapture(str(path_a))
        cap_b = cv2.VideoCapture(str(path_b))
        frames_a: List[Image.Image] = []
        frames_b: List[Image.Image] = []
        try:
            total_a = int(cap_a.get(cv2.CAP_PROP_FRAME_COUNT))
            total_b = int(cap_b.get(cv2.CAP_PROP_FRAME_COUNT))
            total = min(total_a, total_b)
            if total <= 0:
                return frames_a, frames_b
            indices = self._frame_indices(total, max_frames)
            for index in indices:
                frame_a = self._read_frame(cap_a, index)
                frame_b = self._read_frame(cap_b, index)
                if frame_a is None or frame_b is None:
                    continue
                frames_a.append(Image.fromarray(frame_a))
                frames_b.append(Image.fromarray(frame_b))
        finally:
            cap_a.release()
            cap_b.release()
        return frames_a, frames_b

    def _frame_indices(self, total: int, max_frames: int) -> List[int]:
        if total <= max_frames:
            return list(range(total))
        positions = np.linspace(0, total - 1, num=max_frames, dtype=np.int32)
        return sorted(set(int(value) for value in positions))

    def _read_frame(self, capture: cv2.VideoCapture, index: int) -> np.ndarray | None:
        capture.set(cv2.CAP_PROP_POS_FRAMES, float(index))
        success, frame = capture.read()
        if not success or frame is None:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
