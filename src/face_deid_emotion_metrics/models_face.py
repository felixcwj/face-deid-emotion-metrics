from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import lpips
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from decord import VideoReader as DecordVideoReader
from decord import cpu as decord_cpu
from decord import gpu as decord_gpu
from decord._ffi.base import DECORDError
from facenet_pytorch import InceptionResnetV1, MTCNN
from torch.utils import dlpack as torch_dlpack
from torchvision import transforms

logger = logging.getLogger(__name__)


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
    def __init__(self, device: torch.device, lpips_distance_max: float = 1.0, track_threshold: float = 0.5, video_batch_size: int = 4) -> None:
        self.device = device
        self.detector = MTCNN(keep_all=True, device=self.device)
        self.embedder = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
        self.lpips_model = lpips.LPIPS(net="vgg").to(self.device)
        self.lpips_distance_max = lpips_distance_max
        self.track_threshold = track_threshold
        self.video_batch_size = max(1, video_batch_size)
        self._decord_gpu_enabled = self.device.type == "cuda"
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
        descriptors_a_sequence = self._detect_faces_in_batches(frames_a)
        descriptors_b_sequence = self._detect_faces_in_batches(frames_b)
        for descriptors_a, descriptors_b in zip(descriptors_a_sequence, descriptors_b_sequence):
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
        boxes, probs = self.detector.detect(image)
        face_tensors = self.detector.extract(image, boxes, save_path=None)
        return self._build_descriptors(image, boxes, probs, face_tensors)

    def _detect_faces_in_batches(self, images: Sequence[Image.Image]) -> List[List[FaceDescriptor]]:
        if not images:
            return []
        descriptors: List[List[FaceDescriptor]] = []
        for start in range(0, len(images), self.video_batch_size):
            chunk = images[start : start + self.video_batch_size]
            descriptors.extend(self._detect_faces_batch(chunk))
        return descriptors

    def _detect_faces_batch(self, images: Sequence[Image.Image]) -> List[List[FaceDescriptor]]:
        if not images:
            return []
        boxes_batch, probs_batch = self.detector.detect(list(images))
        tensors_batch = self.detector.extract(list(images), boxes_batch, save_path=None)
        boxes_list = self._ensure_batch_list(boxes_batch, len(images))
        probs_list = self._ensure_batch_list(probs_batch, len(images))
        tensors_list = self._ensure_batch_list(tensors_batch, len(images))
        descriptors_per_image: List[List[FaceDescriptor]] = []
        for image, boxes, probs, tensors in zip(images, boxes_list, probs_list, tensors_list):
            descriptors_per_image.append(self._build_descriptors(image, boxes, probs, tensors))
        return descriptors_per_image

    def _build_descriptors(
        self,
        image: Image.Image,
        boxes: Sequence[Sequence[float]] | None,
        probs: Sequence[float] | None,
        face_tensors: torch.Tensor | Sequence[torch.Tensor] | None,
    ) -> List[FaceDescriptor]:
        descriptors: List[FaceDescriptor] = []
        if boxes is None or face_tensors is None:
            return descriptors
        boxes_list = list(boxes)
        if not boxes_list:
            return descriptors
        if isinstance(face_tensors, torch.Tensor):
            tensors = face_tensors
        elif isinstance(face_tensors, Sequence) and face_tensors:
            tensors = torch.stack(face_tensors)
        else:
            return descriptors
        probs_list = self._ensure_list_length(probs, len(boxes_list))
        width, height = image.size
        for idx, box in enumerate(boxes_list):
            prob_value = probs_list[idx]
            prob = float(prob_value) if prob_value is not None else 1.0
            if prob <= 0:
                continue
            bbox = self._clip_box(box, width, height)
            if bbox is None:
                continue
            face_tensor = tensors[idx]
            crop = image.crop(bbox)
            embedding = self._embed_face_tensor(face_tensor)
            descriptors.append(FaceDescriptor(crop, embedding, bbox))
        return descriptors

    def _match_pairs(self, faces_a: Sequence[FaceDescriptor], faces_b: Sequence[FaceDescriptor]) -> List[Tuple[int, int, float]]:
        if not faces_a or not faces_b:
            return []
        matrix = np.zeros((len(faces_a), len(faces_b)), dtype=np.float32)
        for i, face_a in enumerate(faces_a):
            for j, face_b in enumerate(faces_b):
                matrix[i, j] = self._cosine_similarity(face_a.embedding, face_b.embedding)
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

    def _cosine_similarity(self, embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
        value = float(np.dot(embedding_a, embedding_b))
        return float(np.clip(value, -1.0, 1.0))

    def _embed_face_tensor(self, face_tensor: torch.Tensor) -> np.ndarray:
        if face_tensor.ndim == 3:
            face_tensor = face_tensor.unsqueeze(0)
        face_tensor = face_tensor.to(self.device)
        with torch.no_grad():
            embedding = self.embedder(face_tensor)
        vector = F.normalize(embedding, dim=1)[0]
        return vector.detach().cpu().numpy().astype(np.float32)

    def _clip_box(self, box: Sequence[float], width: int, height: int) -> Tuple[int, int, int, int] | None:
        if box is None or len(box) < 4:
            return None
        x1 = int(max(0, box[0]))
        y1 = int(max(0, box[1]))
        x2 = int(max(0, box[2]))
        y2 = int(max(0, box[3]))
        x1 = min(x1, width)
        y1 = min(y1, height)
        x2 = min(max(x2, x1 + 1), width)
        y2 = min(max(y2, y1 + 1), height)
        return x1, y1, x2, y2

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
        reader_a = self._video_reader(path_a)
        reader_b = self._video_reader(path_b)
        total = min(len(reader_a), len(reader_b))
        if total <= 0:
            return [], []
        indices = self._frame_indices(total, max_frames)
        frames_a = self._fetch_frames(reader_a, indices)
        frames_b = self._fetch_frames(reader_b, indices)
        paired = min(len(frames_a), len(frames_b))
        if paired <= 0:
            return [], []
        return frames_a[:paired], frames_b[:paired]

    def _frame_indices(self, total: int, max_frames: int) -> List[int]:
        if total <= max_frames:
            return list(range(total))
        positions = np.linspace(0, total - 1, num=max_frames, dtype=np.int32)
        return sorted(set(int(value) for value in positions))

    def _video_reader_context(self, force_cpu: bool = False):
        if not force_cpu and self._decord_gpu_enabled:
            index = self.device.index if self.device.index is not None else 0
            return decord_gpu(index)
        return decord_cpu(0)

    def _video_reader(self, path: Path) -> DecordVideoReader:
        try:
            return DecordVideoReader(str(path), ctx=self._video_reader_context())
        except DECORDError as error:
            if self._decord_gpu_enabled:
                logger.warning("Decord GPU decode unavailable (%s); falling back to CPU.", error)
                self._decord_gpu_enabled = False
                return DecordVideoReader(str(path), ctx=self._video_reader_context(force_cpu=True))
            raise

    def _fetch_frames(self, reader: DecordVideoReader, indices: Sequence[int]) -> List[Image.Image]:
        if not indices:
            return []
        batch = reader.get_batch(list(indices))
        torch_batch = torch_dlpack.from_dlpack(batch.to_dlpack())
        frames: List[Image.Image] = []
        for frame_tensor in torch_batch:
            frames.append(self._tensor_to_image(frame_tensor))
        return frames

    def _tensor_to_image(self, frame_tensor: torch.Tensor) -> Image.Image:
        array = frame_tensor.detach().contiguous().to("cpu").numpy()
        return Image.fromarray(array)

    def _ensure_batch_list(self, values: object, length: int) -> List[object | None]:
        if values is None:
            result: List[object | None] = [None] * length
        elif isinstance(values, list):
            result = values
        elif isinstance(values, tuple):
            result = list(values)
        else:
            result = [values]
        if len(result) < length:
            result.extend([None] * (length - len(result)))
        elif len(result) > length:
            result = result[:length]
        return result

    def _ensure_list_length(self, values: Sequence[float] | None, length: int) -> List[float | None]:
        if values is None:
            return [None] * length
        if isinstance(values, np.ndarray):
            seq = values.tolist()
        elif isinstance(values, torch.Tensor):
            seq = values.detach().cpu().tolist()
        elif isinstance(values, list):
            seq = values
        elif isinstance(values, tuple):
            seq = list(values)
        else:
            seq = [values]
        if len(seq) < length:
            seq.extend([None] * (length - len(seq)))
        elif len(seq) > length:
            seq = seq[:length]
        return seq
