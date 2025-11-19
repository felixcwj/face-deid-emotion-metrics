from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.backends.cudnn as cudnn

DEFAULT_EMOTI_MODEL_NAME = "enet_b0_8_va_mtl"
MTCNN_THRESHOLDS = (0.4, 0.5, 0.5)
MTCNN_MIN_FACE_SIZE = 20
DETECTOR_ROTATION_ANGLES = (0, -30, 30, -60, 60)


def require_cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required; CPU fallback is disabled")
    cudnn.benchmark = True
    return torch.device("cuda:0")


FINAL_SCORE_FACENET_WEIGHT = 0.2
FINAL_SCORE_LPIPS_WEIGHT = 0.8
FINAL_SCORE_SCALE = 0.6154
FINAL_SCORE_BIAS = 0.6154


@dataclass(frozen=True)
class PipelineConfig:
    base_dir: Path
    input_dir: Path
    output_dir: Path
    output_path: Path
    device: torch.device
    file_extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".mp4", ".mov", ".mkv")
    max_frames_per_video: int = 16
    lpips_distance_max: float = 1.0
    max_files: Optional[int] = None
    emoti_model_name: str = DEFAULT_EMOTI_MODEL_NAME
    video_backend: str = "auto"
    resize_to: Optional[Tuple[int, int]] = None
    mtcnn_thresholds: Tuple[float, float, float] = MTCNN_THRESHOLDS
    mtcnn_min_face_size: int = MTCNN_MIN_FACE_SIZE
    detector_rotation_angles: Tuple[int | float, ...] = DETECTOR_ROTATION_ANGLES
def final_score_percent(
    facenet_percent: float,
    lpips_percent: float,
    facenet_weight: float | None = None,
    lpips_weight: float | None = None,
    scale: float | None = None,
    bias: float | None = None,
) -> float:
    facenet_weight = FINAL_SCORE_FACENET_WEIGHT if facenet_weight is None else facenet_weight
    lpips_weight = FINAL_SCORE_LPIPS_WEIGHT if lpips_weight is None else lpips_weight
    scale = FINAL_SCORE_SCALE if scale is None else scale
    bias = FINAL_SCORE_BIAS if bias is None else bias
    facenet_norm = max(0.0, min(1.0, facenet_percent / 100.0))
    lpips_norm = max(0.0, min(1.0, lpips_percent / 100.0))
    weight_sum = facenet_weight + lpips_weight
    if weight_sum <= 0.0:
        weighted = 0.0
    else:
        weighted = (facenet_weight * facenet_norm + lpips_weight * lpips_norm) / weight_sum
    value = 100.0 * (scale * (weighted * weighted) + bias)
    risk = value - 60.0
    return max(0.0, min(40.0, risk))
