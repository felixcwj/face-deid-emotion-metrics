from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch


def require_cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required; CPU fallback is disabled")
    return torch.device("cuda:0")


@dataclass(frozen=True)
class PipelineConfig:
    base_dir: Path
    input_dir: Path
    output_dir: Path
    output_path: Path
    device: torch.device
    file_extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".mp4")
    max_frames_per_video: int = 32
    style_similarity_threshold: float = 70.0
    lpips_distance_max: float = 1.0
    max_files: Optional[int] = None
