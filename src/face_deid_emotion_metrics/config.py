from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class PipelineConfig:
    base_dir: Path
    input_dir: Path
    output_dir: Path
    output_path: Path
    file_extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".mp4")
    max_frames_per_video: int = 32
    style_similarity_threshold: float = 70.0
    lpips_distance_max: float = 1.0
