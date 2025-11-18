from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from shutil import which
from typing import List, Sequence, Tuple

import logging
import numpy as np
import torch
from PIL import Image
from decord import VideoReader as DecordVideoReader
from decord import cpu as decord_cpu
from decord import gpu as decord_gpu
from decord._ffi.base import DECORDError
from torch.utils import dlpack as torch_dlpack


class VideoBackendError(RuntimeError):
    """Raised when the configured GPU video backend cannot be initialized."""


@dataclass(frozen=True)
class VideoSamplerConfig:
    device: torch.device
    backend: str = "auto"
    ffmpeg_hwaccel: str = "cuda"


class VideoPairSampler:
    """Loads matched frames from mirrored videos using the configured backend."""

    def __init__(self, config: VideoSamplerConfig) -> None:
        backend = config.backend.lower()
        if backend == "auto":
            self.backends: List[Tuple[str, object]] = []
            try:
                self.backends.append(("decord", _DecordBackend(config.device)))
            except VideoBackendError as error:
                logging.warning("Decord backend unavailable: %s", error)
            try:
                self.backends.append(("ffmpeg", _FfmpegBackend(hwaccel=config.ffmpeg_hwaccel)))
            except VideoBackendError as error:
                logging.warning("ffmpeg backend unavailable: %s", error)
            if not self.backends:
                raise VideoBackendError("No video backend is available (auto mode)")
        elif backend == "decord":
            self.backends = [("decord", _DecordBackend(config.device))]
        elif backend == "ffmpeg":
            self.backends = [("ffmpeg", _FfmpegBackend(hwaccel=config.ffmpeg_hwaccel))]
        else:
            raise ValueError(f"Unsupported video backend: {config.backend}")

    def sample(self, path_a: Path, path_b: Path, max_frames: int) -> Tuple[List[Image.Image], List[Image.Image]]:
        errors: List[str] = []
        for name, backend in self.backends:
            try:
                return backend.sample(path_a, path_b, max_frames)
            except VideoBackendError as error:
                logging.warning("Video backend %s failed for %s: %s", name, path_a, error)
                errors.append(f"{name}: {error}")
        raise VideoBackendError("; ".join(errors))

    @staticmethod
    def frame_indices(total: int, max_frames: int) -> List[int]:
        if total <= max_frames:
            return list(range(total))
        positions = np.linspace(0, total - 1, num=max_frames, dtype=np.int32)
        return sorted({int(value) for value in positions})


class _DecordBackend:
    def __init__(self, device: torch.device) -> None:
        self.device = device

    def sample(self, path_a: Path, path_b: Path, max_frames: int) -> Tuple[List[Image.Image], List[Image.Image]]:
        reader_a = self._open(path_a)
        reader_b = self._open(path_b)
        total = min(len(reader_a), len(reader_b))
        if total <= 0:
            return [], []
        indices = VideoPairSampler.frame_indices(total, max_frames)
        frames_a = self._fetch(reader_a, indices)
        frames_b = self._fetch(reader_b, indices)
        paired = min(len(frames_a), len(frames_b))
        if paired <= 0:
            return [], []
        return frames_a[:paired], frames_b[:paired]

    def _open(self, path: Path) -> DecordVideoReader:
        try:
            return DecordVideoReader(str(path), ctx=self._context())
        except DECORDError as error:  # pragma: no cover - depends on system config
            message = (
                "Failed to initialize the GPU Decord backend. "
                "Use the WSL setup documented in docs/wsl.md to install the CUDA-enabled wheel. "
                f"Original error: {error}"
            )
            raise VideoBackendError(message) from error

    def _fetch(self, reader: DecordVideoReader, indices: Sequence[int]) -> List[Image.Image]:
        if not indices:
            return []
        batch = reader.get_batch(list(indices))
        torch_batch = torch_dlpack.from_dlpack(batch.to_dlpack())
        return [self._tensor_to_image(frame_tensor) for frame_tensor in torch_batch]

    def _context(self):
        if self.device.type == "cuda":
            index = self.device.index if self.device.index is not None else 0
            return decord_gpu(index)
        return decord_cpu(0)

    def _tensor_to_image(self, frame_tensor: torch.Tensor) -> Image.Image:
        array = frame_tensor.detach().contiguous().to("cpu").numpy()
        return Image.fromarray(array)


@dataclass
class _VideoMeta:
    frame_count: int
    width: int
    height: int


class _FfmpegBackend:
    def __init__(self, hwaccel: str = "cuda") -> None:
        self.hwaccel = hwaccel
        self.ffmpeg = which("ffmpeg")
        self.ffprobe = which("ffprobe")
        if not self.ffmpeg or not self.ffprobe:
            raise VideoBackendError("ffmpeg/ffprobe not found. Install ffmpeg with NVDEC support to use the ffmpeg backend.")

    def sample(self, path_a: Path, path_b: Path, max_frames: int) -> Tuple[List[Image.Image], List[Image.Image]]:
        meta_a = self._probe(path_a)
        meta_b = self._probe(path_b)
        total = min(meta_a.frame_count, meta_b.frame_count)
        if total <= 0:
            return [], []
        indices = VideoPairSampler.frame_indices(total, max_frames)
        frames_a = self._extract_frames(path_a, meta_a, indices)
        frames_b = self._extract_frames(path_b, meta_b, indices)
        paired = min(len(frames_a), len(frames_b))
        if paired <= 0:
            return [], []
        return frames_a[:paired], frames_b[:paired]

    def _probe(self, path: Path) -> _VideoMeta:
        cmd = [
            self.ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames,nb_frames,duration,r_frame_rate,width,height",
            "-of",
            "json",
            str(path),
        ]
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as error:  # pragma: no cover - depends on system ffmpeg
            raise VideoBackendError(f"ffprobe failed for {path}: {error.stderr}") from error
        try:
            data = json.loads(result.stdout)
            stream = data["streams"][0]
        except Exception as error:  # pragma: no cover - defensive
            raise VideoBackendError(f"Unable to parse ffprobe metadata for {path}: {error}") from error
        width = int(stream.get("width") or 0)
        height = int(stream.get("height") or 0)
        frame_count = self._resolve_frame_count(stream)
        if frame_count <= 0:
            raise VideoBackendError(f"Unable to determine frame count for {path}")
        return _VideoMeta(frame_count=frame_count, width=width, height=height)

    def _resolve_frame_count(self, stream: dict) -> int:
        for key in ("nb_read_frames", "nb_frames"):
            value = stream.get(key)
            if value and value != "N/A":
                try:
                    return int(float(value))
                except ValueError:
                    continue
        duration = stream.get("duration")
        frame_rate = stream.get("r_frame_rate")
        if duration and frame_rate and duration != "N/A" and frame_rate != "0/0":
            try:
                num, den = frame_rate.split("/")
                fps = float(num) / float(den or 1)
                seconds = float(duration)
                estimated = int(max(1.0, round(seconds * fps)))
                return estimated
            except (ValueError, ZeroDivisionError):
                pass
        return 0

    def _extract_frames(self, path: Path, meta: _VideoMeta, indices: Sequence[int]) -> List[Image.Image]:
        if not indices:
            return []
        select_expr = "+".join(f"eq(n\\,{index})" for index in indices)
        vf = f"select='{select_expr}',format=rgb24"
        cmd = [
            self.ffmpeg,
            "-v",
            "error",
            "-hwaccel",
            self.hwaccel,
            "-i",
            str(path),
            "-vf",
            vf,
            "-vsync",
            "0",
            "-f",
            "rawvideo",
            "pipe:1",
        ]
        try:
            result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as error:  # pragma: no cover - system dependent
            stderr = error.stderr.decode("utf-8", errors="ignore") if error.stderr else str(error)
            raise VideoBackendError(
                f"ffmpeg hardware decode failed for {path}. Verify that 'ffmpeg -hwaccels' lists 'cuda'. Stderr: {stderr}"
            ) from error
        frame_size = meta.width * meta.height * 3
        if frame_size <= 0:
            return []
        data = result.stdout
        available = len(data) // frame_size
        if available <= 0:
            return []
        array = np.frombuffer(data[: available * frame_size], dtype=np.uint8)
        array = array.reshape((available, meta.height, meta.width, 3))
        return [Image.fromarray(frame) for frame in array]
