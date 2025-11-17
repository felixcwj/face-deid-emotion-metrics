#!/usr/bin/env python
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from face_deid_emotion_metrics.video_reader import VideoPairSampler, VideoSamplerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick sanity check for GPU video decoding.")
    parser.add_argument("--video", required=True, help="Path to a test .mp4 video (WSL path, e.g. /mnt/d/RAPA/input/foo.mp4)")
    parser.add_argument("--frames", type=int, default=8, help="Number of frames to decode")
    parser.add_argument("--backend", choices=("ffmpeg", "decord"), default="ffmpeg", help="Video backend to exercise")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_path = Path(args.video)
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")
    device = torch.device("cuda:0")
    sampler = VideoPairSampler(VideoSamplerConfig(device=device, backend=args.backend))
    start = time.time()
    frames_a, _ = sampler.sample(video_path, video_path, args.frames)
    elapsed = time.time() - start
    if not frames_a:
        raise SystemExit("Decoder returned zero frames.")
    print(f"Decoded {len(frames_a)} frames from {video_path} via {args.backend} backend in {elapsed:.2f}s on {device}.")


if __name__ == "__main__":
    main()
