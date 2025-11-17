from __future__ import annotations

import argparse
import logging
from pathlib import Path

from tqdm import tqdm

from .config import PipelineConfig, require_cuda_device
from .excel_writer import ExcelWriter
from .pipeline import MetricsPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Face de-identification and emotion metrics pipeline")
    parser.add_argument("--base-dir", required=True, help="Base directory containing mirrored input and output folders")
    parser.add_argument("--output", default="report.xlsx", help="Excel file to create")
    parser.add_argument("--max-frames-per-video", type=int, default=32, help="Maximum frames sampled per video")
    parser.add_argument("--style-threshold", type=float, default=70.0, help="LPIPS similarity threshold for style change detection")
    parser.add_argument("--lpips-distance-max", type=float, default=1.0, help="Maximum LPIPS distance mapped to 0 percent similarity")
    parser.add_argument("--max-files", type=int, help="Maximum number of files to process for debugging")
    parser.add_argument("--video-backend", default="decord", help="Video decoding backend to use (decord is the default)")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(message)s")
    base_dir = Path(args.base_dir).expanduser()
    input_dir = base_dir / "input"
    output_dir = base_dir / "output"
    if not input_dir.exists() or not output_dir.exists():
        raise FileNotFoundError("Base directory must contain input and output folders")
    device = require_cuda_device()
    logging.info("Using device: %s", device)
    output_path = Path(args.output).expanduser()
    config = PipelineConfig(
        base_dir=base_dir,
        input_dir=input_dir,
        output_dir=output_dir,
        output_path=output_path,
        device=device,
        max_frames_per_video=args.max_frames_per_video,
        style_similarity_threshold=args.style_threshold,
        lpips_distance_max=args.lpips_distance_max,
        max_files=args.max_files,
        video_backend=args.video_backend,
    )
    pipeline = MetricsPipeline(config)
    progress_bar = tqdm(total=0, unit="file", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {percentage:6.2f}%]")
    file_bar = tqdm(
        total=100,
        position=1,
        leave=False,
        bar_format="File {l_bar}{bar}| {n:5.1f}/100 [{elapsed}] {postfix}",
        disable=False,
    )

    def progress_callback(event: str, value) -> None:
        if event == "start":
            progress_bar.reset(total=int(value))
            progress_bar.refresh()
        elif event == "update":
            progress_bar.update(int(value))
        elif event == "file_progress_start":
            payload = value or {}
            relative_path = payload.get("relative_path", "")
            file_bar.reset(total=100)
            file_bar.n = 0
            file_bar.set_description(relative_path[:60])
            file_bar.set_postfix_str("starting")
            file_bar.refresh()
        elif event == "file_progress":
            payload = value or {}
            percent = float(payload.get("percent", 0.0))
            relative_path = payload.get("relative_path", "")
            message = payload.get("message", "")
            file_bar.n = max(0.0, min(percent, 100.0))
            if relative_path:
                file_bar.set_description(relative_path[:60])
            if message:
                file_bar.set_postfix_str(message)
            file_bar.refresh()
        elif event == "file_progress_end":
            file_bar.n = 100.0
            file_bar.set_postfix_str("done")
            file_bar.refresh()

    try:
        dataframe = pipeline.run(progress_callback=progress_callback)
    finally:
        progress_bar.close()
        file_bar.close()
    writer = ExcelWriter()
    writer.write(dataframe, output_path)
    logging.info("Wrote %s rows to %s", len(dataframe.index), output_path)


if __name__ == "__main__":
    main()
