from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple

from openpyxl import Workbook
from tqdm import tqdm

from .config import PipelineConfig, require_cuda_device
from .pipeline import MetricsPipeline
from .sample_workbook import SampleWorkbook, SheetSpec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Face de-identification and emotion metrics pipeline")
    parser.add_argument("--base-dir", required=True, help="Base directory containing mirrored input and output folders")
    parser.add_argument("--output", help="Excel file to create (defaults to <base>/rapa_report_samples.xlsx)")
    parser.add_argument("--max-frames-per-video", type=int, default=32, help="Maximum frames sampled per video")
    parser.add_argument("--style-threshold", type=float, default=70.0, help="LPIPS similarity threshold for style change detection")
    parser.add_argument("--lpips-distance-max", type=float, default=1.0, help="Maximum LPIPS distance mapped to 0 percent similarity")
    parser.add_argument("--max-files", type=int, help="Maximum number of files to process for debugging")
    parser.add_argument("--video-backend", default="decord", help="Video decoding backend to use (decord is the default)")
    parser.add_argument("--profile-only", action="store_true", help="Process a 10-sample deterministic profile run instead of writing Excel")
    parser.add_argument("--seed", type=int, default=42, help="Seed used for sampling files in profiling mode")
    parser.add_argument("--resize-512", action="store_true", help="Resize each frame or image to 512x512 before inference")
    parser.add_argument("--profile-kind", choices=["all", "image", "video"], default="all", help="Restrict profiling samples to images, videos, or all")
    parser.add_argument("--debug-random-10", action="store_true", help="Generate a 10-row random debug sample (5 videos + 5 images)")
    parser.add_argument("--debug-output", default="random_debug_10.xlsx", help="Path for the debug sample workbook")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    return parser.parse_args()


def _print_profile_summary(result) -> None:
    print(f"Profiling {result.processed} of {result.requested} requested pairs")
    totals = dict(result.totals)
    total_duration = totals.get("total", result.elapsed)
    stages = ["load", "resize", "facenet", "lpips", "emoti", "other", "total"]
    for stage in stages:
        duration = totals.get(stage, 0.0)
        milliseconds = duration * 1000.0
        if stage == "total":
            percent = 100.0 if total_duration > 0 else 0.0
        elif total_duration > 0:
            percent = (duration / total_duration) * 100.0
        else:
            percent = 0.0
        print(f"{stage:<7}: {milliseconds:9.0f} ms ({percent:5.1f}%)")


def _write_debug_sample(records, output_path: Path) -> None:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "debug_sample"
    headers = ["filename", "media_type", "facenet_percent", "lpips_percent", "final_score_percent"]
    sheet.append(headers)
    for record in records:
        sheet.append(
            [
                record.get("filename", ""),
                record.get("media_type", ""),
                "" if record.get("facenet_percent") is None else float(record["facenet_percent"]),
                "" if record.get("lpips_percent") is None else float(record["lpips_percent"]),
                "" if record.get("final_score_percent") is None else float(record["final_score_percent"]),
            ]
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(output_path)


def _print_debug_sample(records) -> None:
    print("Random debug sample (filename | media | FaceNet | LPIPS | Final score)")
    def fmt(value):
        if value is None:
            return "NA"
        return f"{value:5.1f}"
    for record in records:
        facenet = record.get("facenet_percent")
        lpips = record.get("lpips_percent")
        final_score = record.get("final_score_percent")
        line = f"{record.get('filename',''):<20} | {record.get('media_type',''):>5} | {fmt(facenet)} | {fmt(lpips)} | {fmt(final_score)}"
        print(line)


def _default_output_path(base_dir: Path, override: str | None) -> Path:
    if override:
        return Path(override).expanduser()
    return (base_dir / "rapa_report_samples.xlsx").expanduser()


def _first_last(paths: List[str], count: int) -> List[str]:
    if not paths or count <= 0:
        return []
    count = min(count, len(paths))
    first = paths[:count]
    last = paths[-count:] if len(paths) > count else []
    combined: List[str] = []
    for entry in first + last:
        if entry not in combined:
            combined.append(entry)
    return combined


def _random_sample(paths: List[str], size: int, rng: random.Random) -> List[str]:
    if not paths or size <= 0:
        return []
    if size >= len(paths):
        return list(paths)
    return rng.sample(paths, size)


def _select_sample_specs(pairs: List[Tuple[str, Path, Path]], seed: int) -> List[SheetSpec]:
    relative_paths = [relative for relative, _, _ in pairs]
    sample_40 = _first_last(relative_paths, 20)
    rng = random.Random(seed)
    sample_100 = _random_sample(relative_paths, 100, rng)
    sample_500 = _random_sample(relative_paths, 500, rng)
    return [
        SheetSpec("sample_40", sample_40),
        SheetSpec("sample_100", sample_100),
        SheetSpec("sample_500", sample_500),
    ]


def _collect_pending_paths(specs: List[SheetSpec], workbook: SampleWorkbook) -> List[str]:
    pending: List[str] = []
    seen: set[str] = set()
    for spec in specs:
        for relative in spec.paths:
            if workbook.has_entry(spec.name, relative):
                continue
            if relative in seen:
                continue
            pending.append(relative)
            seen.add(relative)
    return pending


def _write_rationale(base_dir: Path) -> None:
    lines: List[str] = []
    lines.append("Final score blends FaceNet identity similarity (20%) with LPIPS perceptual similarity (80%) so that style changes dominate while still penalizing cases where identity survives.")
    lines.append("This weighting emphasizes filter artifacts, texture changes, and photorealism, which are the primary risks in RAPA de-identification reviews.")
    lines.append("")
    references = [
        (
            "Schroff et al. (2015) FaceNet: A Unified Embedding for Face Recognition and Clustering.",
            [
                "FaceNet demonstrates that cosine distance between embeddings predicts whether two faces belong to the same person, so it supplies the identity component of the score.",
                "Keeping FaceNet at 20% ensures we still flag cases where the anonymized output is too recognizable."
            ],
        ),
        (
            "Zhang et al. (2018) The Unreasonable Effectiveness of Deep Features as a Perceptual Metric (LPIPS).",
            [
                "LPIPS is strongly correlated with human judgments of perceptual similarity across filters, textures, and style transfers.",
                "Weighting LPIPS at 80% lets the metric respond to the aggressive filters used in our pipeline without relying on ad-hoc heuristics."
            ],
        ),
        (
            "Hukkelås et al. (2019) DeepPrivacy: A Generative Adversarial Network for Face Anonymization.",
            [
                "DeepPrivacy emphasizes that anonymization quality depends on visual plausibility, not just identity removal.",
                "Our LPIPS-heavy score mirrors their observation that reviewers care most about whether the anonymized output still looks stylistically linked to the original."
            ],
        ),
        (
            "Karras et al. (2019) Analyzing and Improving the Image Quality of StyleGAN.",
            [
                "StyleGAN research shows that perceptual metrics align with human opinions of texture fidelity and overall realism.",
                "Prioritizing LPIPS therefore helps us track the same visual cues that modern GAN evaluations rely on."
            ],
        ),
    ]
    for reference, sentences in references:
        lines.append(reference)
        for sentence in sentences:
            lines.append(sentence)
        lines.append("")
    rationale_path = base_dir / "근거.txt"
    rationale_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def _run_sample_workbook(pipeline: MetricsPipeline, base_dir: Path, output_path: Path, seed: int) -> None:
    pairs = pipeline.matched_pairs()
    if not pairs:
        raise RuntimeError("No matched input/output pairs were found under the base directory")
    specs = _select_sample_specs(pairs, seed)
    headers = [
        "filename",
        "media_type",
        "facenet_percent",
        "lpips_percent",
        "final_score_percent",
        "emoti_emotion_percent",
        "person_count",
        "duration_label",
        "load_ms",
        "resize_ms",
        "facenet_ms",
        "lpips_ms",
        "emoti_ms",
        "other_ms",
        "total_ms",
        "error",
    ]
    workbook = SampleWorkbook(output_path, specs, headers)
    pending = _collect_pending_paths(specs, workbook)
    records: Dict[str, Dict[str, object]] = {}
    if pending:
        progress_bar = tqdm(total=len(pending), unit="file", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
        def progress_callback(event: str, value) -> None:
            if event == "start":
                progress_bar.reset(total=int(value))
                progress_bar.refresh()
            elif event == "update":
                progress_bar.update(int(value))
        records = pipeline.compute_records_for_paths(pending, progress_callback=progress_callback)
        progress_bar.close()
    for spec in specs:
        for relative in spec.paths:
            if workbook.has_entry(spec.name, relative):
                continue
            record = records.get(relative)
            if not record:
                logging.warning("No metrics captured for %s", relative)
                continue
            workbook.append_record(spec.name, record)
    workbook.save()
    _write_rationale(base_dir)
    logging.info("Wrote sample workbook to %s", output_path)


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
    output_path = _default_output_path(base_dir, args.output)
    resize_target = (512, 512) if args.resize_512 else None
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
        resize_to=resize_target,
        style_change_lpips_threshold=args.style_threshold,
    )
    pipeline = MetricsPipeline(config)
    if args.profile_only:
        result = pipeline.run_profile(sample_count=10, seed=args.seed, kind=args.profile_kind)
        _print_profile_summary(result)
        return
    if args.debug_random_10:
        records = pipeline.run_random_sample(seed=args.seed, image_count=5, video_count=5)
        if not records:
            raise RuntimeError("No files available for debug sampling")
        debug_output_path = Path(args.debug_output).expanduser()
        _write_debug_sample(records, debug_output_path)
        _print_debug_sample(records)
        logging.info("Wrote debug sample to %s", debug_output_path)
        return
    _run_sample_workbook(pipeline, base_dir, output_path, args.seed)


if __name__ == "__main__":
    main()
