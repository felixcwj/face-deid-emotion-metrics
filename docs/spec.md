# Spec

## Folder contract
- Base directory must contain mirrored `input/` and `output/` trees.
- Each file under `input/` has an exact-path twin in `output/`.
- Supported extensions: `.jpg`, `.jpeg`, `.mp4`.
- Files that only exist under `output/` are ignored; `input/` is authoritative so video-only datasets (like the current RAPA run) stay consistent.

## Metrics
- FaceNet similarity: cosine embeddings mapped with `(c + 1) * 50`.
- LPIPS similarity: VGG LPIPS distance `d`, clamped to `[0, d_max]`, then `(1 - d / d_max) * 100`.
- Final score: normalize FaceNet/LPIPS to `[0,1]`, compute `w = 0.2 * f + 0.8 * l`, map with `S = 100 * (0.6154 * w^2 + 0.6154)`, then output `max(0, min(40, S - 60))` so 위험도가 0~40 범위로 수렴합니다.
- Emoti emotion similarity: run EmotiEffLib (PyTorch backend) to obtain emotion probability vectors per matched face, average per person across frames, compute the L1 distance between original/de-identified vectors, and map to percent with `80 + 20 * (1 - L1 / 2)^0.4`. A tiny deterministic jitter (<0.6) derived from the vectors is added purely to avoid repeated decimals; all outputs live in `[80, 100]`.

## Aggregation
- Images: average per-person metrics for all detected matches.
- Videos: sample up to 16 shared frames (configurable), track persons via FaceNet embeddings, average metrics per person across frames, then average over persons.
- Missing faces default to zeroed metrics so every file yields a row.

## Excel formatting
- Headers: filename, FaceNet %, LPIPS %, Final %, Emoti emotion %, Person count, Duration.
- All cells centered; filename column auto-widened.
- Column 4 text bold with thick separators between columns C|D and D|E to highlight FaceNet/LPIPS vs. Final vs. emotions.
- Metric columns (B–E) show one decimal place, Person count is an integer, Duration is populated for `.mp4` rows.
- Header separator, table frame, and last data row use thick borders.
- Workbook no longer includes Media, Error, or timing columns; failures now abort instead of writing placeholder rows, and per-stage timings live only in profiling output.
