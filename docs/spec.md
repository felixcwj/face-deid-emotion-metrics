# Spec

## Folder contract
- Base directory must contain mirrored `input/` and `output/` trees.
- Each file under `input/` has an exact-path twin in `output/`.
- Supported extensions: `.jpg`, `.jpeg`, `.mp4`.

## Metrics
- FaceNet similarity: cosine embeddings mapped with `(c + 1) * 50`.
- LPIPS similarity: VGG LPIPS distance `d`, clamped to `[0, d_max]`, then `(1 - d / d_max) * 100`.
- Final score: if LPIPS similarity `< threshold`, use `0.3 * FaceNet + 0.7 * LPIPS`; otherwise FaceNet only.
- Emoti emotion similarity: run EmotiEffLib (PyTorch backend) to obtain emotion probability vectors per matched face, average per person across frames, compute the L1 distance between original/de-identified vectors, and map to percent with `(1 - L1 / 2) * 100`.

## Aggregation
- Images: average per-person metrics for all detected matches.
- Videos: sample up to 32 shared frames, track persons via FaceNet embeddings, average metrics per person across frames, then average over persons.
- Missing faces default to zeroed metrics so every file yields a row.

## Excel formatting
- Headers: filename, FaceNet %, LPIPS %, Final %, Emoti emotion %, Person count, Duration.
- All cells centered; filename column auto-widened.
- Column 4 text bold with thick separators between columns C|D and D|E to highlight FaceNet/LPIPS vs. Final vs. emotions.
- Metric columns (B–E) show one decimal place, Person count is an integer, Duration is populated for `.mp4` rows.
- Header separator, table frame, and last data row use thick borders.
