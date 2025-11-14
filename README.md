# face-deid-emotion-metrics

GPU-ready toolkit that evaluates how well face de-identification preserves identity and emotion cues. The pipeline scans any base directory with mirrored `input/` and `output/` trees, pairs every `.jpg` and `.mp4`, computes five metrics per file, and exports a formatted Excel report.

## Expected folder structure

```
base_dir/
  input/
    subject_group/
      emotion_bucket/
        file.jpg or file.mp4
  output/
    subject_group/
      emotion_bucket/
        file.jpg or file.mp4
```

`input/` and `output/` must share the same relative file paths (for example, `emotion/male/positive/sample.jpg`).

## Metrics

1. **FaceNet similarity (%)**
   - Generate FaceNet embeddings for each matched face and compute cosine similarity `c`.
   - Map to percent with `(c + 1.0) * 50.0`.
2. **LPIPS style similarity (%)**
   - Run LPIPS (VGG backbone) on matched face crops to obtain perceptual distance `d`.
   - Clamp `d` to `[0, d_max]` (default `d_max = 1.0`) and compute `max(0, (1 - d / d_max)) * 100`.
3. **Final face similarity score (%)**
   - Determine whether a style/filter change occurred using LPIPS similarity.
   - If style similarity `< style_threshold` (default `70`), compute `0.3 * FaceNet + 0.7 * LPIPS`.
   - Otherwise reuse the FaceNet percent (LPIPS weight `0`).
4. **FER emotion similarity (%)**
   - Use the `fer` package to get emotion probability vectors aligned to `[angry, disgust, fear, happy, sad, surprise, neutral]`.
   - Let `p` and `q` be the original and de-identified vectors; compute `L1 = sum |p_k - q_k|` and map to percent with `(1 - L1 / 2) * 100`.
5. **DeepFace emotion similarity (%)**
   - Repeat the FER procedure using DeepFace emotion predictions and apply the same L1-based formula.

## Multiple persons and videos

- **Images**: If an image contains `N` persons, compute all five metrics per person and average them: `metric_image = (1 / N) * sum(metric_i)`.
- **Videos**: Uniformly sample up to 32 frame pairs (configurable). Track persons over time using FaceNet embeddings, match each tracked person with the corresponding de-identified faces, average FaceNet/LPIPS per person, detect style changes per person, and average FER/DeepFace probability vectors per person before applying the L1 formula. Final video metrics are the simple average across tracked persons.

## Excel report

Each processed file produces one row with columns:
1. filename (relative path under `input/`)
2. FaceNet similarity (%)
3. LPIPS similarity (%)
4. Final face similarity score (%)
5. FER emotion similarity (%)
6. DeepFace emotion similarity (%)

Formatting rules:
- Filename column auto-expands to fit the longest path.
- All cells are horizontally and vertically centered.
- The Final score column uses bold text.
- Thick vertical borders separate (LPIPS | Final) and (Final | FER) to visually split identity/style vs. emotion blocks.

## Requirements

- Windows 10+, PowerShell 7+
- NVIDIA GPU with CUDA-capable drivers (reference: GeForce RTX 3060, 12 GB VRAM)
- Python 3.10+
- CUDA-enabled builds of PyTorch (or DirectML wheels if CUDA is unavailable)

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Usage

Run directly via the CLI:

```powershell
python -m face_deid_emotion_metrics.cli --base-dir D:\RAPA --output D:\RAPA\rapa_report.xlsx --max-frames-per-video 32 --style-threshold 70
```

Use the PowerShell helper:

```powershell
pwsh -File .\scripts\run_rapa.ps1
```

Adjust `--base-dir`, `--output`, `--max-frames-per-video`, `--style-threshold`, and `--lpips-distance-max` for future datasets.

## Output

The pipeline writes a single Excel workbook summarizing all files. The Final score column is bold and separated by thick borders so reviewers can quickly distinguish identity/style metrics from FER/DeepFace emotion metrics.
