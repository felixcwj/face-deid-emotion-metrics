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

Each processed file (including both `.jpg` stills and `.mp4` videos) produces one row with columns:
1. filename (relative path under `input/`, extension included so `.jpg` and `.mp4` appear separately)
2. FaceNet (%)
3. LPIPS (%)
4. Final score (%)
5. FER emotion (%)
6. DeepFace emotion (%)
7. Person count
8. Duration (mp4 only)

Formatting rules:
- Filename column auto-expands to fit the longest path.
- Metric headers and values stay centered with one decimal place. Person count is an integer, Duration is text such as `45s` or `2m 10s`.
- Final score (%) stays bold and sits between thick vertical borders (LPIPS | Final) and (Final | FER).
- The header row, table frame, and last data row are outlined with thick borders so the table reads well in Excel.
- Columns E/F are widened so `FER emotion (%)` and `DeepFace emotion (%)` are fully visible in a fresh workbook.
- **Person count** is the number of distinct tracked identities (faces) detected in that file (images count detections in a single frame, videos count unique tracks across sampled frames).
- **Duration** is filled for `.mp4` rows using the rounded runtime (for example `58s` or `1m 42s`). Still-image rows keep the Duration cell empty.

## Requirements

- Windows 10+, PowerShell 7+
- NVIDIA GPU with CUDA-capable drivers (reference: GeForce RTX 3060, 12 GB VRAM). PyTorch must detect this GPU (`torch.cuda.is_available()`).
- Python 3.10+
- CUDA-enabled builds of PyTorch (the CLI fails fast if CUDA is missing)

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Usage

Run directly via the CLI (progress bar and `Using device: cuda:0` will appear when CUDA is active):

```powershell
python -m face_deid_emotion_metrics.cli --base-dir D:\RAPA --output D:\RAPA\rapa_report.xlsx --max-frames-per-video 32 --style-threshold 70
```

Use the PowerShell helper:

```powershell
pwsh -File .\scripts\run_rapa.ps1
```

Adjust `--base-dir`, `--output`, `--max-frames-per-video`, `--style-threshold`, and `--lpips-distance-max` for future datasets.

### Debug / sample runs

Add `--max-files N` to process only the first `N` matched file pairs (sorted by relative path). This option is strictly for troubleshooting and does not change the default behavior; omitting it processes the entire dataset.

## Output

The pipeline writes a single Excel workbook summarizing all files with the columns listed above. Each `.jpg` and `.mp4` appears as its own row, FaceNet/LPIPS/Final/FER/DeepFace values keep one decimal, the Final score column remains bold, Person count reflects the number of tracked identities in that file, and Duration is filled for videos only.

## Run on D:\RAPA from a fresh PowerShell 7 session

```powershell
pwsh
Set-Location C:\projects\face-deid-emotion-metrics
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
python -m face_deid_emotion_metrics.cli --base-dir D:\RAPA --output D:\RAPA\rapa_report_full.xlsx
```

You should see `Using device: cuda:0` and a live progress bar. If CUDA is unavailable, the run will abort immediately so you can fix the GPU environment before rerunning.
