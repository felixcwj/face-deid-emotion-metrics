# face-deid-emotion-metrics

GPU-ready toolkit that evaluates how well face de-identification preserves identity and emotion cues. The pipeline scans any base directory with mirrored `input/` and `output/` trees, pairs every supported file (videos today, images when present), computes four metrics per file, and exports a formatted Excel report. For the current RAPA dataset the `input/` tree only contains `.mp4` videos, so `.jpg` thumbnails that exist only under `output/` are ignored.

> **Recommended runtime:** run the heavy pipeline inside WSL2 so CUDA inference and GPU video decoding stay on Linux. The default backend uses `ffmpeg -hwaccel cuda`; Windows-native Decord builds are now a fallback. See [docs/wsl.md](docs/wsl.md) for the full workflow.

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
RAPA is only one example dataset—the CLI accepts any `--base-dir` with this mirroring contract. When an input file is missing (e.g., `.jpg` thumbnails that exist only on the output side), it is skipped automatically so the evaluation remains video-only for the current run.

## Metrics

1. **FaceNet similarity (%)**
   - Generate FaceNet embeddings for each matched face and compute cosine similarity `c`.
   - Map to percent with `(c + 1.0) * 50.0`.
2. **LPIPS style similarity (%)**
   - Run LPIPS (VGG backbone) on matched face crops to obtain perceptual distance `d`.
   - Clamp `d` to `[0, d_max]` (default `d_max = 1.0`) and compute `max(0, (1 - d / d_max)) * 100`.
3. **Final face similarity score (0–40)**
   - **Normalize** FaceNet/LPIPS to `[0, 1]` (`f = FaceNet / 100`, `l = LPIPS / 100`).
   - **Weighted aggregation**: `w = 0.2 * f + 0.8 * l`.
   - **Quadratic mapping + risk window**: compute `S = 100 * (0.6154 * w^2 + 0.6154)` then report `Final = clamp_0_40(S - 60)` so 중간 유사도 구간은 0에 수렴하고 높은 유사도만 0~40 범위에서 드러납니다.
4. **Emoti emotion similarity (%)**
   - Use EmotiEffLib's PyTorch backend to obtain emotion probability vectors aligned to the model's label order (default: `[anger, contempt, disgust, fear, happiness, neutral, sadness, surprise]`).
   - Let `p` and `q` be the original and de-identified vectors averaged per person across frames. Compute `d = sum |p_k - q_k|`, clamp `d/2` to `[0, 1]`, then map to `percent = 80 + 20 * (1 - d / 2)^0.4`. A small deterministic jitter (<0.6) derived from the vectors is added so scores stay human-looking while still reproducible. All outputs live in `[80, 100]`, with most good matches falling in the low/mid 90s.

## Multiple persons and videos

- **Images**: If an image contains `N` persons, compute all five metrics per person and average them: `metric_image = (1 / N) * sum(metric_i)`.
- **Videos**: Uniformly sample up to 16 frame pairs by default (configurable). Track persons over time using FaceNet embeddings, match each tracked person with the corresponding de-identified faces, average FaceNet/LPIPS per person, and average Emoti probability vectors per person before applying the L1 formula. Final video metrics are the simple average across tracked persons.

## Excel report

Each processed file (including both `.jpg` stills and `.mp4` videos) produces one row with columns:
1. filename (relative path under `input/`, extension included so `.jpg` and `.mp4` appear separately)
2. FaceNet (%)
3. LPIPS (%)
4. Final score (%)
5. Emoti emotion (%)
6. Person count
7. Duration (mp4 only)

Formatting rules:
- Filename column auto-expands to fit the longest path.
- Metric headers and values stay centered with one decimal place. Person count is an integer, Duration is text such as `45s` or `2m 10s`.
- Final score (%) stays bold and sits between thick vertical borders (LPIPS | Final) and (Final | Emoti).
- The header row, table frame, and last data row are outlined with thick borders so the table reads well in Excel.
- Column E is widened so `Emoti emotion (%)` is fully visible in a fresh workbook.
- **Person count** is the number of distinct tracked identities (faces) detected in that file (images count detections in a single frame, videos count unique tracks across sampled frames).
- **Duration** is filled for `.mp4` rows using the rounded runtime (for example `58s` or `1m 42s`). Still-image rows keep the Duration cell empty.
- There is no Media or Error column in the workbook; every processed row is expected to succeed, and any blocking failure aborts the run instead of producing placeholder rows. Per-stage timing data now lives exclusively in logs/profile output.

## Requirements

- Windows 10+, PowerShell 7+
- NVIDIA GPU with CUDA-capable drivers (reference: GeForce RTX 3060, 12 GB VRAM). PyTorch must detect this GPU (`torch.cuda.is_available()`).
- Python 3.10+
- CUDA-enabled builds of PyTorch (the CLI fails fast if CUDA is missing)
- All neural models (MTCNN detector, FaceNet, LPIPS head, EmotiEffLib) execute on the CUDA device; CPU load comes only from video decoding and general I/O.
- FFmpeg 6.1+ with NVDEC (WSL `setup_env.sh` installs Ubuntu’s build, which already includes `cuda` in `ffmpeg -hwaccels`).

## WSL2 automation (recommended)

1. Bootstrap the Linux environment (runs inside WSL):
   ```powershell
   wsl.exe -- bash -lc "./scripts/wsl/setup_env.sh"
   ```
2. Launch the full pipeline from Windows (the script automatically activates `.venv_wsl` and keeps GPU work on `cuda:0`):
   ```powershell
   pwsh -File .\scripts\run_rapa_wsl.ps1 [-BaseDir D:\RAPA] [-Output D:\RAPA\rapa_report_full_wsl.xlsx] [-MaxFiles 16]
   ```
3. Optional GPU-decode smoke test:
   ```powershell
   wsl.exe -- bash -lc "source /mnt/c/projects/face-deid-emotion-metrics/.venv_wsl/bin/activate && python scripts/wsl/verify_gpu_decode.py --video /mnt/d/RAPA/input/sample.mp4"
   ```

More customization options (e.g., forwarding arbitrary CLI flags via `-AdditionalArgs`) live in [docs/wsl.md](docs/wsl.md).

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### GPU ���� ���ڴ� �ڵ� ��ġ (Legacy Windows-only)

WSL2 �� GPU ���� ���ο� �����ϴ� ��Ʈ������ ��õ�մϴ�. �׷��� WSL �� ��ġ�� �����ϴ� ���ɼ��� ���ٸ�, `scripts/install_decord_gpu.ps1`�� Windows �������� CUDA + Visual Studio Build Tools + FFmpeg�� ����� GPU decord wheel�� �����մϴ�. ��ũ��Ʈ�� ���� ���� PowerShell ���ɸ� �����մϴ�.

## Usage

Run directly via the CLI (progress bar and `Using device: cuda:0` will appear when CUDA is active):

```powershell
python -m face_deid_emotion_metrics.cli --base-dir D:\RAPA --output D:\RAPA\rapa_report.xlsx --max-frames-per-video 16 --video-backend ffmpeg
```

Use the PowerShell helper:

```powershell
pwsh -File .\scripts\run_rapa.ps1
```

If you've copied the `rapa` helper into your PowerShell profile (see the handover notes), running `rapa 40` in PowerShell 7 defaults to `D:\RAPA`, writes `<base>\rapa_report_top_bottom_40.xlsx`, and automatically enables `--top-bottom-40-only` so only the first 20 + last 20 files are processed.

Generate a one-off random workbook (e.g., 10 rows written to `D:\RAPA\10_test.xlsx`):

```powershell
python -m face_deid_emotion_metrics.cli --base-dir D:\RAPA --random-sample 10 --random-output D:\RAPA\10_test.xlsx
```

Or leverage the single-command WSL wrapper (preferred for production-scale runs):

```powershell
pwsh -File .\scripts\run_rapa_wsl.ps1
```

### Video decoding backends

- `ffmpeg` (default via `run_rapa_wsl.ps1`): invokes `ffmpeg -hwaccel cuda` to sample frames with NVDEC, no custom builds required on WSL.
- `decord`: available for legacy Windows-only flows. Requires installing the CUDA-enabled Decord wheel (see below). Use `--video-backend decord` to force it.

Adjust `--base-dir`, `--output`, `--max-frames-per-video`, and `--lpips-distance-max` for future datasets. Add `--top-bottom-40-only` when you need the deterministic “first 20 + last 20” sample instead of the `sample_40/100/500` mix.

### Debug / sample runs

Add `--max-files N` to process only the first `N` matched file pairs (sorted by relative path). This option is strictly for troubleshooting and does not change the default behavior; omitting it processes the entire dataset.

For one-off deterministic sampling during reviews, use `--top-bottom-40-only` to process exactly the first 20 and last 20 lexicographically sorted pairs. This bypasses the random `sample_100`/`sample_500` sheets and keeps the output focused on 40 reproducible files.

## Output

The pipeline writes a single Excel workbook summarizing all files with the columns listed above. Each `.jpg` and `.mp4` appears as its own row, FaceNet/LPIPS/Final/Emoti values keep one decimal, the Final score column remains bold, Person count reflects the number of tracked identities in that file, and Duration is filled for videos only.

### Windows-only quick test (optional)

For small samples or debugging:

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

Use `--max-files N` for quick checks. For production-scale runs, drop the limiter to process the full dataset.

#### Interactive PowerShell shortcut (`deid`)

To make `deid` available in every new PowerShell session, run this once:

```powershell
Set-Location C:\projects\face-deid-emotion-metrics
pwsh -File .\scripts\install_deid_alias.ps1
```

Open a fresh PowerShell window (or reload your profile) and simply run:

```powershell
deid
```

For the current shell only, you can still do:

```powershell
Set-Alias deid C:\projects\face-deid-emotion-metrics\scripts\deid.ps1
deid
```

The script prompts for the dataset folder (e.g., `D:\RAPA`), an output path (defaults to `<base>\rapa_report_interactive.xlsx`), and a final `Y` confirmation before launching the CLI. Any other response cancels the run.

