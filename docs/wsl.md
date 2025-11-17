# WSL2 + CUDA pipeline

This project now prefers running the heavy pipeline inside **WSL2** because the Linux CUDA stack offers reliable GPU decoding (NVDEC) without the brittle Visual Studio builds.

## Prerequisites

1. Windows 10/11 with WSL2 enabled.
2. NVIDIA driver 560+ with CUDA 12.6 support.
3. An Ubuntu (or Debian-based) distribution registered in WSL. Install the official `cuda-toolkit` package so `nvidia-smi` works inside WSL:
   ```powershell
   wsl.exe -- sudo apt-get update
   wsl.exe -- sudo apt-get install -y cuda-toolkit
   ```
4. The dataset must be accessible from Windows as `D:\RAPA` (exposed to WSL as `/mnt/d/RAPA`).

Verify GPU access from WSL:

```powershell
wsl.exe -- nvidia-smi
```

## One-time environment bootstrap

From Windows PowerShell (repo root `C:\projects\face-deid-emotion-metrics`):

```powershell
wsl.exe -- bash -lc "./scripts/wsl/setup_env.sh"
```

The script performs:

- Installs apt packages (`python3-venv`, `build-essential`, `ffmpeg`, GPU video dependencies).
- Creates `.venv_wsl/` and installs Python requirements in editable mode.
- Optionally installs a CUDA-enabled Decord wheel if you plan to force `--video-backend decord`.

## GPU decoding smoke test

Use the helper script to confirm NVDEC decoding before running the full dataset (this exercises the default `ffmpeg` backend):

```powershell
wsl.exe -- bash -lc "
    source /mnt/c/projects/face-deid-emotion-metrics/.venv_wsl/bin/activate &&
    python scripts/wsl/verify_gpu_decode.py --video /mnt/d/RAPA/input/sample.mp4 --frames 16
"
```

Expected output:

```
Decoded 16 frames from /mnt/d/RAPA/input/sample.mp4 on cuda:0 in 0.37s.
```

Any `ffmpeg` errors usually mean CUDA hwaccel is not available (`ffmpeg -hwaccels` must list `cuda`). If you switch back to the Decord backend, reinstall its CUDA wheel via `setup_env.sh`.

## Full run from Windows

After the venv is ready, invoke the entire pipeline from Windows with one command (runs the CLI with `--video-backend ffmpeg`):

```powershell
pwsh -File .\scripts\run_rapa_wsl.ps1
```

Options:

- `-BaseDir 'E:\RAPA2'` – custom dataset path.
- `-Output 'D:\RAPA\rapa_report_full_wsl.xlsx'` – output Excel location.
- `-MaxFiles 8` – debug runs.
- `-AdditionalArgs '--max-frames-per-video' '16'` – any CLI flag is forwarded to `python -m face_deid_emotion_metrics.cli`.

The script transparently converts Windows paths to `/mnt/<drive>/...`, runs `scripts/wsl/run_pipeline.sh` inside WSL, and ensures CUDA decoding + inference stay on `cuda:0`.

## Manual WSL CLI usage

Inside WSL you can always run the CLI directly (after `. .venv_wsl/bin/activate`):

```bash
python -m face_deid_emotion_metrics.cli \
    --base-dir /mnt/d/RAPA \
    --output /mnt/d/RAPA/rapa_report_full_wsl.xlsx
```

Use `--max-files` to limit processed entries. The `--video-backend` flag defaults to `ffmpeg` (NVDEC via `ffmpeg -hwaccel cuda`) and can be switched to `decord` if you install the CUDA wheel manually.
