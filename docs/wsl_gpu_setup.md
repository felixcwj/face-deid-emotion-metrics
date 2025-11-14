# WSL2 Ubuntu GPU Workflow

## 1. Verify installed distros

```powershell
wsl -l -v
```

Ensure an Ubuntu distro is listed with version 2. Use that name in later `wsl -d <distro>` commands.

## 2. Enter Ubuntu from Windows PowerShell

```powershell
wsl -d Ubuntu
```

Swap `Ubuntu` with your distro name if different.

## 3. Optional GPU probe

Inside Ubuntu, you can confirm GPU visibility via:

```bash
sudo /usr/lib/wsl/lib/nvidia-smi
```

(Some installs expose `nvidia-smi` directly on the PATH.)

## 4. Create the project environment

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip python3-dev build-essential git

cd /mnt/c/projects/face-deid-emotion-metrics

python3 -m venv .venv_wsl
source .venv_wsl/bin/activate

export PIP_BREAK_SYSTEM_PACKAGES=1
python -m pip install --upgrade pip

pip install "torch==2.5.1" "torchvision==0.20.1" --index-url https://download.pytorch.org/whl/cu121
pip install "tensorflow[and-cuda]"

pip install -r requirements.txt
pip install -e .
```

Notes:
- `python --version` inside `.venv_wsl` typically reports Python 3.12.x.
- Keeping `PIP_BREAK_SYSTEM_PACKAGES=1` avoids conflicts with system packages.

## 5. Validate GPU availability

### PyTorch

```bash
python - <<'PY'
import torch
print(torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device 0:", torch.cuda.get_device_name(0))
PY
```

### TensorFlow

```bash
python - <<'PY'
import tensorflow as tf
print(tf.__version__)
gpus = tf.config.list_physical_devices("GPU")
print("TF GPUs:", gpus)
if not gpus:
    raise SystemExit("TensorFlow does not see a GPU")
PY
```

## 6. Run the full pipeline

```bash
cd /mnt/c/projects/face-deid-emotion-metrics
source .venv_wsl/bin/activate
export PIP_BREAK_SYSTEM_PACKAGES=1

python -m face_deid_emotion_metrics.cli \
  --base-dir "/mnt/d/RAPA" \
  --output "/mnt/d/RAPA/rapa_report_full_wsl.xlsx"
```

`/mnt/d/RAPA` corresponds to `D:\RAPA`. The CLI prints `Using device: cuda:0`, shows a tqdm progress bar, and writes `rapa_report_full_wsl.xlsx` with columns:

1. filename
2. FaceNet (%)
3. LPIPS (%)
4. Final score (%)
5. FER emotion (%)
6. DeepFace emotion (%)
7. Person count
8. Duration

Use `--max-files N` for small debug runs; omit it for production-scale processing.
