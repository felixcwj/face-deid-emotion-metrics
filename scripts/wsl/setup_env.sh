#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${ROOT}/.venv_wsl"
REQ_FILE="${ROOT}/requirements.txt"
STAMP_FILE="${VENV_DIR}/.requirements.sha1"
PYTHON_BIN="${VENV_DIR}/bin/python"
PIP_BIN="${VENV_DIR}/bin/pip"

log() {
    echo "[setup $(date +%H:%M:%S)] $*"
}

ensure_apt_packages() {
    if command -v apt-get >/dev/null 2>&1; then
        log "Installing system dependencies via apt-get (requires sudo)..."
        sudo apt-get update
        sudo apt-get install -y python3-venv python3-pip build-essential ffmpeg libgl1 libglib2.0-0
    else
        log "Skipping apt-get install (apt-get not available)."
    fi
}

ensure_venv() {
    if [[ -d "${VENV_DIR}" ]]; then
        return
    fi
    log "Creating WSL virtualenv at ${VENV_DIR}"
    python3 -m venv "${VENV_DIR}"
}

activate_venv() {
    # shellcheck disable=SC1090
    source "${VENV_DIR}/bin/activate"
}

install_python_dependencies() {
    activate_venv
    "${PYTHON_BIN}" -m pip install --upgrade pip setuptools wheel
    local new_hash
    new_hash="$(sha1sum "${REQ_FILE}" | awk '{print $1}')"
    local current_hash="missing"
    if [[ -f "${STAMP_FILE}" ]]; then
        current_hash="$(cat "${STAMP_FILE}")"
    fi
    if [[ "${current_hash}" != "${new_hash}" ]]; then
        log "Installing Python requirements (hash ${new_hash})"
        "${PIP_BIN}" install -r "${REQ_FILE}"
        "${PIP_BIN}" install -e "${ROOT}"
        echo "${new_hash}" >"${STAMP_FILE}"
    else
        log "Python requirements already installed (hash ${new_hash})"
    fi
}

detect_cuda_tag() {
    "${PYTHON_BIN}" - <<'PY' 2>/dev/null || true
import torch
version = getattr(torch.version, "cuda", None)
if not version:
    raise SystemExit(1)
digits = version.replace(".", "")
print(digits)
PY
}

ensure_decord_gpu() {
    activate_venv
    if "${PYTHON_BIN}" - <<'PY'; then
import decord
from decord import gpu
vr = decord.VideoReader  # noqa: F401
ctx = gpu(0)
del ctx
PY
        log "Decord GPU backend already available."
        return
    fi
    local cuda_tag
    cuda_tag="$(detect_cuda_tag)"
    if [[ -z "${cuda_tag}" ]]; then
        log "Unable to detect CUDA version from PyTorch; leaving decord as-is."
        return
    fi
    local package="decord-cu${cuda_tag}"
    log "Installing ${package} from https://mlc.ai/wheels ..."
    if ! "${PIP_BIN}" install --extra-index-url https://mlc.ai/wheels "${package}"; then
        log "Failed to install ${package}. Refer to docs/wsl.md for manual GPU decord setup."
    fi
}

main() {
    log "WSL GPU environment bootstrap starting..."
    ensure_apt_packages
    ensure_venv
    install_python_dependencies
    ensure_decord_gpu
    log "WSL GPU environment ready."
}

main "$@"
