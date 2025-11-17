#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SETUP_SCRIPT="${ROOT}/scripts/wsl/setup_env.sh"
VENV_DIR="${ROOT}/.venv_wsl"
DEFAULT_BASE="/mnt/d/RAPA"
DEFAULT_OUTPUT="${DEFAULT_BASE}/rapa_report_full_wsl.xlsx"
DEFAULT_VIDEO_BACKEND="ffmpeg"

usage() {
    cat <<'EOF'
Usage: run_pipeline.sh [--base-dir PATH] [--output PATH] [--max-files N] [--video-backend NAME] [--log-level LEVEL] [-- ...extra CLI args]

Runs python -m face_deid_emotion_metrics.cli inside the dedicated WSL virtualenv.
EOF
}

require_setup() {
    if [[ ! -x "${SETUP_SCRIPT}" ]]; then
        echo "Missing setup script at ${SETUP_SCRIPT}" >&2
        exit 1
    fi
    if [[ ! -d "${VENV_DIR}" || ! -f "${VENV_DIR}/bin/activate" ]]; then
        "${SETUP_SCRIPT}"
    fi
}

parse_args() {
    BASE_DIR="${DEFAULT_BASE}"
    OUTPUT_PATH="${DEFAULT_OUTPUT}"
    MAX_FILES=""
    VIDEO_BACKEND="${DEFAULT_VIDEO_BACKEND}"
    LOG_LEVEL=""
    EXTRA_ARGS=()
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --base-dir)
                BASE_DIR="$2"
                shift 2
                ;;
            --output)
                OUTPUT_PATH="$2"
                shift 2
                ;;
            --max-files)
                MAX_FILES="$2"
                shift 2
                ;;
            --video-backend)
                VIDEO_BACKEND="$2"
                shift 2
                ;;
            --log-level)
                LOG_LEVEL="$2"
                shift 2
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            --)
                shift
                EXTRA_ARGS+=("$@")
                break
                ;;
            *)
                EXTRA_ARGS+=("$1")
                shift
                ;;
        esac
    done
}

run_pipeline() {
    if [[ ! -d "${VENV_DIR}" ]]; then
        echo "Virtualenv not found at ${VENV_DIR}. Run setup_env.sh first." >&2
        exit 1
    fi
    # shellcheck disable=SC1090
    source "${VENV_DIR}/bin/activate"
    if [[ ! -d "${BASE_DIR}" ]]; then
        echo "Base directory not found: ${BASE_DIR}" >&2
        exit 1
    fi
    mkdir -p "$(dirname "${OUTPUT_PATH}")"
    cmd=(python -m face_deid_emotion_metrics.cli --base-dir "${BASE_DIR}" --output "${OUTPUT_PATH}")
    if [[ -n "${MAX_FILES}" ]]; then
        cmd+=(--max-files "${MAX_FILES}")
    fi
    if [[ -n "${VIDEO_BACKEND}" ]]; then
        cmd+=(--video-backend "${VIDEO_BACKEND}")
    fi
    if [[ -n "${LOG_LEVEL}" ]]; then
        cmd+=(--log-level "${LOG_LEVEL}")
    fi
    if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
        cmd+=("${EXTRA_ARGS[@]}")
    fi
    echo "[run $(date +%H:%M:%S)] Launching pipeline with base=${BASE_DIR} output=${OUTPUT_PATH}"
    local start_ts end_ts duration
    start_ts=$(date +%s)
    "${cmd[@]}"
    end_ts=$(date +%s)
    duration=$(( end_ts - start_ts ))
    echo "[run $(date +%H:%M:%S)] Pipeline finished in ${duration}s"
}

parse_args "$@"
require_setup
run_pipeline
