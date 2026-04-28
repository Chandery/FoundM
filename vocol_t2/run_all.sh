#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${ROOT_DIR}/.." && pwd)"

CONFIG_PATH="${1:-${ROOT_DIR}/configs/covid.yaml}"
CONDA_ENV="${2:-curia-t2}"
START_FROM="${3:-extract}"  # extract | train | predict | eval

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[ERROR] Config not found: ${CONFIG_PATH}"
  exit 1
fi

run_step() {
  local step="$1"
  shift
  echo ""
  echo "========== ${step} =========="
  echo "Command: $*"
  "$@"
}

should_run() {
  local step="$1"
  case "${START_FROM}" in
    extract) return 0 ;;
    train) [[ "${step}" != "extract" ]] ;;
    predict) [[ "${step}" == "predict" || "${step}" == "eval" ]] ;;
    eval) [[ "${step}" == "eval" ]] ;;
    *)
      echo "[ERROR] Invalid START_FROM=${START_FROM}, expected: extract|train|predict|eval"
      exit 1
      ;;
  esac
}

cd "${PROJECT_DIR}"

if should_run extract; then
  run_step "extract" \
     python "${ROOT_DIR}/extract_features.py" --config "${CONFIG_PATH}"
fi

if should_run train; then
  run_step "train" \
     python "${ROOT_DIR}/train_head_lightning.py" --config "${CONFIG_PATH}"
fi

if should_run predict; then
  run_step "predict" \
     python "${ROOT_DIR}/predict_head.py" --config "${CONFIG_PATH}"
fi

if should_run eval; then
  run_step "eval" \
     python "${ROOT_DIR}/eval_predictions.py" --config "${CONFIG_PATH}"
fi

echo ""
echo "Done."
echo "Config: ${CONFIG_PATH}"
echo "Env: ${CONDA_ENV}"
echo "Start from: ${START_FROM}"
