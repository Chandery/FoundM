#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

echo "[init] repo root: $ROOT"
if conda run -n curia-t2 python -V >/dev/null 2>&1; then
  CONDA_RUN=(conda run -n curia-t2)
  echo "[init] using conda env: curia-t2"
elif [ -d "$ROOT/.conda/curia-t2" ]; then
  CONDA_RUN=(conda run -p "$ROOT/.conda/curia-t2")
  echo "[init] using conda env prefix: $ROOT/.conda/curia-t2"
else
  echo "[init] missing conda env 'curia-t2' and missing fallback prefix"
  exit 1
fi

# Filesystem checks
test -d "$ROOT/curia_t2"
test -f "$ROOT/curia_t2/train_curia_task2.py"
test -f "$ROOT/curia_t2/predict_curia_task2.py"
test -f "$ROOT/curia_t2/download_curia.py"

echo "[init] python info (from conda env)"
"${CONDA_RUN[@]}" python -V

echo "[init] cli help smoke tests"
"${CONDA_RUN[@]}" python -m py_compile curia_t2/*.py
"${CONDA_RUN[@]}" python curia_t2/train_curia_task2.py --help >/dev/null
"${CONDA_RUN[@]}" python curia_t2/predict_curia_task2.py --help >/dev/null
"${CONDA_RUN[@]}" python curia_t2/download_curia.py --help >/dev/null

echo "[init] done"
