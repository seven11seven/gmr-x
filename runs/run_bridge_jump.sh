#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -n "${PYTHON:-}" ]]; then
  read -r -a PYTHON_CMD <<< "${PYTHON}"
else
  PYTHON_CMD=(conda run -n gmr python)
fi

"${PYTHON_CMD[@]}" "${PROJECT_ROOT}/tools/motion_jump_guard/postprocess_csv.py" \
  "${PROJECT_ROOT}/outputs/d20_25dof/fz_dance_13.csv" \
  --fps 120 \
  --left 10 \
  --right 10 \
  --threshold 6 \
  --columns joints \
  --max-insert-frames 120 \
  --smooth-radius 10 \
  --smooth-padding 10 \
  --smooth-method gaussian \
  -o "${PROJECT_ROOT}/outputs/d20_25dof/fz_dance_13_guard.csv"
