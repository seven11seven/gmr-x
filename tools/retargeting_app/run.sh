#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ -n "${PYTHON:-}" ]]; then
  read -r -a PYTHON_CMD <<< "${PYTHON}"
elif command -v conda >/dev/null 2>&1 && conda env list | grep -qE '^gmr\s'; then
  GMR_PY="$(conda run -n gmr python -c 'import sys; print(sys.executable)' 2>/dev/null || true)"
  if [[ -n "${GMR_PY}" && -x "${GMR_PY}" ]]; then
    PYTHON_CMD=("${GMR_PY}")
  else
    PYTHON_CMD=(conda run --no-capture-output -n gmr python)
  fi
else
  PYTHON_CMD=(python)
fi

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8765}"

echo "Starting GMR Stage-1 app at http://${HOST}:${PORT}"
exec "${PYTHON_CMD[@]}" "${SCRIPT_DIR}/server.py" --host "${HOST}" --port "${PORT}" "$@"
