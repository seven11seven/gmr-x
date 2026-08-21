#!/usr/bin/env bash
# One-click launcher for the GMR retargeting web app.
#
# Usage (from anywhere):
#   bash /path/to/GMR/start_web_app.sh
#   ./start_web_app.sh                  # if cwd is GMR/
#
# Optional env overrides:
#   HOST=0.0.0.0 PORT=8765 PYTHON=/path/to/python ./start_web_app.sh
#   ./start_web_app.sh --debug          # extra args forwarded to server.py

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

SERVER="${ROOT}/tools/retargeting_app/server.py"
if [[ ! -f "${SERVER}" ]]; then
  echo "error: cannot find ${SERVER}" >&2
  exit 1
fi

# Resolve Python: PYTHON env → conda env "gmr" → python3/python
if [[ -n "${PYTHON:-}" ]]; then
  read -r -a PYTHON_CMD <<< "${PYTHON}"
elif command -v conda >/dev/null 2>&1 && conda env list 2>/dev/null | grep -qE '^gmr[[:space:#]]'; then
  GMR_PY="$(conda run -n gmr python -c 'import sys; print(sys.executable)' 2>/dev/null || true)"
  if [[ -n "${GMR_PY}" && -x "${GMR_PY}" ]]; then
    PYTHON_CMD=("${GMR_PY}")
  else
    PYTHON_CMD=(conda run --no-capture-output -n gmr python)
  fi
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_CMD=(python3)
else
  PYTHON_CMD=(python)
fi

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8765}"

# Warn if port is already in use
if command -v ss >/dev/null 2>&1; then
  if ss -tln 2>/dev/null | grep -qE ":${PORT}\\b"; then
    echo "warning: port ${PORT} appears to be in use." >&2
    echo "  Stop the other process, or run with: PORT=8766 $0" >&2
  fi
fi

echo "GMR retargeting web app"
echo "  project : ${ROOT}"
echo "  python  : ${PYTHON_CMD[*]}"
echo "  url     : http://${HOST}:${PORT}"
echo "  Stage-1 : http://${HOST}:${PORT}/"
echo "  Stage-2 : http://${HOST}:${PORT}/stage2"
echo "  Stage-3 : http://${HOST}:${PORT}/stage3  (Aux)"
echo "  Stage-4 : http://${HOST}:${PORT}/stage4  (Weights)"
echo "  Stage-5 : http://${HOST}:${PORT}/stage5  (Retarget)"
echo "  (Ctrl+C to stop)"
echo

exec "${PYTHON_CMD[@]}" "${SERVER}" --host "${HOST}" --port "${PORT}" "$@"
