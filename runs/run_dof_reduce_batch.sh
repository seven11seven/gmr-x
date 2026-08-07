#!/bin/bash
# Batch DOF reduction: strip wrist columns from all CSV files in a folder.
#
# Usage:
#   bash runs/run_dof_reduce_batch.sh <src_dir> <dst_dir>
#
# Example:
#   bash runs/run_dof_reduce_batch.sh \
#       outputs/d20_29dof \
#       outputs/d20_25dof

set -euo pipefail

SRC_DIR="${1:?Usage: $0 <src_dir> <dst_dir>}"
DST_DIR="${2:?Usage: $0 <src_dir> <dst_dir>}"

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STRIP_PY="${PROJECT_ROOT}/tools/csv_mujoco_viewer/strip_wrist.py"

if [ ! -d "${PROJECT_ROOT}/${SRC_DIR}" ]; then
    echo "ERROR: source directory not found: ${PROJECT_ROOT}/${SRC_DIR}"
    exit 1
fi

mkdir -p "${PROJECT_ROOT}/${DST_DIR}"

echo "============================================"
echo "Batch DOF Reduction (29dof → 25dof)"
echo "============================================"
echo "  Source : ${SRC_DIR}"
echo "  Target : ${DST_DIR}"
echo "============================================"

count=0
failed=0

for csv_file in "${PROJECT_ROOT}/${SRC_DIR}"/*.csv; do
    [ -e "$csv_file" ] || continue

    fname=$(basename "$csv_file")
    out_file="${PROJECT_ROOT}/${DST_DIR}/${fname}"

    echo ""
    echo "[$((count + 1))] ${fname}"
    if python "${STRIP_PY}" "$csv_file" -o "$out_file"; then
        count=$((count + 1))
    else
        echo "  FAILED: ${fname}"
        failed=$((failed + 1))
    fi
done

echo ""
echo "============================================"
echo "Done: ${count} succeeded, ${failed} failed"
echo "Output in: ${DST_DIR}"
echo "============================================"
