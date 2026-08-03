#!/usr/bin/env python3
"""
Strip wrist DOF columns from a D20 29-DOF motion CSV (or PKL).

The D20 CSV has 36 columns::

    0-2   base_x, base_y, base_z
    3-6   base_qx, base_qy, base_qz, base_qw      (xyzw)
    7-9   waist_yaw, waist_roll, waist_pitch
    10-13 L_shoulder_pitch, roll, yaw, elbow_pitch
    14-16 L_wrist_roll, pitch, yaw                  ← removed
    17-20 R_shoulder_pitch, roll, yaw, elbow_pitch
    21-23 R_wrist_roll, pitch, yaw                  ← removed
    24-29 L_hip_pitch, roll, yaw, knee, ankle_pitch, ankle_roll
    30-35 R_hip_pitch, roll, yaw, knee, ankle_pitch, ankle_roll

After stripping the 6 wrist columns, 30 columns remain (23 DOF + base).

Usage::

    python tools/csv_mujoco_viewer/strip_wrist.py motion.csv
    python tools/csv_mujoco_viewer/strip_wrist.py motion.csv -o nowrist.csv
    python tools/csv_mujoco_viewer/strip_wrist.py motion.pkl -o nowrist.csv
"""

import argparse
import pickle
import pathlib
import sys

import numpy as np

# 0-based indices of wrist columns to remove
WRIST_COLS = [15, 16,   # L_wrist pitch, yaw
              22, 23]   # R_wrist pitch, yaw

HEADER_36 = [
    "base_x", "base_y", "base_z",
    "base_qx", "base_qy", "base_qz", "base_qw",
    "waist_yaw", "waist_roll", "waist_pitch",
    "L_shoulder_pitch", "L_shoulder_roll", "L_shoulder_yaw", "L_elbow_pitch",
    "L_wrist_roll", "L_wrist_pitch", "L_wrist_yaw",
    "R_shoulder_pitch", "R_shoulder_roll", "R_shoulder_yaw", "R_elbow_pitch",
    "R_wrist_roll", "R_wrist_pitch", "R_wrist_yaw",
    "L_hip_pitch", "L_hip_roll", "L_hip_yaw",
    "L_knee_pitch", "L_ankle_pitch", "L_ankle_roll",
    "R_hip_pitch", "R_hip_roll", "R_hip_yaw",
    "R_knee_pitch", "R_ankle_pitch", "R_ankle_roll",
]


def _is_pickle(path: pathlib.Path) -> bool:
    with open(path, "rb") as fh:
        head = fh.read(2)
    return len(head) == 2 and head[0] == 0x80 and 0x02 <= head[1] <= 0x05


def load_data(path: pathlib.Path) -> np.ndarray:
    """Return (N, 36) array from CSV or PKL."""
    if _is_pickle(path):
        print("  (detected pickle format)")
        with open(path, "rb") as fh:
            cache = pickle.load(fh)
        rows = []
        for fd in cache["all_frame_data"]:
            qpos = fd["qpos"]
            base_quat_xyzw = np.roll(qpos[3:7], -1)
            rows.append(np.concatenate([qpos[0:3], base_quat_xyzw, qpos[7:]]))
        return np.stack(rows, axis=0)

    data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def strip_wrist(data: np.ndarray) -> np.ndarray:
    """Remove wrist columns.  Input (N, 36) → output (N, 30)."""
    assert data.shape[1] == 36, f"Expected 36 columns, got {data.shape[1]}"
    keep = [i for i in range(36) if i not in WRIST_COLS]
    return data[:, keep]


def main():
    ap = argparse.ArgumentParser(
        description="Strip wrist DOF columns from a D20 29-DOF motion CSV/PKL")
    ap.add_argument("input", type=str, help="Input CSV or PKL file")
    ap.add_argument("-o", "--output", type=str, default=None,
                    help="Output CSV path (default: <input>_nowrist.csv)")
    args = ap.parse_args()

    in_path = pathlib.Path(args.input).resolve()
    if not in_path.exists():
        print(f"ERROR: file not found: {in_path}")
        sys.exit(1)

    out_path = pathlib.Path(args.output) if args.output else \
        in_path.parent / f"{in_path.stem}_nowrist.csv"

    print(f"Input : {in_path}")
    data = load_data(in_path)
    print(f"  Loaded {data.shape[0]} frames × {data.shape[1]} cols")

    stripped = strip_wrist(data)
    print(f"  Stripped → {stripped.shape[0]} frames × {stripped.shape[1]} cols")

    # Build header
    keep = [i for i in range(36) if i not in WRIST_COLS]
    out_header = ",".join(HEADER_36[i] for i in keep)

    np.savetxt(out_path, stripped, delimiter=",", fmt="%.8f",
               header=out_header, comments="")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
