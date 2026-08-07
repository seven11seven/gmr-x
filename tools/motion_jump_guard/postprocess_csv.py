#!/usr/bin/env python3
"""Post-process retargeted motion CSVs with a velocity jump guard."""

import argparse
import pathlib
import sys

import numpy as np


QUAT_COL_NAMES = {"base_qx", "base_qy", "base_qz", "base_qw"}


def load_csv(path: pathlib.Path) -> tuple[list[str], np.ndarray]:
    with open(path, "r", encoding="utf-8") as fh:
        header = fh.readline().strip().split(",")
    data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] != len(header):
        raise ValueError(
            f"Header has {len(header)} columns, but data has {data.shape[1]} columns."
        )
    return header, data


def resolve_columns(header: list[str], mode: str) -> np.ndarray:
    if mode == "joints":
        if len(header) <= 7:
            raise ValueError("Cannot use --columns joints: CSV has no columns after base.")
        return np.arange(7, len(header), dtype=int)
    if mode == "scalar":
        return np.asarray(
            [i for i, name in enumerate(header) if name not in QUAT_COL_NAMES],
            dtype=int,
        )
    if mode == "all":
        return np.arange(len(header), dtype=int)
    raise ValueError(f"Unsupported column mode: {mode}")


def quaternion_columns(header: list[str]) -> np.ndarray:
    indices = [header.index(name) for name in ("base_qx", "base_qy", "base_qz", "base_qw") if name in header]
    if len(indices) == 4:
        return np.asarray(indices, dtype=int)
    return np.asarray([], dtype=int)


def normalize_quaternions(data: np.ndarray, quat_cols: np.ndarray) -> None:
    if quat_cols.size == 0:
        return
    quat = data[:, quat_cols]
    norms = np.linalg.norm(quat, axis=1)
    valid = norms > 1e-12
    quat[valid] /= norms[valid, None]
    data[:, quat_cols] = quat


def centered_window_mean(values: np.ndarray, left: int, right: int) -> np.ndarray:
    if values.shape[0] == 0:
        return values.copy()
    padded = np.pad(values, ((left, right), (0, 0)), mode="edge")
    csum = np.vstack([np.zeros((1, values.shape[1])), np.cumsum(padded, axis=0)])
    window = left + right + 1
    return (csum[window:] - csum[:-window]) / window


def detect_jump_frames(
    data: np.ndarray,
    cols: np.ndarray,
    fps: float,
    left: int,
    right: int,
    threshold: float,
    min_mean_velocity: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    velocities = np.abs(np.diff(data[:, cols], axis=0)) * fps
    mean_velocities = centered_window_mean(velocities, left, right)
    ratio = velocities / np.maximum(mean_velocities, min_mean_velocity)
    transition_flags = np.any(ratio > threshold, axis=1)

    frame_flags = np.zeros(data.shape[0], dtype=bool)
    if data.shape[0] > 2:
        frame_flags[1:-1] = transition_flags[:-1]
    return frame_flags, velocities, mean_velocities


def bridge_intervals(
    data: np.ndarray,
    cols: np.ndarray,
    start_idx: int,
    next_idx: int,
    target_velocity: np.ndarray,
    fps: float,
    max_insert_frames: int,
    min_mean_velocity: float,
) -> int:
    original_intervals = next_idx - start_idx
    per_frame_delta = np.maximum(target_velocity, min_mean_velocity) / fps
    required = np.abs(data[next_idx, cols] - data[start_idx, cols]) / per_frame_delta
    intervals = int(np.ceil(np.max(required))) if required.size else original_intervals
    intervals = max(original_intervals, intervals)
    return min(intervals, original_intervals + max_insert_frames)


def interpolate_segment(start: np.ndarray, end: np.ndarray, intervals: int) -> np.ndarray:
    alphas = np.linspace(1.0 / intervals, 1.0, intervals)
    return start[None, :] + alphas[:, None] * (end - start)[None, :]


def insert_jump_bridges(
    data: np.ndarray,
    cols: np.ndarray,
    frame_flags: np.ndarray,
    mean_velocities: np.ndarray,
    fps: float,
    max_insert_frames: int,
    min_mean_velocity: float,
) -> tuple[np.ndarray, int, int, list[tuple[int, int]]]:
    if not np.any(frame_flags):
        return data.copy(), 0, 0, []

    output = [data[0].copy()]
    bridge_ranges = []
    inserted = 0
    groups = 0
    i = 1
    while i < data.shape[0]:
        if not frame_flags[i]:
            output.append(data[i].copy())
            i += 1
            continue

        group_start = i
        while i < data.shape[0] - 1 and frame_flags[i]:
            i += 1
        group_end = i - 1
        prev_idx = group_start - 1
        next_idx = min(group_end + 1, data.shape[0] - 1)

        velocity_start = max(group_start - 1, 0)
        velocity_end = max(group_end, velocity_start)
        target_velocity = np.mean(mean_velocities[velocity_start : velocity_end + 1], axis=0)
        intervals = bridge_intervals(
            data,
            cols,
            prev_idx,
            next_idx,
            target_velocity,
            fps,
            max_insert_frames,
            min_mean_velocity,
        )

        bridge = interpolate_segment(data[prev_idx], data[next_idx], intervals)
        output_start = len(output)
        output.extend(bridge)
        output_end = len(output)
        bridge_ranges.append((output_start, output_end))
        inserted += max(0, intervals - (next_idx - prev_idx))
        groups += 1
        i = next_idx + 1

    return np.vstack(output), inserted, groups, bridge_ranges


def moving_average_filter(values: np.ndarray, radius: int) -> np.ndarray:
    padded = np.pad(values, ((radius, radius), (0, 0)), mode="edge")
    csum = np.vstack([np.zeros((1, values.shape[1])), np.cumsum(padded, axis=0)])
    window = 2 * radius + 1
    return (csum[window:] - csum[:-window]) / window


def gaussian_filter(values: np.ndarray, radius: int, sigma: float | None) -> np.ndarray:
    sigma = sigma if sigma is not None else max(radius / 2.0, 1e-6)
    xs = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (xs / sigma) ** 2)
    kernel /= np.sum(kernel)
    padded = np.pad(values, ((radius, radius), (0, 0)), mode="edge")
    filtered = np.empty_like(values)
    for col_idx in range(values.shape[1]):
        filtered[:, col_idx] = np.convolve(padded[:, col_idx], kernel, mode="valid")
    return filtered


def savgol_filter(values: np.ndarray, radius: int, polyorder: int) -> np.ndarray:
    from scipy.signal import savgol_filter as scipy_savgol_filter

    window = 2 * radius + 1
    if values.shape[0] < window:
        return values.copy()
    polyorder = min(polyorder, window - 1)
    return scipy_savgol_filter(
        values,
        window_length=window,
        polyorder=polyorder,
        axis=0,
        mode="interp",
    )


def filter_columns(
    data: np.ndarray,
    cols: np.ndarray,
    radius: int,
    method: str,
    gaussian_sigma: float | None,
    savgol_polyorder: int,
) -> np.ndarray:
    values = data[:, cols]
    if method == "moving_average":
        return moving_average_filter(values, radius)
    if method == "gaussian":
        return gaussian_filter(values, radius, gaussian_sigma)
    if method == "savgol":
        return savgol_filter(values, radius, savgol_polyorder)
    raise ValueError(f"Unsupported smoothing method: {method}")


def smooth_columns(
    data: np.ndarray,
    cols: np.ndarray,
    radius: int,
    bridge_ranges: list[tuple[int, int]],
    padding: int,
    method: str,
    gaussian_sigma: float | None,
    savgol_polyorder: int,
) -> tuple[np.ndarray, int]:
    if radius <= 0 or data.shape[0] <= 2 or not bridge_ranges:
        return data, 0

    mask = np.zeros(data.shape[0], dtype=bool)
    for start, end in bridge_ranges:
        smooth_start = max(0, start - padding)
        smooth_end = min(data.shape[0], end + padding)
        mask[smooth_start:smooth_end] = True

    smoothed = data.copy()
    filtered = filter_columns(
        data,
        cols,
        radius,
        method,
        gaussian_sigma,
        savgol_polyorder,
    )
    rows = np.where(mask)[0]
    smoothed[np.ix_(rows, cols)] = filtered[rows]
    smoothed[0, cols] = data[0, cols]
    smoothed[-1, cols] = data[-1, cols]
    return smoothed, int(np.sum(mask))


def default_output_path(input_path: pathlib.Path) -> pathlib.Path:
    return input_path.parent / f"{input_path.stem}_jump_guard.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect high frame-to-frame velocity jumps and insert bridge frames."
    )
    parser.add_argument("input", type=pathlib.Path, help="Input retargeted motion CSV.")
    parser.add_argument("-o", "--output", type=pathlib.Path, default=None)
    parser.add_argument("--fps", type=float, default=120.0)
    parser.add_argument("--left", type=int, default=20, help="Left smoothing window in frames.")
    parser.add_argument("--right", type=int, default=20, help="Right smoothing window in frames.")
    parser.add_argument("--threshold", type=float, default=5.0)
    parser.add_argument(
        "--columns",
        choices=("joints", "scalar", "all"),
        default="joints",
        help="Columns used for jump detection and smoothing.",
    )
    parser.add_argument(
        "--min-mean-velocity",
        type=float,
        default=1e-3,
        help="Floor for the window velocity mean to avoid division by zero.",
    )
    parser.add_argument(
        "--max-insert-frames",
        type=int,
        default=120,
        help="Maximum additional frames inserted for one jump group.",
    )
    parser.add_argument(
        "--smooth-radius",
        type=int,
        default=0,
        help="Optional centered smoothing radius after insertion. 0 disables it.",
    )
    parser.add_argument(
        "--smooth-padding",
        type=int,
        default=10,
        help="Extra frames before/after inserted bridge ranges that get smoothed.",
    )
    parser.add_argument(
        "--smooth-method",
        choices=("moving_average", "gaussian", "savgol"),
        default="moving_average",
    )
    parser.add_argument(
        "--gaussian-sigma",
        type=float,
        default=None,
        help="Sigma for --smooth-method gaussian. Defaults to smooth_radius / 2.",
    )
    parser.add_argument(
        "--savgol-polyorder",
        type=int,
        default=3,
        help="Polynomial order for --smooth-method savgol.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = args.input.resolve()
    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}", file=sys.stderr)
        return 1
    if args.fps <= 0:
        print("ERROR: --fps must be positive.", file=sys.stderr)
        return 1
    if args.left < 0 or args.right < 0 or args.smooth_radius < 0 or args.smooth_padding < 0:
        print("ERROR: window sizes must be non-negative.", file=sys.stderr)
        return 1
    if args.threshold <= 0:
        print("ERROR: --threshold must be positive.", file=sys.stderr)
        return 1
    if args.gaussian_sigma is not None and args.gaussian_sigma <= 0:
        print("ERROR: --gaussian-sigma must be positive.", file=sys.stderr)
        return 1
    if args.savgol_polyorder < 0:
        print("ERROR: --savgol-polyorder must be non-negative.", file=sys.stderr)
        return 1

    output_path = args.output.resolve() if args.output else default_output_path(input_path)
    header, data = load_csv(input_path)
    cols = resolve_columns(header, args.columns)
    quat_cols = quaternion_columns(header)

    frame_flags, velocities, mean_velocities = detect_jump_frames(
        data,
        cols,
        args.fps,
        args.left,
        args.right,
        args.threshold,
        args.min_mean_velocity,
    )
    processed, inserted, groups, bridge_ranges = insert_jump_bridges(
        data,
        cols,
        frame_flags,
        mean_velocities,
        args.fps,
        args.max_insert_frames,
        args.min_mean_velocity,
    )

    smoothed_frames = 0
    if args.smooth_radius > 0:
        processed, smoothed_frames = smooth_columns(
            processed,
            cols,
            args.smooth_radius,
            bridge_ranges,
            args.smooth_padding,
            args.smooth_method,
            args.gaussian_sigma,
            args.savgol_polyorder,
        )
    normalize_quaternions(processed, quat_cols)

    flagged_frames = int(np.sum(frame_flags))
    print(f"Input : {input_path}")
    print(f"Frames: {data.shape[0]} -> {processed.shape[0]} (+{processed.shape[0] - data.shape[0]})")
    print(f"Jump frames: {flagged_frames}, groups: {groups}, inserted frames: {inserted}")
    print(f"Columns mode: {args.columns} ({len(cols)} columns), fps: {args.fps:g}")
    print(f"Threshold: {args.threshold:g}, velocity window: left={args.left}, right={args.right}")
    if args.smooth_radius > 0:
        print(
            f"Smoothing: {args.smooth_method}, radius={args.smooth_radius}, "
            f"padding={args.smooth_padding}, affected frames={smoothed_frames}"
        )
    if velocities.size and flagged_frames:
        ratios = velocities / np.maximum(mean_velocities, args.min_mean_velocity)
        worst = float(np.max(ratios))
        print(f"Worst instant/window velocity ratio: {worst:.3f}")

    if args.dry_run:
        print(f"Dry run: would write {output_path}")
        return 0

    np.savetxt(
        output_path,
        processed,
        delimiter=",",
        fmt="%.8f",
        header=",".join(header),
        comments="",
    )
    print(f"Output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
