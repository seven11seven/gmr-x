#!/usr/bin/env python3
"""
Change the playback speed of a segment in a robot-motion CSV file.

The CSV must contain per-frame robot poses: base position (x, y, z),
base orientation quaternion (qx, qy, qz, qw), and joint positions.

Algorithm
---------
The script defines a piecewise-linear time-warp function that maps each
new-frame timestamp to an original-frame timestamp.  Inside the chosen
segment, time runs *slower* (factor < 1) or *faster* (factor > 1); outside
it, time is unchanged.  Each new frame is then computed by interpolating
the two nearest original frames:

* **positions / joint angles** – linear interpolation (lerp)
* **quaternions** – spherical linear interpolation (slerp)

Examples
--------
.. code-block:: bash

    # Slow down frames 100–200 to half speed (2× duration)
    python tools/motion_speed_warp/warp_speed.py \\
        outputs/d20_25dof/dance_1.csv \\
        --fps 120 --start-frame 100 --end-frame 200 --factor 0.5 \\
        -o outputs/d20_25dof/dance_1_slow.csv

    # Double the speed of the first 2 seconds
    python tools/motion_speed_warp/warp_speed.py \\
        outputs/d20_25dof/dance_1.csv \\
        --fps 120 --start-time 0.0 --end-time 2.0 --factor 2.0 \\
        -o outputs/d20_25dof/dance_1_fast.csv

    # Slow entire motion to 0.8× (omit --start/--end to warp everything)
    python tools/motion_speed_warp/warp_speed.py \\
        outputs/d20_25dof/dance_1.csv \\
        --fps 120 --factor 0.8 \\
        -o outputs/d20_25dof/dance_1_slow80.csv
"""

import argparse
import csv
import pathlib
import sys

import numpy as np

# ---------------------------------------------------------------------------
#  Quaternion helpers
# ---------------------------------------------------------------------------

_QUAT_SUFFIXES = ("_qx", "_qy", "_qz", "_qw")


def _find_quaternion_groups(columns: list[str]) -> dict[str, list[int]]:
    """Return ``{prefix: [idx_qx, idx_qy, idx_qz, idx_qw]}`` for every
    complete quaternion group found in *columns*.

    The indices refer to positions in the column list so that we can
    index into a raw row (list of floats) directly.
    """
    groups: dict[str, dict[str, int]] = {}
    for idx, col in enumerate(columns):
        for sfx in _QUAT_SUFFIXES:
            if col.endswith(sfx):
                prefix = col[: -len(sfx)]
                comp = sfx[1:]  # 'qx', 'qy', 'qz', 'qw'
                groups.setdefault(prefix, {})[comp] = idx
                break
    return {
        prefix: [d["qx"], d["qy"], d["qz"], d["qw"]]
        for prefix, d in groups.items()
        if len(d) == 4
    }


def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two quaternions (xyzw)."""
    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)

    dot = float(np.dot(q0, q1))
    # Take the shorter arc
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    DOT_THRESHOLD = 0.9995
    if dot > DOT_THRESHOLD:
        result = q0 + t * (q1 - q0)
        norm = float(np.linalg.norm(result))
        if norm < 1e-12:
            return q0.copy()
        return result / norm

    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * t

    s0 = np.cos(theta) - dot * np.sin(theta) / sin_theta_0
    s1 = np.sin(theta) / sin_theta_0
    return s0 * q0 + s1 * q1


# ---------------------------------------------------------------------------
#  Time-warp core
# ---------------------------------------------------------------------------


def _build_time_map(
    t_start: float,
    t_end: float,
    factor: float,
):
    """Return a function ``new_t -> old_t`` for the piecewise-linear warp.

    Parameters
    ----------
    t_start : float
        Start of the warped segment (seconds, original time).
    t_end : float
        End of the warped segment (seconds, original time).
    factor : float
        Speed factor.  ``0.5`` = half speed (2× duration inside segment),
        ``2.0`` = double speed (½× duration inside segment).
    """
    seg_dur = t_end - t_start
    new_seg_dur = seg_dur / factor
    offset = new_seg_dur - seg_dur  # how much longer/shorter the tail becomes

    def warp(new_t: float) -> float:
        if new_t <= t_start:
            return new_t
        elif new_t <= t_start + new_seg_dur:
            return t_start + (new_t - t_start) * factor
        else:
            return new_t - offset

    return warp


def _new_duration(
    original_last_time: float,
    t_start: float,
    t_end: float,
    factor: float,
) -> float:
    """Total duration of the new (warped) motion."""
    if original_last_time <= t_start:
        return original_last_time
    if original_last_time <= t_end:
        return t_start + (original_last_time - t_start) / factor
    new_seg_dur = (t_end - t_start) / factor
    offset = new_seg_dur - (t_end - t_start)
    return original_last_time + offset


# ---------------------------------------------------------------------------
#  CSV load / save
# ---------------------------------------------------------------------------

def _load_csv(csv_path: pathlib.Path) -> tuple[list[str], list[list[float]]]:
    """Read a CSV and return (header, rows)."""
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [[float(v) for v in line] for line in reader]
    return header, rows


def _save_csv(
    output_path: pathlib.Path,
    header: list[str],
    rows: list[list[float]],
) -> None:
    """Write rows to a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


# ---------------------------------------------------------------------------
#  Interpolation
# ---------------------------------------------------------------------------


def _interpolate_row(
    row0: list[float],
    row1: list[float],
    t: float,
    quat_groups: dict[str, list[int]],
    n_cols: int,
) -> list[float]:
    """Return a single interpolated row.

    Parameters
    ----------
    row0, row1 : list[float]
        The two bracketing original rows.
    t : float
        Interpolation parameter in [0, 1].
    quat_groups : dict
        ``{prefix: [idx_qx, idx_qy, idx_qz, idx_qw]}`` for slerp columns.
    n_cols : int
        Total number of columns.
    """
    result = [0.0] * n_cols

    # Collect quaternion column indices for skip in lerp
    quat_indices: set[int] = set()
    for idxs in quat_groups.values():
        quat_indices.update(idxs)

    # --- slerp quaternion groups ---
    for __prefix, idxs in quat_groups.items():
        q0 = np.array([row0[i] for i in idxs], dtype=float)
        q1 = np.array([row1[i] for i in idxs], dtype=float)
        qi = slerp(q0, q1, t)
        for j, idx in enumerate(idxs):
            result[idx] = float(qi[j])

    # --- lerp everything else ---
    for i in range(n_cols):
        if i in quat_indices:
            continue
        result[i] = row0[i] + t * (row1[i] - row0[i])

    return result


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------


def warp_csv_motion(
    csv_path: pathlib.Path,
    fps: float,
    factor: float,
    start_time: float | None = None,
    end_time: float | None = None,
    start_frame: int | None = None,
    end_frame: int | None = None,
    output_path: pathlib.Path | None = None,
) -> tuple[list[str], list[list[float]]]:
    """Load a CSV, warp speed of a segment, return (header, new_rows).

    The segment can be specified either by **time** (seconds) or by
    **frame index** (0-based).  When both are given, frame indices
    take precedence.

    Parameters
    ----------
    csv_path : pathlib.Path
        Path to the input CSV.
    fps : float
        Original frames per second.
    factor : float
        Speed factor for the segment (0.5 = half speed).
    start_time, end_time : float or None
        Segment boundaries in seconds.
    start_frame, end_frame : int or None
        Segment boundaries in frame indices (0-based, inclusive).
    output_path : pathlib.Path or None
        If given, the warped CSV is written to this path.

    Returns
    -------
    header : list[str]
        CSV column names.
    new_rows : list[list[float]]
        The warped motion data.
    """
    header, rows = _load_csv(csv_path)
    n_frames = len(rows)
    n_cols = len(header)
    dt = 1.0 / fps

    # --- resolve segment boundaries -------------------------------------------
    if start_frame is not None and end_frame is not None:
        t_start = start_frame * dt
        t_end = end_frame * dt
    elif start_time is not None and end_time is not None:
        t_start = start_time
        t_end = end_time
    else:
        # Warp the entire motion
        t_start = 0.0
        t_end = (n_frames - 1) * dt

    # Clamp to valid range
    t_start = max(0.0, t_start)
    t_end = min((n_frames - 1) * dt, t_end)

    if t_end <= t_start:
        raise ValueError(
            f"Empty or invalid segment: start={t_start:.3f}s, end={t_end:.3f}s"
        )
    if factor <= 0:
        raise ValueError(f"Speed factor must be positive, got {factor}")

    orig_dur = (n_frames - 1) * dt
    print(f"Original frames: {n_frames}  |  fps: {fps}  |  duration: {orig_dur:.3f}s")
    print(f"Warp segment:    [{t_start:.3f}s, {t_end:.3f}s]  × {factor:.3f}")

    # --- build time warp ------------------------------------------------------
    warp_fn = _build_time_map(t_start, t_end, factor)
    original_last_time = (n_frames - 1) * dt
    new_dur = _new_duration(original_last_time, t_start, t_end, factor)
    new_n_frames = int(round(new_dur * fps)) + 1

    print(f"New frames:      {new_n_frames}  |  duration: {new_dur:.3f}s")

    # --- identify quaternion columns ------------------------------------------
    quat_groups = _find_quaternion_groups(header)
    if quat_groups:
        print(f"Quaternion groups (slerp): {', '.join(quat_groups.keys())}")
    else:
        print("No quaternion columns detected – all values use linear interpolation")

    # --- resample -------------------------------------------------------------
    new_rows: list[list[float]] = []
    for i in range(new_n_frames):
        new_t = i * dt
        old_t = warp_fn(new_t)

        # Find bracketing original frames
        old_idx = old_t / dt
        i0 = int(np.floor(old_idx))
        i1 = i0 + 1
        frac = old_idx - i0  # in [0, 1)

        # Clamp to valid range
        i0 = max(0, min(i0, n_frames - 1))
        i1 = max(0, min(i1, n_frames - 1))
        frac = max(0.0, min(frac, 1.0))

        if i0 == i1 or frac < 1e-12:
            new_rows.append(list(rows[i0]))
        else:
            new_rows.append(
                _interpolate_row(rows[i0], rows[i1], frac, quat_groups, n_cols)
            )

    # --- write output ---------------------------------------------------------
    if output_path is not None:
        _save_csv(output_path, header, new_rows)
        print(f"Saved to: {output_path}")

    return header, new_rows


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Change the playback speed of a segment in a robot-motion CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "csv",
        type=pathlib.Path,
        help="Path to the input CSV file.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        required=True,
        help="Original frames per second of the CSV.",
    )

    seg_group = parser.add_argument_group("Segment specification")
    seg_start = seg_group.add_mutually_exclusive_group()
    seg_start.add_argument(
        "--start-time",
        type=float,
        default=None,
        help="Segment start in seconds.",
    )
    seg_start.add_argument(
        "--start-frame",
        type=int,
        default=None,
        help="Segment start as a frame index (0-based).",
    )
    seg_end = seg_group.add_mutually_exclusive_group()
    seg_end.add_argument(
        "--end-time",
        type=float,
        default=None,
        help="Segment end in seconds.",
    )
    seg_end.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="Segment end as a frame index (0-based, inclusive).",
    )

    parser.add_argument(
        "--factor",
        type=float,
        required=True,
        help=(
            "Speed factor for the segment. "
            "0.5 = half speed (2× duration), 2.0 = double speed (½× duration)."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=pathlib.Path,
        default=None,
        help="Output CSV path.  Required unless you just want a dry-run summary.",
    )

    args = parser.parse_args()

    # --- consistency checks ---------------------------------------------------
    # Start
    has_start_frame = args.start_frame is not None
    has_start_time = args.start_time is not None
    if not has_start_frame and not has_start_time:
        # neither given → will warp entire motion, which is fine
        pass

    # End
    has_end_frame = args.end_frame is not None
    has_end_time = args.end_time is not None

    # Mismatched pairs
    if has_start_frame != has_end_frame:
        parser.error("--start-frame and --end-frame must be given together.")
    if has_start_time != has_end_time:
        parser.error("--start-time and --end-time must be given together.")
    # Can't mix frame and time
    if has_start_frame and has_start_time:
        parser.error(
            "Use either --start/--end-frame OR --start/--end-time, not both."
        )
    if has_start_frame and has_end_time:
        parser.error(
            "Use either --start/--end-frame OR --start/--end-time, not both."
        )

    warp_csv_motion(
        csv_path=args.csv,
        fps=args.fps,
        factor=args.factor,
        start_time=args.start_time if has_start_time else None,
        end_time=args.end_time if has_end_time else None,
        start_frame=args.start_frame if has_start_frame else None,
        end_frame=args.end_frame if has_start_frame else None,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
