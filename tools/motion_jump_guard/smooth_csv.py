#!/usr/bin/env python3
"""Smooth robot motion CSV trajectories in joint/position space.

Typical usage::

    python tools/motion_jump_guard/smooth_csv.py outputs/d20_25dof/fz_dance_13.csv \\
        --radius 10 --method gaussian -o smoothed.csv

The script keeps base-quaternion columns unchanged (except normalisation).
All other scalar columns (base position + joints) are smoothed independently.
"""

import argparse
import pathlib
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

QUAT_COLS = {"base_qx", "base_qy", "base_qz", "base_qw"}


def _ensure_2d(data: np.ndarray) -> np.ndarray:
    return data.reshape(1, -1) if data.ndim == 1 else data


def load_csv(path: pathlib.Path) -> tuple[list[str], np.ndarray]:
    with open(path, encoding="utf-8") as fh:
        header = fh.readline().strip().split(",")
    data = _ensure_2d(np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64))
    if data.shape[1] != len(header):
        raise ValueError(
            f"Header has {len(header)} columns but data has {data.shape[1]}"
        )
    return header, data


def column_indices(header: list[str], mode: str) -> np.ndarray:
    """Return column indices to smooth based on *mode*.

    ``"joints"``  – everything except the first 7 columns (base pos + quat).
    ``"scalar"``  – all columns except quaternion columns.
    ``"all"``     – every column (including quaternions – use with care).
    """
    if mode == "joints":
        return np.arange(7, len(header), dtype=int)
    if mode == "scalar":
        return np.asarray(
            [i for i, name in enumerate(header) if name not in QUAT_COLS], dtype=int
        )
    if mode == "all":
        return np.arange(len(header), dtype=int)
    raise ValueError(f"Unknown --columns mode: {mode}")


def quat_indices(header: list[str]) -> np.ndarray:
    qs = ["base_qx", "base_qy", "base_qz", "base_qw"]
    idx = [header.index(n) for n in qs if n in header]
    return np.asarray(idx, dtype=int) if len(idx) == 4 else np.asarray([], dtype=int)


def normalise_quaternions(data: np.ndarray, qi: np.ndarray) -> None:
    if qi.size == 0:
        return
    q = data[:, qi]
    n = np.linalg.norm(q, axis=1)
    ok = n > 1e-12
    q[ok] /= n[ok, None]
    data[:, qi] = q


# ---------------------------------------------------------------------------
# Smoothing kernels
# ---------------------------------------------------------------------------


def moving_average(col: np.ndarray, radius: int) -> np.ndarray:
    """Centred moving-average filter with edge-padding."""
    padded = np.pad(col, (radius, radius), mode="edge")
    csum = np.cumsum(np.insert(padded, 0, 0))
    w = 2 * radius + 1
    return (csum[w:] - csum[:-w]) / w


def gaussian(col: np.ndarray, radius: int, sigma: float | None) -> np.ndarray:
    """Centred Gaussian-weighted filter."""
    sigma = sigma if sigma is not None else max(radius / 2.0, 1e-6)
    xs = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (xs / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(col, (radius, radius), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def savgol(col: np.ndarray, radius: int, polyorder: int) -> np.ndarray:
    """Savitzky-Golay filter (scipy)."""
    from scipy.signal import savgol_filter

    w = 2 * radius + 1
    if col.shape[0] < w:
        return col.copy()
    return savgol_filter(
        col, window_length=w, polyorder=min(polyorder, w - 1), mode="interp"
    )


# ---------------------------------------------------------------------------
# Main smoothing entry-point
# ---------------------------------------------------------------------------


def smooth_data(
    data: np.ndarray,
    cols: np.ndarray,
    radius: int,
    method: str,
    *,
    sigma: float | None = None,
    savgol_polyorder: int = 3,
) -> np.ndarray:
    """Return a copy of *data* with columns *cols* smoothed in-place."""
    if radius <= 0 or data.shape[0] < 3:
        return data.copy()

    out = data.copy()
    apply = _pick_method(method, radius, sigma, savgol_polyorder)

    for ci in cols:
        out[:, ci] = apply(out[:, ci])

    return out


def _pick_method(
    method: str,
    radius: int,
    sigma: float | None,
    polyorder: int,
):
    if method == "moving_average":
        return lambda c: moving_average(c, radius)
    if method == "gaussian":
        return lambda c: gaussian(c, radius, sigma)
    if method == "savgol":
        return lambda c: savgol(c, radius, polyorder)
    raise ValueError(f"Unknown --method: {method}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Smooth robot-motion CSV trajectories via per-column filtering."
    )
    p.add_argument("input", type=pathlib.Path, help="Input CSV file.")
    p.add_argument("-o", "--output", type=pathlib.Path, default=None,
                   help="Output CSV (default: <input>_smooth.csv).")
    p.add_argument("--radius", type=int, default=10,
                   help="Filter half-window in frames (default: 10).")
    p.add_argument("--method", choices=("gaussian", "moving_average", "savgol"),
                   default="gaussian", help="Smoothing method (default: gaussian).")
    p.add_argument("--columns", choices=("joints", "scalar", "all"),
                   default="scalar",
                   help="Which columns to smooth. 'joints' = col 7+ only; "
                        "'scalar' = all except quaternions (default); "
                        "'all' = everything.")
    p.add_argument("--sigma", type=float, default=None,
                   help="Gaussian sigma (default: radius/2).")
    p.add_argument("--savgol-polyorder", type=int, default=3,
                   help="Savitzky-Golay polynomial order (default: 3).")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    src = args.input.resolve()
    if not src.exists():
        print(f"ERROR: file not found: {src}", file=sys.stderr)
        return 1
    if args.radius < 1:
        print("ERROR: --radius must be >= 1.", file=sys.stderr)
        return 1

    dst = (args.output.resolve() if args.output
           else src.parent / f"{src.stem}_smooth.csv")

    header, data = load_csv(src)
    cols = column_indices(header, args.columns)
    qi = quat_indices(header)

    print(f"Input  : {src}  ({data.shape[0]} frames, {len(header)} columns)")
    print(f"Output : {dst}")
    print(f"Method : {args.method}, radius = {args.radius} frames")
    print(f"Columns: {args.columns} → smoothing {len(cols)} / {len(header)} columns")
    if qi.size:
        print(f"Quaternion columns (indices {qi.tolist()}) will be normalised "
              f"{'and smoothed' if any(c in cols for c in qi) else 'but NOT smoothed'}.")

    # Smooth
    out = smooth_data(
        data, cols, args.radius, args.method,
        sigma=args.sigma,
        savgol_polyorder=args.savgol_polyorder,
    )
    normalise_quaternions(out, qi)

    np.savetxt(
        dst, out, delimiter=",", fmt="%.8f",
        header=",".join(header), comments="",
    )
    print(f"Done.  Wrote {out.shape[0]} frames to {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
