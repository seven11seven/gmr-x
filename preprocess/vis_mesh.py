#!/usr/bin/env python3
"""
Visualize HiPHI (or any) OBJ meshes in the MuJoCo viewer.

OBJ files in ``datas/hiphi/object_meshes`` are authored in centimetres with
Y-up. This tool scales them to metres, stands them on the ground (Y-up →
Z-up), and optionally overlays an AABB so you can judge size.

Quickstart::

    # Gallery of every unique (non-mirror) mesh
    python preprocess/vis_mesh.py

    # One mesh, by stem or path
    python preprocess/vis_mesh.py Chair_Q_1
    python preprocess/vis_mesh.py datas/hiphi/object_meshes/Table_A_1.obj

    # Print AABB sizes without opening the viewer
    python preprocess/vis_mesh.py --list
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import mujoco as mj
import mujoco.viewer as mjv

DEFAULT_MESH_DIR = _PROJECT_ROOT / "datas" / "hiphi" / "object_meshes"
DEFAULT_SCALE = 0.01  # OBJ centimetres → MuJoCo metres

# Category colour from the filename prefix (Chair_Q_1 → Chair).
_CATEGORY_RGBA = {
    "Ball": (0.90, 0.22, 0.22, 1.0),
    "ScoccerBall": (0.95, 0.95, 0.95, 1.0),
    "Bench": (0.55, 0.38, 0.22, 1.0),
    "Bottle": (0.35, 0.70, 0.85, 1.0),
    "Box": (0.78, 0.62, 0.32, 1.0),
    "Bucket": (0.40, 0.55, 0.40, 1.0),
    "Chair": (0.72, 0.45, 0.28, 1.0),
    "Clothrack": (0.55, 0.55, 0.58, 1.0),
    "Mop": (0.45, 0.35, 0.25, 1.0),
    "StepStool": (0.65, 0.50, 0.30, 1.0),
    "Table": (0.50, 0.36, 0.22, 1.0),
    "Trashbin": (0.30, 0.45, 0.35, 1.0),
}
_DEFAULT_RGBA = (0.70, 0.70, 0.70, 1.0)
_AABB_RGBA = (0.20, 0.80, 1.00, 0.18)


# ---------------------------------------------------------------------------
#  Mesh IO / sizing
# ---------------------------------------------------------------------------

def load_vertices(obj_path: pathlib.Path) -> np.ndarray:
    verts = []
    with open(obj_path) as fh:
        for line in fh:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
    if not verts:
        raise ValueError(f"no vertices in {obj_path}")
    return np.asarray(verts, dtype=np.float64)


def mesh_aabb(verts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mn = verts.min(axis=0)
    mx = verts.max(axis=0)
    return mn, mx, mx - mn


def category_of(stem: str) -> str:
    return stem.split("_", 1)[0]


def rgba_of(stem: str) -> tuple[float, float, float, float]:
    return _CATEGORY_RGBA.get(category_of(stem), _DEFAULT_RGBA)


def yup_to_zup(verts: np.ndarray) -> np.ndarray:
    """(x, y, z)_Yup → (x, -z, y)_Zup  ≡  Rx(+90°)."""
    return np.column_stack([verts[:, 0], -verts[:, 2], verts[:, 1]])


# ---------------------------------------------------------------------------
#  Discovery
# ---------------------------------------------------------------------------

def discover_obj_files(mesh_dir: pathlib.Path, include_mirror: bool) -> list[pathlib.Path]:
    files = sorted(mesh_dir.glob("*.obj"))
    if not include_mirror:
        files = [p for p in files if "__mirror" not in p.stem]
    return files


def resolve_mesh_args(
    names: list[str],
    mesh_dir: pathlib.Path,
    include_mirror: bool,
) -> list[pathlib.Path]:
    """Resolve CLI tokens to existing .obj paths."""
    if not names:
        files = discover_obj_files(mesh_dir, include_mirror)
        if not files:
            raise FileNotFoundError(f"no .obj files in {mesh_dir}")
        return files

    resolved: list[pathlib.Path] = []
    seen: set[pathlib.Path] = set()
    for token in names:
        path = pathlib.Path(token)
        candidates: list[pathlib.Path] = []
        if path.is_file():
            candidates = [path]
        elif (mesh_dir / token).is_file():
            candidates = [mesh_dir / token]
        elif (mesh_dir / f"{token}.obj").is_file():
            candidates = [mesh_dir / f"{token}.obj"]
        else:
            candidates = sorted(mesh_dir.glob(token))
            if not candidates:
                candidates = sorted(mesh_dir.glob(f"{token}.obj"))
        if not candidates:
            raise FileNotFoundError(
                f"cannot find mesh '{token}' (looked in {mesh_dir})"
            )
        for c in candidates:
            c = c.resolve()
            if not include_mirror and "__mirror" in c.stem:
                continue
            if c not in seen:
                seen.add(c)
                resolved.append(c)
    if not resolved:
        raise FileNotFoundError(f"no meshes matched {names}")
    return resolved


def print_size_table(paths: list[pathlib.Path], scale: float) -> None:
    header = (
        f"{'mesh':<22} {'n_v':>7} "
        f"{'dx_m':>7} {'dy_m':>7} {'dz_m':>7} "
        f"{'height_m':>8}  min_cm                     max_cm"
    )
    print(header)
    print("-" * len(header))
    for path in paths:
        verts = load_vertices(path)
        mn, mx, size_cm = mesh_aabb(verts)
        size_m = size_cm * scale
        # After Y-up → Z-up, height is original Y.
        height = size_m[1]
        print(
            f"{path.stem:<22} {len(verts):7d} "
            f"{size_m[0]:7.3f} {size_m[1]:7.3f} {size_m[2]:7.3f} "
            f"{height:8.3f}  "
            f"[{mn[0]:7.1f},{mn[1]:7.1f},{mn[2]:7.1f}]  "
            f"[{mx[0]:7.1f},{mx[1]:7.1f},{mx[2]:7.1f}]"
        )
    print()
    print("dx/dy/dz are the raw OBJ AABB (cm×scale). height_m = dy (Y-up).")


# ---------------------------------------------------------------------------
#  MJCF
# ---------------------------------------------------------------------------

def _rgba_str(rgba: tuple[float, float, float, float]) -> str:
    return " ".join(f"{c:.3f}" for c in rgba)


def _grid_spacing(paths: list[pathlib.Path], scale: float, y_up: bool) -> float:
    max_xy = 0.0
    for path in paths:
        verts = load_vertices(path) * scale
        world = yup_to_zup(verts) if y_up else verts
        size = world.max(0) - world.min(0)
        max_xy = max(max_xy, float(size[0]), float(size[1]))
    return max(max_xy * 1.35, 1.2)


def build_mjcf(
    paths: list[pathlib.Path],
    scale: float,
    y_up: bool,
    show_aabb: bool,
) -> str:
    """Build a self-contained MJCF string that instances every mesh."""
    # One meshdir only works when every file shares a parent. Mixed dirs
    # fall back to per-asset absolute paths.
    parents = {p.parent.resolve() for p in paths}
    common_dir = next(iter(parents)) if len(parents) == 1 else None

    root = ET.Element("mujoco", model="hiphi_meshes")
    compiler_kw = {"angle": "degree", "autolimits": "true"}
    if common_dir is not None:
        compiler_kw["meshdir"] = str(common_dir)
    ET.SubElement(root, "compiler", **compiler_kw)
    visual = ET.SubElement(root, "visual")
    ET.SubElement(
        visual, "headlight",
        diffuse="0.6 0.6 0.6", ambient="0.3 0.3 0.3", specular="0.2 0.2 0.2",
    )
    ET.SubElement(visual, "global", azimuth="135", elevation="-25")

    asset = ET.SubElement(root, "asset")
    ET.SubElement(
        asset, "texture", name="grid", type="2d", builtin="checker",
        rgb1="0.22 0.28 0.34", rgb2="0.14 0.18 0.22",
        width="512", height="512",
    )
    ET.SubElement(
        asset, "material", name="grid", texture="grid",
        texrepeat="10 10", texuniform="true", reflectance="0.05",
    )
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(
        world, "light", pos="0 0 8", dir="0 0 -1", directional="true",
        diffuse="0.5 0.5 0.5",
    )
    ET.SubElement(
        world, "geom", name="floor", type="plane", size="40 40 0.1",
        material="grid", contype="0", conaffinity="0",
    )
    # 1 m RGB axes at the origin as a size reference.
    for axis, rgba, from_, to in (
        ("x", "1 0 0 1", "0 0 0.002", "1 0 0.002"),
        ("y", "0 1 0 1", "0 0 0.002", "0 1 0.002"),
        ("z", "0 0 1 1", "0 0 0", "0 0 1"),
    ):
        ET.SubElement(
            world, "geom", name=f"axis_{axis}", type="capsule",
            fromto=f"{from_} {to}", size="0.012", rgba=rgba,
            contype="0", conaffinity="0", group="2",
        )

    n = len(paths)
    cols = max(1, math.ceil(math.sqrt(n)))
    spacing = _grid_spacing(paths, scale, y_up) if n > 1 else 0.0
    euler = "90 0 0" if y_up else "0 0 0"
    scale_str = f"{scale} {scale} {scale}"

    centers = []
    for i, path in enumerate(paths):
        row, col = divmod(i, cols) if n > 1 else (0, 0)
        gx = (col - 0.5 * (cols - 1)) * spacing
        n_rows = math.ceil(n / cols)
        gy = (0.5 * (n_rows - 1) - row) * spacing

        verts = load_vertices(path)
        mn, mx, _size_cm = mesh_aabb(verts)
        verts_m = verts * scale
        verts_w = yup_to_zup(verts_m) if y_up else verts_m
        wmin, wmax = verts_w.min(0), verts_w.max(0)
        # Sit on z=0, centre the footprint on (gx, gy).
        body_pos = np.array([
            gx - 0.5 * (wmin[0] + wmax[0]),
            gy - 0.5 * (wmin[1] + wmax[1]),
            -wmin[2],
        ])
        centers.append(np.array([gx, gy, 0.5 * (wmax[2] - wmin[2])]))

        mesh_name = f"mesh_{i}_{path.stem}"
        body_name = path.stem
        file_attr = path.name if common_dir is not None else str(path.resolve())
        ET.SubElement(asset, "mesh", name=mesh_name, file=file_attr, scale=scale_str)

        body = ET.SubElement(
            world, "body", name=body_name,
            pos=f"{body_pos[0]:.6f} {body_pos[1]:.6f} {body_pos[2]:.6f}",
            euler=euler,
        )
        rgba = rgba_of(path.stem)
        ET.SubElement(
            body, "geom", name=f"{body_name}_mesh", type="mesh", mesh=mesh_name,
            rgba=_rgba_str(rgba), contype="0", conaffinity="0", group="0",
        )
        if show_aabb:
            center_local = 0.5 * (mn + mx) * scale
            half_local = 0.5 * (mx - mn) * scale
            half_local = np.maximum(half_local, 1e-4)
            ET.SubElement(
                body, "geom", name=f"{body_name}_aabb", type="box",
                size=f"{half_local[0]:.6f} {half_local[1]:.6f} {half_local[2]:.6f}",
                pos=f"{center_local[0]:.6f} {center_local[1]:.6f} {center_local[2]:.6f}",
                rgba=_rgba_str(_AABB_RGBA), contype="0", conaffinity="0",
                group="1",
            )

    centers = np.asarray(centers)
    lookat = centers.mean(axis=0)
    extent = float(np.linalg.norm(centers.max(0) - centers.min(0))) + 2.0
    ET.SubElement(
        root, "statistic",
        center=f"{lookat[0]:.4f} {lookat[1]:.4f} {lookat[2]:.4f}",
        extent=f"{extent:.4f}",
    )
    return ET.tostring(root, encoding="unicode")


def launch_viewer(xml: str, look_hint: np.ndarray | None = None) -> None:
    model = mj.MjModel.from_xml_string(xml)
    data = mj.MjData(model)
    mj.mj_forward(model, data)

    print("Viewer keys:  mouse orbit / zoom / pan")
    print("  Left UI → Geom group 0 = mesh, 1 = AABB, 2 = 1 m axes")
    print("Close the window to exit.")

    with mjv.launch_passive(model, data, show_left_ui=True, show_right_ui=False) as viewer:
        viewer.opt.flags[mj.mjtVisFlag.mjVIS_TRANSPARENT] = 0
        viewer.opt.frame = mj.mjtFrame.mjFRAME_NONE
        viewer.opt.label = (
            mj.mjtLabel.mjLABEL_BODY if model.nmesh <= 8 else mj.mjtLabel.mjLABEL_NONE
        )
        if look_hint is not None:
            viewer.cam.lookat[:] = look_hint
        else:
            viewer.cam.lookat[:] = model.stat.center
        viewer.cam.distance = max(float(model.stat.extent) * 1.4, 2.0)
        viewer.cam.elevation = -25
        viewer.cam.azimuth = 135
        while viewer.is_running():
            mj.mj_forward(model, data)
            viewer.sync()
            time.sleep(0.01)


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualize OBJ meshes (HiPHI by default) in MuJoCo.",
    )
    p.add_argument(
        "meshes", nargs="*",
        help="Mesh stem, filename, path, or glob. Default: all unique meshes.",
    )
    p.add_argument(
        "--mesh-dir", type=pathlib.Path, default=DEFAULT_MESH_DIR,
        help=f"Directory of OBJ files (default: {DEFAULT_MESH_DIR})",
    )
    p.add_argument(
        "--scale", type=float, default=DEFAULT_SCALE,
        help="Vertex scale applied in MJCF (default: 0.01, cm→m).",
    )
    p.add_argument(
        "--list", action="store_true",
        help="Print AABB sizes and exit (no viewer).",
    )
    p.add_argument(
        "--include-mirror", action="store_true",
        help="Also load *__mirror.obj files.",
    )
    p.add_argument(
        "--no-y-up", action="store_true",
        help="Do not apply Rx(+90°) Y-up → Z-up conversion.",
    )
    p.add_argument(
        "--no-aabb", action="store_true",
        help="Do not overlay a translucent AABB box on each mesh.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    mesh_dir = args.mesh_dir
    if not mesh_dir.is_absolute():
        mesh_dir = (_PROJECT_ROOT / mesh_dir).resolve()

    paths = resolve_mesh_args(args.meshes, mesh_dir, args.include_mirror)
    print(f"Loaded {len(paths)} mesh(es) from {mesh_dir}")
    print_size_table(paths, args.scale)

    if args.list:
        return

    y_up = not args.no_y_up
    xml = build_mjcf(paths, args.scale, y_up=y_up, show_aabb=not args.no_aabb)
    launch_viewer(xml)


if __name__ == "__main__":
    main()
