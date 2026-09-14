#!/usr/bin/env python3
"""Visualize correspondence-AE training clouds and 1-1 SMPL-X ↔ robot slots.

Slots that share an index are the same semantic surface point. They are colored
by Mixamo-style body part so the match is readable.

Examples::

    python scripts/vis_correspondence_slots.py --data data/correspondence_ae_tpose_4096.npz
    python scripts/vis_correspondence_slots.py \\
        --slots output/correspondence_template_residual_ae/correspondence_slots_final.npz \\
        --data data/correspondence_ae_tpose_4096.npz --robot d20
    python scripts/vis_correspondence_slots.py \\
        --pairs output/correspondence_template_residual_ae/correspondence_contact_smplx_neutral_d20.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_PROJECT_ROOT / "general_motion_retargeting" / "points_contact") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "general_motion_retargeting" / "points_contact"))

# Mixamo / SMPL vertex-segmentation names (HiPHI-style).
PART_NAMES = (
    "hips",
    "torso",
    "head",
    "leftShoulder",
    "rightShoulder",
    "leftUpperArm",
    "rightUpperArm",
    "leftArm",
    "rightArm",
    "leftForeArm",
    "rightForeArm",
    "leftHand",
    "rightHand",
    "leftUpLeg",
    "rightUpLeg",
    "leftLeg",
    "rightLeg",
    "leftFoot",
    "rightFoot",
)
PART_COLORS = {
    "hips": (0.55, 0.35, 0.18),
    "torso": (0.62, 0.62, 0.66),
    "head": (0.95, 0.78, 0.22),
    "leftShoulder": (0.12, 0.38, 0.82),
    "rightShoulder": (0.82, 0.16, 0.18),
    "leftUpperArm": (0.22, 0.55, 0.98),
    "rightUpperArm": (0.92, 0.32, 0.28),
    "leftArm": (0.08, 0.72, 0.90),
    "rightArm": (0.95, 0.48, 0.18),
    "leftForeArm": (0.30, 0.82, 0.72),
    "rightForeArm": (0.95, 0.62, 0.32),
    "leftHand": (0.55, 0.90, 1.00),
    "rightHand": (1.00, 0.80, 0.52),
    "leftUpLeg": (0.16, 0.62, 0.30),
    "rightUpLeg": (0.62, 0.22, 0.78),
    "leftLeg": (0.38, 0.82, 0.42),
    "rightLeg": (0.78, 0.42, 0.88),
    "leftFoot": (0.12, 0.42, 0.26),
    "rightFoot": (0.46, 0.16, 0.56),
}
# SMPL-X body joints 0–21 → Mixamo part. LeftArm/RightArm are then split
# along the humerus into leftUpperArm/rightUpperArm (proximal) vs Arm (distal).
_SMPLX_JOINT_PART = (
    "hips",
    "leftUpLeg",
    "rightUpLeg",
    "torso",
    "leftLeg",
    "rightLeg",
    "torso",
    "leftFoot",
    "rightFoot",
    "torso",
    "leftFoot",
    "rightFoot",
    "head",
    "leftShoulder",
    "rightShoulder",
    "head",
    "leftArm",
    "rightArm",
    "leftForeArm",
    "rightForeArm",
    "leftHand",
    "rightHand",
)


def _is_robot_name(name: str) -> bool:
    name = str(name)
    return not any(name == p or name.startswith(f"{p}_") for p in ("smpl", "smplx", "smplh"))


def _side_from_name(n: str) -> str | None:
    if "left" in n or n.startswith("l_"):
        return "left"
    if "right" in n or n.startswith("r_"):
        return "right"
    return None


def robot_body_to_part(name: str) -> str:
    n = str(name).lower()
    side = _side_from_name(n)

    def lr(left: str, right: str) -> str:
        if side == "left":
            return left
        if side == "right":
            return right
        return "torso"

    if any(k in n for k in ("head", "neck")) and "shoulder" not in n:
        return "head"
    if "hand" in n or "wrist" in n:
        return lr("leftHand", "rightHand")
    if "elbow" in n or "forearm" in n:
        return lr("leftForeArm", "rightForeArm")
    if "collar" in n or "shoulder_pitch" in n:
        return lr("leftShoulder", "rightShoulder")
    if "shoulder_roll" in n:
        return lr("leftUpperArm", "rightUpperArm")
    if "shoulder_yaw" in n or "upper_arm" in n or "shoulder" in n:
        return lr("leftArm", "rightArm")
    if "ankle" in n or "foot" in n or "toe" in n:
        return lr("leftFoot", "rightFoot")
    if "knee" in n:
        return lr("leftLeg", "rightLeg")
    if "hip" in n or "upleg" in n or "thigh" in n:
        return lr("leftUpLeg", "rightUpLeg")
    if any(k in n for k in ("pelvis", "base_link", "aux_base")):
        return "hips"
    if any(k in n for k in ("waist", "torso", "spine", "imu")):
        return "torso"
    return "torso"


def colors_from_parts(parts: np.ndarray) -> np.ndarray:
    rgba = np.ones((len(parts), 4), dtype=np.float32)
    for i, part in enumerate(parts):
        rgb = PART_COLORS.get(str(part), (0.7, 0.7, 0.7))
        rgba[i, :3] = rgb
    return rgba


def _split_upper_arm(points: np.ndarray, joints: np.ndarray, parts: np.ndarray) -> np.ndarray:
    """Split Mixamo LeftArm/RightArm into proximal UpperArm and distal Arm."""
    parts = np.asarray(parts, dtype=object).copy()
    for arm_name, upper_name, sho, elb in (
        ("leftArm", "leftUpperArm", 16, 18),
        ("rightArm", "rightUpperArm", 17, 19),
    ):
        mask = parts == arm_name
        if not np.any(mask):
            continue
        axis = joints[elb] - joints[sho]
        length = float(np.linalg.norm(axis)) + 1e-8
        axis = axis / length
        t = ((points[mask] - joints[sho]) @ axis) / length
        idx = np.where(mask)[0]
        parts[idx[t < 0.5]] = upper_name
    return parts


def assign_parts_from_smplx_joints(points: np.ndarray, joints: np.ndarray) -> np.ndarray:
    d = points[:, None, :] - joints[None, :, :]
    nearest = np.argmin(np.einsum("ijk,ijk->ij", d, d), axis=1)
    parts = np.asarray([_SMPLX_JOINT_PART[int(j)] for j in nearest], dtype=object)
    return _split_upper_arm(points, joints, parts)


def vertex_parts_from_lbs(lbs: np.ndarray, verts: np.ndarray, joints: np.ndarray) -> np.ndarray:
    w = np.asarray(lbs, dtype=np.float64)
    if w.shape[1] > 22:
        w = w.copy()
        jaw_end = min(25, w.shape[1])
        if jaw_end > 22:
            w[:, 15] += w[:, 22:jaw_end].sum(axis=1)
        if w.shape[1] >= 40:
            w[:, 20] += w[:, 25:40].sum(axis=1)
        if w.shape[1] >= 55:
            w[:, 21] += w[:, 40:55].sum(axis=1)
    nearest = np.argmax(w[:, :22], axis=1)
    parts = np.asarray([_SMPLX_JOINT_PART[int(j)] for j in nearest], dtype=object)
    return _split_upper_arm(verts, joints, parts)


def assign_parts_from_smplx_mesh(
    points: np.ndarray,
    verts: np.ndarray,
    vert_parts: np.ndarray,
    offset: np.ndarray,
) -> np.ndarray:
    from scipy.spatial import cKDTree

    centered = np.asarray(verts, dtype=np.float64) - np.asarray(offset, dtype=np.float64)[None, :]
    _, idx = cKDTree(centered).query(np.asarray(points, dtype=np.float64), k=1)
    return np.asarray(vert_parts, dtype=object)[idx]


def assign_parts_heuristic(points: np.ndarray) -> np.ndarray:
    """Y-up fallback when SMPL-X joints are unavailable."""
    p = np.asarray(points, dtype=np.float64)
    y = p[:, 1]
    x = p[:, 0]
    y01 = (y - y.min()) / max(float(y.max() - y.min()), 1e-8)
    x_span = max(float(np.percentile(np.abs(x), 95)), 1e-8)
    ax = np.abs(x)
    parts = np.full(len(p), "torso", dtype=object)
    parts[y01 < 0.18] = "hips"
    parts[y01 > 0.88] = "head"
    arm = (y01 > 0.48) & (y01 < 0.90) & (ax > 0.26 * x_span)
    left, right = x >= 0, x < 0
    parts[arm & left & (ax > 0.72 * x_span)] = "leftHand"
    parts[arm & right & (ax > 0.72 * x_span)] = "rightHand"
    parts[arm & left & (ax > 0.55 * x_span) & (ax <= 0.72 * x_span)] = "leftForeArm"
    parts[arm & right & (ax > 0.55 * x_span) & (ax <= 0.72 * x_span)] = "rightForeArm"
    parts[arm & left & (ax > 0.40 * x_span) & (ax <= 0.55 * x_span)] = "leftArm"
    parts[arm & right & (ax > 0.40 * x_span) & (ax <= 0.55 * x_span)] = "rightArm"
    parts[arm & left & (ax <= 0.40 * x_span)] = "leftUpperArm"
    parts[arm & right & (ax <= 0.40 * x_span)] = "rightUpperArm"
    collar = (y01 > 0.72) & (y01 < 0.88) & (ax > 0.12 * x_span) & (ax < 0.32 * x_span)
    parts[collar & left] = "leftShoulder"
    parts[collar & right] = "rightShoulder"
    leg = (y01 < 0.50) & (ax > 0.06 * x_span)
    parts[leg & left & (y01 < 0.12)] = "leftFoot"
    parts[leg & right & (y01 < 0.12)] = "rightFoot"
    parts[leg & left & (y01 >= 0.12) & (y01 < 0.30)] = "leftLeg"
    parts[leg & right & (y01 >= 0.12) & (y01 < 0.30)] = "rightLeg"
    parts[leg & left & (y01 >= 0.30)] = "leftUpLeg"
    parts[leg & right & (y01 >= 0.30)] = "rightUpLeg"
    return parts


def load_smplx_rest_template(
    smplx_model: Path, gender: str = "neutral"
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    try:
        from general_motion_retargeting.utils.smplx_from_mixamo import MixamoToSMPLX
        import torch
    except Exception:
        return None
    if not smplx_model.exists():
        return None
    try:
        driver = MixamoToSMPLX(smplx_model, gender=gender, ext="pkl")
        with torch.no_grad():
            out = driver.model()
            verts = out.vertices[0].cpu().numpy().astype(np.float64)
            joints = out.joints[0].cpu().numpy().astype(np.float64)[:22]
        vert_parts = vertex_parts_from_lbs(driver._lbs, verts, joints)
        return verts, joints, vert_parts
    except Exception as exc:
        print(f"  [warn] could not load SMPL-X template: {exc}")
        return None


def _sample_idx(names: list[str], query: str | None, kind: str) -> int:
    if query:
        if query not in names:
            raise ValueError(f"{kind} {query!r} not in {names}")
        return names.index(query)
    if kind == "source":
        for prefix in ("smplx_neutral", "smplx"):
            for i, name in enumerate(names):
                if name == prefix or name.startswith("smplx"):
                    return i
        return 0
    for prefer in ("d20",):
        if prefer in names:
            return names.index(prefer)
    for i, name in enumerate(names):
        if _is_robot_name(name):
            return i
    raise ValueError(f"no robot sample in {names}")


def assign_robot_parts_from_dataset(robot_pts: np.ndarray, data, rob_i: int) -> np.ndarray | None:
    if data is None or f"face_body_ids_{rob_i}" not in data or f"body_names_{rob_i}" not in data:
        return None
    from scipy.spatial import cKDTree

    verts = np.asarray(data[f"mesh_vertices_{rob_i}"], dtype=np.float64)
    faces = np.asarray(data[f"mesh_faces_{rob_i}"])
    body_ids = np.asarray(data[f"face_body_ids_{rob_i}"])
    body_names = np.asarray(data[f"body_names_{rob_i}"]).astype(str)
    _, nearest_v = cKDTree(verts).query(np.asarray(robot_pts, dtype=np.float64), k=1)
    face_for_v = np.full(len(verts), -1, dtype=np.int32)
    face_for_v[faces[:, 0]] = np.arange(len(faces), dtype=np.int32)
    face_ids = face_for_v[nearest_v]
    parts = np.full(len(robot_pts), "torso", dtype=object)
    valid = face_ids >= 0
    for i in np.where(valid)[0]:
        bid = int(body_ids[int(face_ids[i])])
        name = body_names[bid] if 0 <= bid < len(body_names) else ""
        parts[i] = robot_body_to_part(name)
    return parts


def load_visualization_payload(args) -> dict:
    if args.pairs is not None:
        pairs = np.load(args.pairs, allow_pickle=True)
        source_pts = np.asarray(pairs["source_points"], dtype=np.float32)
        robot_pts = np.asarray(pairs["robot_points"], dtype=np.float32)
        robot_parts = None
        if "robot_body_names" in pairs:
            robot_parts = np.asarray(
                [robot_body_to_part(n) for n in pairs["robot_body_names"].astype(str)],
                dtype=object,
            )
        return {
            "source_name": str(np.asarray(pairs["source_name"])),
            "robot_name": str(np.asarray(pairs["robot_name"])),
            "source_points": source_pts,
            "robot_points": robot_pts,
            "source_target": None,
            "robot_target": None,
            "robot_parts": robot_parts,
            "source_offset": np.zeros(3, dtype=np.float32),
            "paired": True,
        }

    if args.slots is None and args.data is None:
        raise ValueError("pass --pairs, or --data / --slots")

    slots = np.load(args.slots, allow_pickle=True) if args.slots else None
    data = np.load(args.data, allow_pickle=True) if args.data else None
    if slots is None and data is None:
        raise ValueError("nothing to load")

    names = np.asarray((slots or data)["names"]).astype(str).tolist()
    src_i = _sample_idx(names, args.source, "source")
    rob_i = _sample_idx(names, args.robot, "robot")
    source_name, robot_name = names[src_i], names[rob_i]

    if slots is not None:
        source_pts = np.asarray(slots["reconstructed_slots"][src_i], dtype=np.float32)
        robot_pts = np.asarray(slots["reconstructed_slots"][rob_i], dtype=np.float32)
        source_target = np.asarray(slots["target_points"][src_i], dtype=np.float32)
        robot_target = np.asarray(slots["target_points"][rob_i], dtype=np.float32)
        paired = True
    else:
        source_pts = np.asarray(data["points"][src_i], dtype=np.float32)
        robot_pts = np.asarray(data["points"][rob_i], dtype=np.float32)
        source_target = source_pts
        robot_target = robot_pts
        paired = False
        print("  [warn] dataset-only view: point indices are independent FPS samples, not AE matches")

    source_offset = (
        np.asarray(data["root_offsets"][src_i], dtype=np.float32)
        if data is not None and "root_offsets" in data
        else np.zeros(3, dtype=np.float32)
    )
    robot_parts = assign_robot_parts_from_dataset(robot_pts, data, rob_i)

    return {
        "source_name": source_name,
        "robot_name": robot_name,
        "source_points": source_pts,
        "robot_points": robot_pts,
        "source_target": source_target if args.show_target else None,
        "robot_target": robot_target if args.show_target else None,
        "robot_parts": robot_parts,
        "source_offset": source_offset,
        "paired": paired,
    }


def resolve_source_parts(payload: dict, template) -> np.ndarray:
    pts = payload["source_points"]
    if template is not None:
        verts, joints, vert_parts = template
        joints_c = joints - payload["source_offset"][None, :]
        try:
            return assign_parts_from_smplx_mesh(
                pts, verts, vert_parts, payload["source_offset"]
            )
        except Exception as exc:
            print(f"  [warn] LBS part assignment failed ({exc}); using nearest joint")
            return assign_parts_from_smplx_joints(pts, joints_c)
    return assign_parts_heuristic(pts)


def side_by_side(source: np.ndarray, robot: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    src = source.copy()
    rob = robot.copy()
    gap = 0.25 + float(src[:, 0].max() - src[:, 0].min()) * 0.15
    offset = float(src[:, 0].max() - rob[:, 0].min()) + gap
    rob[:, 0] = rob[:, 0] + offset
    return src, rob, offset


def plot_matplotlib(payload: dict, source_parts: np.ndarray, robot_parts: np.ndarray, args) -> None:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    src, rob, _ = side_by_side(payload["source_points"], payload["robot_points"])
    src_colors = colors_from_parts(source_parts)
    rob_colors = colors_from_parts(robot_parts)
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(src[:, 0], src[:, 2], src[:, 1], c=src_colors, s=args.point_size, depthshade=False, label=payload["source_name"])
    ax.scatter(rob[:, 0], rob[:, 2], rob[:, 1], c=rob_colors, s=args.point_size, depthshade=False, label=payload["robot_name"])
    if payload["source_target"] is not None:
        tsrc, trob, _ = side_by_side(payload["source_target"], payload["robot_target"])
        ax.scatter(tsrc[:, 0], tsrc[:, 2], tsrc[:, 1], c="0.75", s=max(1, args.point_size * 0.4), alpha=0.15, depthshade=False)
        ax.scatter(trob[:, 0], trob[:, 2], trob[:, 1], c="0.75", s=max(1, args.point_size * 0.4), alpha=0.15, depthshade=False)
    if args.lines > 0 and payload.get("paired", True):
        rng = np.random.default_rng(0)
        n = min(int(args.lines), len(src))
        pick = rng.choice(len(src), size=n, replace=False)
        for i in pick:
            ax.plot(
                [src[i, 0], rob[i, 0]],
                [src[i, 2], rob[i, 2]],
                [src[i, 1], rob[i, 1]],
                color=src_colors[i, :3],
                alpha=0.25,
                linewidth=0.4,
            )
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=PART_COLORS[name], markersize=7, label=name)
        for name in PART_NAMES
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8, ncol=1, frameon=False)
    ax.set_title(f"{payload['source_name']}  ↔  {payload['robot_name']}   ({args.color_by})")
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Y (up)")
    ax.view_init(elev=12, azim=-75)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    fig.tight_layout()
    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=160)
        print(f"  saved {args.save}")
    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)


def plot_trimesh(payload: dict, source_parts: np.ndarray, robot_parts: np.ndarray, args) -> None:
    import trimesh

    src, rob, _ = side_by_side(payload["source_points"], payload["robot_points"])
    src_colors = (colors_from_parts(source_parts) * 255).astype(np.uint8)
    rob_colors = (colors_from_parts(robot_parts) * 255).astype(np.uint8)
    geoms = [
        trimesh.PointCloud(src, colors=src_colors),
        trimesh.PointCloud(rob, colors=rob_colors),
    ]
    if args.lines > 0 and payload.get("paired", True):
        rng = np.random.default_rng(0)
        n = min(int(args.lines), len(src))
        pick = rng.choice(len(src), size=n, replace=False)
        segs = np.stack([src[pick], rob[pick]], axis=1)
        path = trimesh.load_path(segs)
        geoms.append(path)
    scene = trimesh.Scene(geoms)
    print(f"  trimesh  {payload['source_name']} ↔ {payload['robot_name']}")
    scene.show()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize correspondence AE slots colored by body part.")
    p.add_argument("--data", type=Path, default=None, help="Dataset npz from build_correspondence_ae_dataset.py")
    p.add_argument("--slots", type=Path, default=None, help="correspondence_slots_final.npz")
    p.add_argument("--pairs", type=Path, default=None, help="correspondence_contact_<src>_<robot>.npz")
    p.add_argument("--source", type=str, default=None)
    p.add_argument("--robot", type=str, default=None)
    p.add_argument("--smplx-model", type=Path, default=_PROJECT_ROOT / "assets" / "smpl-x")
    p.add_argument("--gender", type=str, default="neutral")
    p.add_argument("--color-by", choices=("source-part", "robot-body"), default="source-part",
                   help="source-part: same SMPL-X part color on both. robot-body: color by robot link.")
    p.add_argument("--show-target", action="store_true", help="Also draw raw training target points in gray.")
    p.add_argument("--lines", type=int, default=256, help="Number of correspondence lines (0 disables).")
    p.add_argument("--point-size", type=float, default=6.0)
    p.add_argument("--backend", choices=("matplotlib", "trimesh"), default="matplotlib")
    p.add_argument("--save", type=Path, default=None, help="Save a matplotlib PNG.")
    p.add_argument("--no-show", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    payload = load_visualization_payload(args)
    template = None
    if args.color_by == "source-part":
        template = load_smplx_rest_template(args.smplx_model.resolve(), args.gender)
        if template is None:
            print("  using geometric body-part heuristic (SMPL-X template unavailable)")
    source_parts = resolve_source_parts(payload, template)
    robot_parts = payload.get("robot_parts")
    if args.color_by == "robot-body" and robot_parts is not None:
        source_parts = robot_parts if payload["paired"] else source_parts
    elif payload["paired"] or robot_parts is None:
        robot_parts = source_parts
    counts = {name: int(np.sum(source_parts == name)) for name in PART_NAMES}
    print(f"  pair  {payload['source_name']} ↔ {payload['robot_name']}  slots={len(source_parts)}")
    print("  smpl  " + "  ".join(f"{k}={v}" for k, v in counts.items()))
    if robot_parts is not source_parts:
        rcounts = {name: int(np.sum(robot_parts == name)) for name in PART_NAMES}
        print("  robot " + "  ".join(f"{k}={v}" for k, v in rcounts.items()))
    if args.backend == "trimesh":
        plot_trimesh(payload, source_parts, robot_parts, args)
    else:
        plot_matplotlib(payload, source_parts, robot_parts, args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
