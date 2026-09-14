#!/usr/bin/env python3
"""Visualize correspondence-AE training clouds and 1-1 SMPL-X ↔ robot slots.

Slots that share an index are the same semantic surface point. They are colored
by body part (head / torso / L-R arm / L-R leg) so the match is readable.

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

PART_NAMES = ("head", "torso", "left_arm", "right_arm", "left_leg", "right_leg")
PART_COLORS = {
    "head": (0.95, 0.75, 0.20),
    "torso": (0.55, 0.55, 0.58),
    "left_arm": (0.20, 0.55, 0.95),
    "right_arm": (0.90, 0.25, 0.25),
    "left_leg": (0.20, 0.75, 0.40),
    "right_leg": (0.70, 0.30, 0.85),
}
# SMPL-X body joints 0–21 → coarse part.
_SMPLX_JOINT_PART = (
    "torso", "left_leg", "right_leg", "torso",
    "left_leg", "right_leg", "torso",
    "left_leg", "right_leg", "torso",
    "left_leg", "right_leg", "head",
    "left_arm", "right_arm", "head",
    "left_arm", "right_arm", "left_arm", "right_arm",
    "left_arm", "right_arm",
)


def _is_robot_name(name: str) -> bool:
    name = str(name)
    return not any(name == p or name.startswith(f"{p}_") for p in ("smpl", "smplx", "smplh"))


def robot_body_to_part(name: str) -> str:
    n = str(name).lower()
    if any(k in n for k in ("head", "neck", "imu")):
        return "head"
    if any(k in n for k in ("shoulder", "elbow", "wrist", "hand", "arm")):
        return "left_arm" if "left" in n or n.startswith("l_") else "right_arm"
    if any(k in n for k in ("hip", "knee", "ankle", "thigh", "calf", "leg", "foot")):
        return "left_leg" if "left" in n or n.startswith("l_") else "right_leg"
    return "torso"


def colors_from_parts(parts: np.ndarray) -> np.ndarray:
    rgba = np.ones((len(parts), 4), dtype=np.float32)
    for i, part in enumerate(parts):
        rgb = PART_COLORS.get(str(part), (0.7, 0.7, 0.7))
        rgba[i, :3] = rgb
    return rgba


def assign_parts_from_smplx_joints(points: np.ndarray, joints: np.ndarray) -> np.ndarray:
    d = points[:, None, :] - joints[None, :, :]
    nearest = np.argmin(np.einsum("ijk,ijk->ij", d, d), axis=1)
    return np.asarray([_SMPLX_JOINT_PART[int(j)] for j in nearest])


def assign_parts_heuristic(points: np.ndarray) -> np.ndarray:
    """Y-up fallback when SMPL-X joints are unavailable."""
    p = np.asarray(points, dtype=np.float64)
    y = p[:, 1]
    x = p[:, 0]
    y01 = (y - y.min()) / max(float(y.max() - y.min()), 1e-8)
    x_span = max(float(np.percentile(np.abs(x), 95)), 1e-8)
    parts = np.full(len(p), "torso", dtype=object)
    parts[y01 > 0.88] = "head"
    arm = (y01 > 0.42) & (y01 < 0.90) & (np.abs(x) > 0.28 * x_span)
    parts[arm & (x > 0)] = "left_arm"
    parts[arm & (x < 0)] = "right_arm"
    leg = y01 < 0.50
    parts[leg & (x >= 0)] = "left_leg"
    parts[leg & (x < 0)] = "right_leg"
    return parts


def load_smplx_rest_joints(smplx_model: Path, gender: str = "neutral") -> np.ndarray | None:
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
            joints = driver.model().joints[0].cpu().numpy().astype(np.float64)
        return joints[:22]
    except Exception as exc:
        print(f"  [warn] could not load SMPL-X joints: {exc}")
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


def load_visualization_payload(args) -> dict:
    if args.pairs is not None:
        pairs = np.load(args.pairs, allow_pickle=True)
        source_pts = np.asarray(pairs["source_points"], dtype=np.float32)
        robot_pts = np.asarray(pairs["robot_points"], dtype=np.float32)
        if args.color_by == "robot-body" and "robot_body_names" in pairs:
            parts = np.asarray([robot_body_to_part(n) for n in pairs["robot_body_names"].astype(str)])
        else:
            parts = None
        return {
            "source_name": str(np.asarray(pairs["source_name"])),
            "robot_name": str(np.asarray(pairs["robot_name"])),
            "source_points": source_pts,
            "robot_points": robot_pts,
            "source_target": None,
            "robot_target": None,
            "parts": parts,
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
    parts = None
    if args.color_by == "robot-body" and data is not None and f"face_body_ids_{rob_i}" in data:
        faces = data[f"mesh_faces_{rob_i}"]
        verts = data[f"mesh_vertices_{rob_i}"]
        body_ids = data[f"face_body_ids_{rob_i}"]
        body_names = data[f"body_names_{rob_i}"].astype(str) if f"body_names_{rob_i}" in data else None
        d = robot_pts[:, None, :] - verts[None, :, :]
        nearest_v = np.argmin(np.einsum("ijk,ijk->ij", d, d), axis=1)
        # Face whose first vertex is nearest is a cheap proxy; use vertex→face majority.
        face_for_v = np.full(len(verts), -1, dtype=np.int32)
        face_for_v[faces[:, 0]] = np.arange(len(faces), dtype=np.int32)
        face_ids = face_for_v[nearest_v]
        valid = face_ids >= 0
        parts = np.full(len(robot_pts), "torso", dtype=object)
        bids = np.where(valid, body_ids[face_ids.clip(min=0)], -1)
        if body_names is not None:
            parts[valid] = [robot_body_to_part(body_names[int(b)] if 0 <= int(b) < len(body_names) else "") for b in bids[valid]]
        else:
            parts[valid] = ["torso"] * int(valid.sum())

    return {
        "source_name": source_name,
        "robot_name": robot_name,
        "source_points": source_pts,
        "robot_points": robot_pts,
        "source_target": source_target if args.show_target else None,
        "robot_target": robot_target if args.show_target else None,
        "parts": parts,
        "source_offset": source_offset,
        "paired": paired,
    }


def resolve_parts(payload: dict, smplx_joints: np.ndarray | None) -> np.ndarray:
    if payload["parts"] is not None:
        return payload["parts"]
    pts = payload["source_points"]
    if smplx_joints is not None:
        joints = smplx_joints - payload["source_offset"][None, :]
        return assign_parts_from_smplx_joints(pts, joints)
    return assign_parts_heuristic(pts)


def side_by_side(source: np.ndarray, robot: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    src = source.copy()
    rob = robot.copy()
    gap = 0.25 + float(src[:, 0].max() - src[:, 0].min()) * 0.15
    offset = float(src[:, 0].max() - rob[:, 0].min()) + gap
    rob[:, 0] = rob[:, 0] + offset
    return src, rob, offset


def plot_matplotlib(payload: dict, parts: np.ndarray, args) -> None:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    src, rob, _ = side_by_side(payload["source_points"], payload["robot_points"])
    colors = colors_from_parts(parts)
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(src[:, 0], src[:, 2], src[:, 1], c=colors, s=args.point_size, depthshade=False, label=payload["source_name"])
    ax.scatter(rob[:, 0], rob[:, 2], rob[:, 1], c=colors, s=args.point_size, depthshade=False, label=payload["robot_name"])
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
                color=colors[i, :3],
                alpha=0.25,
                linewidth=0.4,
            )
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=PART_COLORS[name], markersize=8, label=name)
        for name in PART_NAMES
    ]
    ax.legend(handles=handles, loc="upper right")
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


def plot_trimesh(payload: dict, parts: np.ndarray, args) -> None:
    import trimesh

    src, rob, _ = side_by_side(payload["source_points"], payload["robot_points"])
    colors = (colors_from_parts(parts) * 255).astype(np.uint8)
    geoms = [
        trimesh.PointCloud(src, colors=colors),
        trimesh.PointCloud(rob, colors=colors),
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
    joints = None
    if args.color_by == "source-part":
        joints = load_smplx_rest_joints(args.smplx_model.resolve(), args.gender)
        if joints is None:
            print("  using geometric body-part heuristic (SMPL-X joints unavailable)")
    parts = resolve_parts(payload, joints)
    counts = {name: int(np.sum(parts == name)) for name in PART_NAMES}
    print(f"  pair  {payload['source_name']} ↔ {payload['robot_name']}  slots={len(parts)}")
    print("  parts " + "  ".join(f"{k}={v}" for k, v in counts.items()))
    if args.backend == "trimesh":
        plot_trimesh(payload, parts, args)
    else:
        plot_matplotlib(payload, parts, args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
