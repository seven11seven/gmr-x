#!/usr/bin/env python3
"""Retarget HiPHI BVH onto a robot and play it in MuJoCo with scene objects.

Standalone CLI (does not import the web app). Uses an IK JSON from
``general_motion_retargeting/ik_configs`` and HiPHI clips under ``datas/hiphi``.

Each clip folder contains ``motion_actor.bvh``, ``metadata.json`` (which mesh),
and ``object_tracks/<id>.csv`` (object 6D). Mirrored clips use ``*__mirror.obj``.
Object meshes live in ``datas/hiphi/object_meshes`` (centimetres, Y-up). Tracks
are metres / Y-up / xyzw. Poses are remapped with the same ``zxy`` axis order as
``BVHParser`` — ``(x, y, z) → (z, x, y)`` — so the object shares the robot frame.
Do **not** use vis_mesh's Rx(+90°) here; that convention does not match the BVH.

SMPL-X↔object stays in the stretch-only world: tracks unscaled, mesh
``0.01`` (cm→m). Robot↔object XY follows GMR Hips
(``root_pos * [sh, sh, sv]`` on X/Y). Size and extra pose come from IK
JSON ``object_robot`` (``mesh_scale``, ``pos_scale``, ``pos_offset``).
The robot mesh is never scaled.

Eval viewer (side by side):
- Robot + Hips-XY objects (config size / offset).
- Stretched SMPL-X + unscaled objects, translated so the two contact
  drawings do not overlap. Red = SMPL-X↔object, green = robot-thigh↔object.

Examples::

    python scripts/hiphi_to_robot.py --list
    python scripts/hiphi_to_robot.py
    python scripts/hiphi_to_robot.py --clip Change_posture-sit_down_0002
    python scripts/hiphi_to_robot.py --clip Change_posture-sit_down_0002__mirror
    python scripts/hiphi_to_robot.py --object-category chair --max-frames 300
    python scripts/hiphi_to_robot.py --clip Posture-sit_0004 --no-smplx
    python scripts/hiphi_to_robot.py --clip Change_posture-sit_down_0002 --smplx-offset 3 0 0
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(_PROJECT_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "tools"))

import mujoco as mj
import mujoco.viewer as mjv

from retargeting_visualizer.visualize_retargeting import load_bvh_all_frames

from general_motion_retargeting.utils.smplx_from_mixamo import (
    SMPLX_JOINT_LEFT_ANKLE,
    SMPLX_JOINT_RIGHT_ANKLE,
    MixamoToSMPLX,
    write_obj,
)

DEFAULT_HIPHI_ROOT = _PROJECT_ROOT / "datas" / "hiphi"
DEFAULT_MESH_DIR = DEFAULT_HIPHI_ROOT / "object_meshes"
DEFAULT_IK = _PROJECT_ROOT / "general_motion_retargeting" / "ik_configs" / "hiphi_sit.json"
DEFAULT_XML = _PROJECT_ROOT / "assets" / "D20_29dof" / "mjcf" / "D20.xml"
DEFAULT_SMPLX = _PROJECT_ROOT / "assets" / "smpl-x"
DEFAULT_CLIP = "Posture-sit_0015"
MESH_SCALE = 0.01  # OBJ centimetres → metres
DEFAULT_CORRESPONDENCE = (
    _PROJECT_ROOT
    / "output"
    / "correspondence_d20"
    / "correspondence_contact_smplx_neutral_d20.npz"
)
# Mixamo bones whose GMR scale matches sitting / thigh–object contact.
LEG_SCALE_BONES = ("LeftUpLeg", "RightUpLeg", "LeftLeg", "RightLeg")
# Same extras as scripts/bvh_to_smplx_viewer.py (world metres after height scale).
LEG_EXTRA_THIGH_M = 0.04
LEG_EXTRA_SHIN_M = 0.03
_LEFT_FOOT_NAMES = ("LeftFoot", "LeftAnkle")
_RIGHT_FOOT_NAMES = ("RightFoot", "RightAnkle")
_HEAD_NAMES = ("Head", "Neck")
# Same remap as BVHParser(axis_order="zxy"): Mixamo/HiPHI (x, y, z) → (z, x, y).
# vis_mesh.py uses Rx(+90°) for a static gallery; that does NOT match retargeted BVH.
_AXIS_IDX = (2, 0, 1)
_P = np.array(
    [[0.0, 0.0, 1.0],
     [1.0, 0.0, 0.0],
     [0.0, 1.0, 0.0]],
    dtype=np.float64,
)
_P_ROT = R.from_matrix(_P)

_CATEGORY_RGBA = {
    "ball": (0.90, 0.22, 0.22, 1.0),
    "scoccerball": (0.95, 0.95, 0.95, 1.0),
    "bench": (0.55, 0.38, 0.22, 1.0),
    "bottle": (0.35, 0.70, 0.85, 1.0),
    "box": (0.78, 0.62, 0.32, 1.0),
    "bucket": (0.40, 0.55, 0.40, 1.0),
    "chair": (0.72, 0.45, 0.28, 1.0),
    "clothrack": (0.55, 0.55, 0.58, 1.0),
    "mop": (0.45, 0.35, 0.25, 1.0),
    "stepstool": (0.65, 0.50, 0.30, 1.0),
    "table": (0.50, 0.36, 0.22, 1.0),
    "trashbin": (0.30, 0.45, 0.35, 1.0),
}
_DEFAULT_RGBA = (0.70, 0.70, 0.70, 1.0)


@dataclass
class HiphiObject:
    object_id: str
    mesh_id: str
    category: str
    mesh_path: Path
    track_path: Path
    pos_zup: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    pos_smplx: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    quat_wxyz: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))
    mesh_scale_xyz: np.ndarray = field(
        default_factory=lambda: np.full(3, MESH_SCALE, dtype=np.float64)
    )
    mesh_scale_smplx: np.ndarray = field(
        default_factory=lambda: np.full(3, MESH_SCALE, dtype=np.float64)
    )
    mocap_id: int = -1
    ref_mocap_id: int = -1


@dataclass
class HiphiClip:
    motion_id: str
    clip_dir: Path
    bvh_path: Path
    meta_path: Path
    meta: dict
    objects: list[HiphiObject]


@dataclass
class ObjectRobotConfig:
    """IK JSON ``object_robot``: robot-column object size and pose tweak.

    World origin after GMR Hips XY::

        pos' = pos * [hips_sh * pos_scale_x, hips_sh * pos_scale_y, pos_scale_z]
               + pos_offset

    Mesh is authored Y-up centimetres::

        mesh_scale_xyz = 0.01 * mesh_scale
    """

    mesh_scale: np.ndarray = field(
        default_factory=lambda: np.ones(3, dtype=np.float64)
    )
    pos_scale: np.ndarray = field(
        default_factory=lambda: np.ones(3, dtype=np.float64)
    )
    pos_offset: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )


def _vec3(value, default: tuple[float, float, float], name: str) -> np.ndarray:
    if value is None:
        return np.asarray(default, dtype=np.float64)
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size == 1:
        arr = np.full(3, float(arr[0]), dtype=np.float64)
    if arr.size != 3:
        raise ValueError(f"object_robot.{name} must be a scalar or length-3, got {value!r}")
    return arr


def parse_object_robot_config(ik_config: dict) -> ObjectRobotConfig:
    raw = ik_config.get("object_robot") or {}
    return ObjectRobotConfig(
        mesh_scale=_vec3(raw.get("mesh_scale"), (1.0, 1.0, 1.0), "mesh_scale"),
        pos_scale=_vec3(raw.get("pos_scale"), (1.0, 1.0, 1.0), "pos_scale"),
        pos_offset=_vec3(raw.get("pos_offset"), (0.0, 0.0, 0.0), "pos_offset"),
    )


@dataclass
class SmplxOverlay:
    verts: np.ndarray
    anchor: np.ndarray
    obj_path: Path
    mesh_id: int = -1
    mocap_id: int = -1
    lookat: np.ndarray | None = None
    body_world: np.ndarray | None = None
    distances: np.ndarray | None = None
    object_ids: np.ndarray | None = None
    object_points_world: np.ndarray | None = None
    contact_threshold: float = 0.10


# ---------------------------------------------------------------------------
#  Discovery
# ---------------------------------------------------------------------------

def discover_clips(hiphi_root: Path) -> list[HiphiClip]:
    clips: list[HiphiClip] = []
    for meta_path in sorted(hiphi_root.rglob("metadata.json")):
        bvh = meta_path.parent / "motion_actor.bvh"
        if not bvh.is_file():
            continue
        try:
            clips.append(load_clip_record(meta_path, hiphi_root))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  skip {meta_path}: {exc}")
    return clips


def load_clip_record(meta_path: Path, hiphi_root: Path) -> HiphiClip:
    with open(meta_path) as fh:
        meta = json.load(fh)
    clip_dir = meta_path.parent
    motion_id = str(meta.get("motion_id") or clip_dir.name)
    objects = []
    for spec in meta.get("objects") or []:
        obj = _object_from_meta(spec, clip_dir, hiphi_root)
        if obj is not None:
            objects.append(obj)
    bvh = clip_dir / "motion_actor.bvh"
    return HiphiClip(
        motion_id=motion_id,
        clip_dir=clip_dir,
        bvh_path=bvh,
        meta_path=meta_path,
        meta=meta,
        objects=objects,
    )


def _object_from_meta(spec: dict, clip_dir: Path, hiphi_root: Path) -> Optional[HiphiObject]:
    object_id = str(spec.get("object_id") or spec.get("mesh_id") or "")
    mesh_id = str(spec.get("mesh_id") or object_id)
    category = str(spec.get("object_category") or mesh_id.split("_", 1)[0]).lower()
    mesh_rel = spec.get("mesh_path") or f"object_meshes/{mesh_id}.obj"
    track_rel = spec.get("trajectory_path") or f"object_tracks/{object_id}.csv"

    mesh_path = _resolve_data_path(mesh_rel, clip_dir, hiphi_root)
    if mesh_path is None:
        mesh_path = DEFAULT_MESH_DIR / f"{mesh_id}.obj"
    if not mesh_path.is_file():
        print(f"  [warn] missing mesh for {object_id}: {mesh_path}")
        return None

    track_path = clip_dir / track_rel
    if not track_path.is_file():
        track_path = clip_dir / "object_tracks" / f"{object_id}.csv"
    if not track_path.is_file():
        print(f"  [warn] missing track for {object_id}: {track_path}")
        return None

    return HiphiObject(
        object_id=object_id or mesh_id,
        mesh_id=mesh_id,
        category=category,
        mesh_path=mesh_path,
        track_path=track_path,
    )


def _resolve_data_path(rel: str, clip_dir: Path, hiphi_root: Path) -> Optional[Path]:
    candidates = [
        Path(rel),
        clip_dir / rel,
        hiphi_root / rel,
        hiphi_root / Path(rel).name,
        DEFAULT_MESH_DIR / Path(rel).name,
    ]
    for path in candidates:
        if path.is_file():
            return path.resolve()
    return None


def resolve_clip(
    clips: list[HiphiClip],
    clip_query: Optional[str],
    bvh_path: Optional[Path],
    object_category: Optional[str],
    hiphi_root: Path,
) -> HiphiClip:
    if bvh_path is not None:
        bvh_path = bvh_path.resolve()
        if not bvh_path.is_file():
            raise FileNotFoundError(f"BVH not found: {bvh_path}")
        meta = bvh_path.parent / "metadata.json"
        if meta.is_file():
            return load_clip_record(meta, hiphi_root)
        return HiphiClip(
            motion_id=bvh_path.parent.name,
            clip_dir=bvh_path.parent,
            bvh_path=bvh_path,
            meta_path=meta,
            meta={},
            objects=[],
        )

    query = (clip_query or "").strip()
    if query:
        matches = [
            c for c in clips
            if query == c.motion_id
            or query in c.motion_id
            or query in str(c.clip_dir)
        ]
        if not matches:
            raise FileNotFoundError(
                f"no HiPHI clip matching {query!r}. Use --list to see available clips."
            )
        return matches[0]

    preferred = [c for c in clips if c.motion_id == DEFAULT_CLIP]
    if preferred:
        return preferred[0]

    cat = (object_category or "chair").lower()
    for clip in clips:
        if any(obj.category == cat for obj in clip.objects):
            return clip
    if clips:
        return clips[0]
    raise FileNotFoundError(f"no HiPHI clips under {hiphi_root}")


def filter_objects(
    objects: list[HiphiObject],
    category: Optional[str],
) -> list[HiphiObject]:
    if not category:
        return list(objects)
    cat = category.lower()
    kept = [o for o in objects if o.category == cat or cat in o.object_id.lower()]
    if kept:
        return kept
    names = ", ".join(f"{o.category}:{o.object_id}" for o in objects) or "(none)"
    print(f"  [warn] no object matches --object-category {category!r} "
          f"(clip has: {names}). Showing all objects.")
    return list(objects)


def print_clip_table(clips: list[HiphiClip]) -> None:
    if not clips:
        print("No clips found.")
        return
    print(f"{'motion_id':<42} {'objects':<28} {'frames':>7} {'fps':>6}")
    print("-" * 88)
    for clip in clips:
        objs = ",".join(f"{o.mesh_id}" for o in clip.objects) or "-"
        frames = clip.meta.get("frame_count", "?")
        fps = clip.meta.get("fps", "?")
        if isinstance(fps, float):
            fps = f"{fps:.1f}"
        print(f"{clip.motion_id:<42} {objs:<28} {str(frames):>7} {str(fps):>6}")
    print(f"\n{len(clips)} clip(s)")


# ---------------------------------------------------------------------------
#  Object tracks  (Y-up metres, xyzw) → same MuJoCo frame as BVHParser
# ---------------------------------------------------------------------------

def hiphi_pose_to_mujoco(pos_yup: np.ndarray, quat_xyzw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map a HiPHI Y-up pose into the BVHParser / robot frame.

    Mesh vertices stay in the authored local Y-up frame. Only the world pose is
    remapped: ``v_mujoco = P @ (R_yup @ v_local)`` with ``P`` the zxy permutation.
    """
    pos = np.asarray(pos_yup, dtype=np.float64)
    pos_out = pos[list(_AXIS_IDX)]
    quat_wxyz = (_P_ROT * R.from_quat(quat_xyzw)).as_quat(scalar_first=True)
    return pos_out, quat_wxyz


def gmr_height_ratio(ik_config: dict, actual_human_height: float) -> float:
    assumed = float(ik_config.get("human_height_assumption", 1.8))
    return float(actual_human_height) / assumed if assumed > 1e-8 else 1.0


def gmr_effective_scales(
    ik_config: dict,
    actual_human_height: float,
    bone_names: tuple[str, ...] | list[str],
) -> tuple[float, float]:
    """Runtime Mixamo scale: JSON table × height ratio, averaged over bones.

    Same numbers ``scale_human_data`` applies in the Hips local frame.
    """
    ratio = gmr_height_ratio(ik_config, actual_human_height)
    table_h = ik_config.get("human_scale_table_horizontal") or ik_config.get("human_scale_table") or {}
    table_v = ik_config.get("human_scale_table_vertical") or ik_config.get("human_scale_table") or table_h
    hs: list[float] = []
    vs: list[float] = []
    for name in bone_names:
        if name in table_h and name in table_v:
            hs.append(float(table_h[name]) * ratio)
            vs.append(float(table_v[name]) * ratio)
        elif name in table_h:
            hs.append(float(table_h[name]) * ratio)
            vs.append(float(table_h[name]) * ratio)
    if not hs:
        root = str(ik_config.get("human_root_name") or "Hips")
        if len(tuple(bone_names)) == 1 and str(tuple(bone_names)[0]) == root:
            return ratio, ratio
        return gmr_effective_scales(ik_config, actual_human_height, (root,))
    return float(np.mean(hs)), float(np.mean(vs))


def hips_object_scales(ik_config: dict, actual_human_height: float) -> tuple[float, float]:
    """BVH Hips scale: ``(actual_human_height / assumed) * JSON Hips v/h``."""
    root = str(ik_config.get("human_root_name") or "Hips")
    return gmr_effective_scales(ik_config, actual_human_height, (root,))


def leg_object_scales(ik_config: dict, actual_human_height: float) -> tuple[float, float]:
    """Effective GMR scale of Mixamo thighs/shins (sitting contact)."""
    return gmr_effective_scales(ik_config, actual_human_height, LEG_SCALE_BONES)


def scale_objects_like_hips(objects: list[HiphiObject], sh: float, sv: float) -> None:
    """Match the scaled BVH Hips: ``pos * [sh, sh, sv]``, size accordingly.

    After zxy, mesh local Y is world up, so local mesh scale is (sh, sv, sh)
    on top of centimetres→metres.
    """
    s_pos = np.array([sh, sh, sv], dtype=np.float64)
    s_mesh = np.array(
        [MESH_SCALE * sh, MESH_SCALE * sv, MESH_SCALE * sh],
        dtype=np.float64,
    )
    for obj in objects:
        if obj.pos_zup.size:
            obj.pos_zup = obj.pos_zup * s_pos
        obj.mesh_scale_xyz = s_mesh.copy()


def scale_objects_to_robot(
    objects: list[HiphiObject],
    hips_sh: float,
    cfg: ObjectRobotConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Robot-column objects. SMPL-X copies (``pos_smplx``) stay unscaled.

    World XY follows GMR Hips ``root_pos * [sh, sh, sv]``. ``cfg.pos_scale``
    and ``cfg.pos_offset`` are extra tweaks. Mesh is ``0.01 * cfg.mesh_scale``
    in the authored local frame (Y-up).
    """
    s_pos = np.array(
        [
            float(hips_sh) * float(cfg.pos_scale[0]),
            float(hips_sh) * float(cfg.pos_scale[1]),
            float(cfg.pos_scale[2]),
        ],
        dtype=np.float64,
    )
    offset = np.asarray(cfg.pos_offset, dtype=np.float64).reshape(3)
    s_mesh = MESH_SCALE * np.asarray(cfg.mesh_scale, dtype=np.float64).reshape(3)
    for obj in objects:
        if obj.pos_zup.size:
            obj.pos_zup = obj.pos_zup * s_pos + offset
        obj.mesh_scale_xyz = s_mesh.copy()
    return s_pos, offset, s_mesh


def map_smplx_object_points_to_robot(
    pts_world: np.ndarray,
    objects: list[HiphiObject],
    point_counts: list[int],
    mesh_mul: np.ndarray,
) -> np.ndarray:
    """Move contact samples from stretch-world objects onto the robot objects.

    Contact is computed with ``pos_smplx`` and uniform ``0.01`` mesh.
    ``mesh_mul`` is ``cfg.mesh_scale`` (local extra on top of that 0.01).
    """
    pts = np.asarray(pts_world, dtype=np.float64)
    out = pts.copy()
    mul = np.asarray(mesh_mul, dtype=np.float64).reshape(3)
    start = 0
    n_frames = int(pts.shape[0])
    for obj, n in zip(objects, point_counts):
        sl = slice(start, start + int(n))
        start += int(n)
        pos0 = np.asarray(
            obj.pos_smplx if obj.pos_smplx.size else obj.pos_zup,
            dtype=np.float64,
        )
        pos1 = np.asarray(obj.pos_zup, dtype=np.float64)
        quats = np.asarray(obj.quat_wxyz, dtype=np.float64)
        n_use = min(n_frames, len(pos0), len(pos1), len(quats))
        for fi in range(n_use):
            rot = R.from_quat(quats[fi], scalar_first=True).as_matrix()
            local = (pts[fi, sl] - pos0[fi]) @ rot
            local = local * mul
            out[fi, sl] = local @ rot.T + pos1[fi]
    return out


def load_object_track(
    csv_path: Path,
    n_frames: int,
    frame_origin: int,
    convert_y_up: bool,
) -> tuple[np.ndarray, np.ndarray]:
    by_frame: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            fi = int(float(row["frame"]))
            pos = np.array([float(row["px"]), float(row["py"]), float(row["pz"])], dtype=np.float64)
            quat_xyzw = np.array(
                [float(row["qx"]), float(row["qy"]), float(row["qz"]), float(row["qw"])],
                dtype=np.float64,
            )
            if convert_y_up:
                pos, quat = hiphi_pose_to_mujoco(pos, quat_xyzw)
            else:
                quat = np.array(
                    [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
                    dtype=np.float64,
                )
            nrm = np.linalg.norm(quat)
            if nrm > 1e-12:
                quat = quat / nrm
            else:
                quat = np.array([1.0, 0.0, 0.0, 0.0])
            by_frame[fi] = (pos, quat)

    if not by_frame:
        raise ValueError(f"empty object track: {csv_path}")

    keys = np.array(sorted(by_frame), dtype=int)
    pos_out = np.zeros((n_frames, 3), dtype=np.float64)
    quat_out = np.zeros((n_frames, 4), dtype=np.float64)
    last_pos, last_quat = by_frame[int(keys[0])]
    for i in range(n_frames):
        src = frame_origin + i
        if src in by_frame:
            last_pos, last_quat = by_frame[src]
        else:
            # nearest recorded frame (tracks are usually dense at 90 Hz)
            idx = int(np.argmin(np.abs(keys - src)))
            last_pos, last_quat = by_frame[int(keys[idx])]
        pos_out[i] = last_pos
        quat_out[i] = last_quat
    return pos_out, quat_out


def rgba_of(obj: HiphiObject) -> tuple[float, float, float, float]:
    return _CATEGORY_RGBA.get(obj.category.lower(), _DEFAULT_RGBA)


def _rgba_str(rgba: tuple[float, float, float, float]) -> str:
    return " ".join(f"{c:.4f}" for c in rgba)


def _bvh_named_pos(human_data: dict, names: tuple[str, ...]) -> np.ndarray:
    for name in names:
        pose = human_data.get(name)
        if pose is not None:
            return np.asarray(pose[0], dtype=np.float64)
    raise KeyError(f"frame has none of {names}")


def scale_mesh_to_skeleton_head(driver: MixamoToSMPLX, human_data: dict) -> tuple[float, float, float]:
    """Uniform scale so the posed mesh crown hits the skeleton Head after ankle-pin."""
    driver.set_scale(1.0, 1.0)
    verts, joints = driver.drive(human_data)
    target_h = float(_bvh_named_pos(human_data, _HEAD_NAMES)[2]
                     - _bvh_named_pos(human_data, _LEFT_FOOT_NAMES)[2])
    posed_h = float(verts[:, 2].max() - joints[SMPLX_JOINT_LEFT_ANKLE][2])
    if posed_h <= 1e-6:
        raise RuntimeError("SMPL-X posed crown-to-ankle height is degenerate")
    scale = target_h / posed_h
    driver.set_scale(scale, scale)
    return scale, target_h, posed_h


def skin_processed_smplx(
    human_frames: list[dict],
    smplx_model: Path,
    gender: str,
) -> tuple[np.ndarray, np.ndarray, MixamoToSMPLX]:
    """Drive the Mixamo→SMPL-X pipeline used by ``bvh_to_smplx_viewer``.

    ``human_frames`` must be the BVH *before* GMR scale/offset. Those IK
    alignments exist to match robot joints and should not move the mesh.
    """
    driver = MixamoToSMPLX(smplx_model, gender=gender, ext="pkl")
    fit = driver.fit_bone_scales(human_frames[0])
    print(
        f"  SMPL-X bone-fit ({fit['mode']})  "
        f"leg×{fit['leg']:.3f}  torso×{fit['torso']:.3f}  arm×{fit['arm']:.3f}"
    )
    scale, target_h, posed_h = scale_mesh_to_skeleton_head(driver, human_frames[0])
    print(
        f"  SMPL-X gender={driver.gender}  scale={scale:.4f}  "
        f"(Head−ankle={target_h:.3f}m / posed crown−ankle={posed_h:.3f}m)"
    )
    if LEG_EXTRA_THIGH_M or LEG_EXTRA_SHIN_M:
        driver.lengthen_leg_segments(
            human_frames[0],
            thigh_m=LEG_EXTRA_THIGH_M,
            shin_m=LEG_EXTRA_SHIN_M,
        )
        print(
            f"  SMPL-X leg extra  thigh+{LEG_EXTRA_THIGH_M * 100:.0f}cm  "
            f"shin+{LEG_EXTRA_SHIN_M * 100:.0f}cm"
        )

    n = len(human_frames)
    first_verts, first_joints = driver.drive(human_frames[0])
    world_verts = np.empty((n, first_verts.shape[0], 3), dtype=np.float32)
    smplx_lfoot = np.zeros((n, 3), dtype=np.float64)
    smplx_rfoot = np.zeros((n, 3), dtype=np.float64)
    bvh_lfoot = np.zeros((n, 3), dtype=np.float64)
    bvh_rfoot = np.zeros((n, 3), dtype=np.float64)
    world_verts[0] = first_verts
    smplx_lfoot[0] = first_joints[SMPLX_JOINT_LEFT_ANKLE]
    smplx_rfoot[0] = first_joints[SMPLX_JOINT_RIGHT_ANKLE]
    bvh_lfoot[0] = _bvh_named_pos(human_frames[0], _LEFT_FOOT_NAMES)
    bvh_rfoot[0] = _bvh_named_pos(human_frames[0], _RIGHT_FOOT_NAMES)
    for i in tqdm(range(1, n), desc="smplx", unit="frame"):
        verts_i, joints_i = driver.drive(human_frames[i])
        world_verts[i] = verts_i
        smplx_lfoot[i] = joints_i[SMPLX_JOINT_LEFT_ANKLE]
        smplx_rfoot[i] = joints_i[SMPLX_JOINT_RIGHT_ANKLE]
        bvh_lfoot[i] = _bvh_named_pos(human_frames[i], _LEFT_FOOT_NAMES)
        bvh_rfoot[i] = _bvh_named_pos(human_frames[i], _RIGHT_FOOT_NAMES)

    smplx_mid = 0.5 * (smplx_lfoot + smplx_rfoot)
    bvh_mid = 0.5 * (bvh_lfoot + bvh_rfoot)
    origin = np.column_stack([smplx_mid[:, 0], smplx_mid[:, 1], smplx_lfoot[:, 2]])
    target = np.column_stack([bvh_mid[:, 0], bvh_mid[:, 1], bvh_lfoot[:, 2]])
    local_verts = np.ascontiguousarray(world_verts - origin[:, None, :], dtype=np.float32)
    print(f"  SMPL-X pin XY=ankle-midpoint  Z=LeftFoot  "
          f"offset0={np.round((target - origin)[0], 3).tolist()}")
    return local_verts, target, driver


# ---------------------------------------------------------------------------
#  Combined MuJoCo scene: robot XML + mocap object meshes + SMPL-X
# ---------------------------------------------------------------------------

def _add_mocap_mesh_body(
    world: ET.Element,
    body_name: str,
    mesh_name: str,
    rgba: str,
    group: str,
) -> None:
    body = ET.SubElement(world, "body", name=body_name, mocap="true")
    ET.SubElement(
        body, "inertial",
        pos="0 0 0", mass="1", diaginertia="1 1 1",
    )
    ET.SubElement(
        body, "geom",
        name=f"{body_name}_geom",
        type="mesh",
        mesh=mesh_name,
        rgba=rgba,
        contype="0",
        conaffinity="0",
        condim="1",
        group=group,
    )


def build_scene_xml(
    robot_xml: Path,
    objects: list[HiphiObject],
    smplx_obj: Path | None = None,
) -> str:
    tree = ET.parse(robot_xml)
    root = tree.getroot()

    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    rel_meshdir = compiler.get("meshdir", ".")
    compiler.set("meshdir", str((robot_xml.parent / rel_meshdir).resolve()))

    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    world = root.find("worldbody")
    if world is None:
        world = ET.SubElement(root, "worldbody")

    used_names = {name for el in root.iter() if (name := el.get("name"))}
    if smplx_obj is not None:
        used_names.update({"smplx_mesh", "smplx_human", "smplx_human_geom"})

    for obj in objects:
        body_name = _unique_name(f"obj_{obj.object_id}", used_names)
        mesh_name = _unique_name(f"mesh_{obj.mesh_id}", used_names)
        sx, sy, sz = (float(x) for x in obj.mesh_scale_xyz)
        ET.SubElement(
            asset, "mesh",
            name=mesh_name,
            file=str(obj.mesh_path.resolve()),
            scale=f"{sx} {sy} {sz}",
            maxhullvert="32",
        )
        _add_mocap_mesh_body(world, body_name, mesh_name, _rgba_str(rgba_of(obj)), "1")
        obj._body_name = body_name  # type: ignore[attr-defined]
        obj._mesh_name = mesh_name  # type: ignore[attr-defined]
        obj._ref_body_name = None  # type: ignore[attr-defined]
        obj._ref_mesh_name = None  # type: ignore[attr-defined]
        if smplx_obj is not None:
            ref_mesh = _unique_name(f"mesh_{obj.mesh_id}_smplx", used_names)
            rx, ry, rz = (float(x) for x in obj.mesh_scale_smplx)
            ET.SubElement(
                asset, "mesh",
                name=ref_mesh,
                file=str(obj.mesh_path.resolve()),
                scale=f"{rx} {ry} {rz}",
                maxhullvert="32",
            )
            ref_name = _unique_name(f"obj_{obj.object_id}_smplx", used_names)
            _add_mocap_mesh_body(world, ref_name, ref_mesh, _rgba_str(rgba_of(obj)), "3")
            obj._ref_body_name = ref_name  # type: ignore[attr-defined]
            obj._ref_mesh_name = ref_mesh  # type: ignore[attr-defined]

    if smplx_obj is not None:
        ET.SubElement(
            asset, "mesh",
            name="smplx_mesh",
            file=str(Path(smplx_obj).resolve()),
            scale="1 1 1",
            maxhullvert="32",
        )
        _add_mocap_mesh_body(
            world, "smplx_human", "smplx_mesh", "0.92 0.72 0.58 1", "2",
        )

    return ET.tostring(root, encoding="unicode")


def _unique_name(base: str, used: set[str]) -> str:
    name = base.replace(" ", "_")
    i = 1
    while name in used:
        i += 1
        name = f"{base}_{i}"
    used.add(name)
    return name


def compile_scene(xml_str: str) -> tuple[mj.MjModel, Path]:
    tmp = tempfile.NamedTemporaryFile(
        prefix="hiphi_scene_", suffix=".xml", delete=False,
    )
    tmp_path = Path(tmp.name)
    tmp.write(xml_str.encode("utf-8"))
    tmp.close()
    try:
        model = mj.MjModel.from_xml_path(str(tmp_path))
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return model, tmp_path


def bind_mocap_ids(model: mj.MjModel, objects: list[HiphiObject]) -> None:
    for obj in objects:
        body_name = getattr(obj, "_body_name", f"obj_{obj.object_id}")
        body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise RuntimeError(f"mocap body {body_name!r} missing from compiled model")
        mid = int(model.body_mocapid[body_id])
        if mid < 0:
            raise RuntimeError(f"body {body_name!r} is not mocap")
        obj.mocap_id = mid


def bind_ref_mocap_ids(model: mj.MjModel, objects: list[HiphiObject]) -> None:
    for obj in objects:
        ref_name = getattr(obj, "_ref_body_name", None)
        if not ref_name:
            obj.ref_mocap_id = -1
            continue
        body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, ref_name)
        if body_id < 0:
            raise RuntimeError(f"mocap body {ref_name!r} missing from compiled model")
        mid = int(model.body_mocapid[body_id])
        if mid < 0:
            raise RuntimeError(f"body {ref_name!r} is not mocap")
        obj.ref_mocap_id = mid


def bind_smplx_overlay(model: mj.MjModel, overlay: SmplxOverlay) -> None:
    mesh_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_MESH, "smplx_mesh")
    if mesh_id < 0:
        raise RuntimeError("compiled scene is missing smplx_mesh")
    body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "smplx_human")
    if body_id < 0:
        raise RuntimeError("compiled scene is missing smplx_human")
    mocap_id = int(model.body_mocapid[body_id])
    if mocap_id < 0:
        raise RuntimeError("smplx_human is not a mocap body")
    overlay.mesh_id = int(mesh_id)
    overlay.mocap_id = mocap_id
    _pin_visual_mesh_frame(model, "smplx_human_geom", overlay.verts[0])


def _pin_visual_mesh_frame(model: mj.MjModel, geom_name: str, verts: np.ndarray) -> None:
    """Undo MuJoCo's compile-time mesh inertia frame on a visual-only geom."""
    gid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, geom_name)
    if gid < 0:
        raise RuntimeError(f"geom {geom_name!r} missing")
    mid = int(model.geom_dataid[gid])
    if mid < 0:
        raise RuntimeError(f"geom {geom_name!r} is not a mesh")
    model.geom_pos[gid] = 0.0
    model.geom_quat[gid] = np.array([1.0, 0.0, 0.0, 0.0])
    adr = int(model.mesh_vertadr[mid])
    nvert = int(model.mesh_vertnum[mid])
    verts = np.asarray(verts, dtype=model.mesh_vert.dtype)
    if verts.shape[0] != nvert:
        raise ValueError(f"SMPL-X vert count {verts.shape[0]} != mesh {nvert}")
    model.mesh_vert[adr:adr + nvert] = verts


# ---------------------------------------------------------------------------
#  Retarget
# ---------------------------------------------------------------------------

def register_custom_robot(xml_path: Path) -> str:
    from general_motion_retargeting.params import ROBOT_BASE_DICT, ROBOT_XML_DICT

    key = "_hiphi_robot"
    ROBOT_XML_DICT[key] = str(xml_path)
    ROBOT_BASE_DICT[key] = "base_link"
    return key


def retarget_frames(
    retargeter,
    human_frames: list,
    object_contact=None,
) -> tuple[np.ndarray, list[tuple[np.ndarray | None, np.ndarray | None]]]:
    from general_motion_retargeting.utils.smplx_object_contact import contact_frame_dict

    qpos_list = []
    contact_pairs = []
    n_active = []
    for i, human_data in enumerate(tqdm(human_frames, desc="retarget", unit="frame")):
        frame = None
        if object_contact is not None:
            frame = contact_frame_dict(
                object_contact,
                i,
                object_contact["body_points"],
                object_contact["sides"],
            )
        qpos = retargeter.retarget(human_data, object_contact_frame=frame)
        qpos_list.append(np.asarray(qpos, dtype=np.float64).copy())
        task = getattr(retargeter, "object_contact_task", None)
        if task is not None and task.n_active > 0:
            robot_pts = task._active_world_points(retargeter.configuration).copy()
            obj_pts = task.object_points.copy()
            contact_pairs.append((robot_pts, obj_pts))
            n_active.append(task.n_active)
        else:
            contact_pairs.append((None, None))
            n_active.append(0)
    if n_active:
        arr = np.asarray(n_active, dtype=np.int32)
        print(
            f"  object-contact active pts min/mean/max="
            f"{int(arr.min())}/{float(arr.mean()):.1f}/{int(arr.max())}"
        )
    return np.stack(qpos_list, axis=0), contact_pairs


def prepare_thigh_object_contact(
    human_frames: list[dict],
    objects: list[HiphiObject],
    smplx_model: Path,
    gender: str,
    sh: float,
    sv: float,
    *,
    align_to_head: bool,
    use_height: float,
    correspondence_path: Path | None,
    object_samples: int = 1024,
    snap_threshold: float = 0.005,
    contact_threshold: float = 0.10,
    seed: int = 0,
) -> dict:
    """Stretched SMPL-X correspondence slots vs object samples (thigh only)."""
    from bvh_to_smplx_viewer import compute_stretched_smplx_motion
    from general_motion_retargeting.utils.smplx_object_contact import (
        compute_object_contact,
        sample_object_local_points,
    )
    from general_motion_retargeting.utils.smplx_robot_correspondence import (
        load_correspondence_contact,
        skin_source_slots,
        thigh_slot_index,
    )

    driver = MixamoToSMPLX(smplx_model, gender=gender, ext="pkl")
    pairs = None
    slot_ids = None
    slot_sides = None
    slot_world = None
    if correspondence_path is not None:
        pairs = load_correspondence_contact(correspondence_path)
        slot_ids, slot_sides = thigh_slot_index(pairs, driver._lbs)
        n_left = int(np.sum(slot_sides == 0))
        n_right = int(np.sum(slot_sides == 1))
        print(
            f"  correspondence  {pairs['path']}  "
            f"slots={pairs['n_slots']}  thigh={len(slot_ids)} (L={n_left} R={n_right})"
        )

    motion = compute_stretched_smplx_motion(
        driver,
        human_frames,
        fit_bones=True,
        align_to_head=align_to_head,
        use_height=None if align_to_head else use_height,
        extra_thigh_m=LEG_EXTRA_THIGH_M,
        extra_shin_m=LEG_EXTRA_SHIN_M,
        verbose=True,
    )
    body_world = np.asarray(motion["vertices"], dtype=np.float32)
    local_verts = np.asarray(motion["vertices_local"], dtype=np.float32)
    anchor = np.asarray(motion["target"], dtype=np.float64)
    lookat = np.asarray(motion["lookat"], dtype=np.float64)
    if pairs is not None:
        slot_world = skin_source_slots(body_world, pairs, slot_ids)
        print("  SMPL-X↔object contact: stretch only, thigh correspondence slots")
    else:
        print("  SMPL-X↔object contact: stretch only (eval mesh, no IK slots)")

    obj_local_parts = []
    for i, obj in enumerate(objects):
        pts = sample_object_local_points(
            obj.mesh_path,
            obj.mesh_scale_xyz,
            object_samples,
            seed + 17 * (i + 1),
        )
        obj_local_parts.append(pts)
        print(f"  {obj.object_id}: {len(pts)} object surface pts")
    slot_contact = None
    if slot_world is not None:
        slot_contact = compute_object_contact(
            slot_world, objects, obj_local_parts, snap_threshold
        )
    full_contact = compute_object_contact(
        body_world, objects, obj_local_parts, snap_threshold
    )
    vis_active = (full_contact["distances"] <= contact_threshold).sum(axis=1)
    if slot_contact is not None:
        slot_distances = slot_contact["distances"]
        active = (slot_distances <= contact_threshold).sum(axis=1)
        print(
            f"  thigh-slot contact  threshold={contact_threshold:.3f}m  "
            f"min={float(slot_distances.min()):.4f}  "
            f"p50={float(np.percentile(slot_distances, 50)):.4f}  "
            f"active min/mean/max={int(active.min())}/{float(active.mean()):.2f}/{int(active.max())}"
        )
    print(
        f"  full-body vis contact  "
        f"active min/mean/max="
        f"{int(vis_active.min())}/{float(vis_active.mean()):.2f}/{int(vis_active.max())}"
    )
    ik_contact = None
    if slot_contact is not None:
        ik_contact = {
            "object_points_world": slot_contact["object_points_world"],
            "object_point_counts": [int(len(p)) for p in obj_local_parts],
            "distances": slot_contact["distances"],
            "object_ids": slot_contact["object_ids"],
            "pair_vectors": slot_contact["pair_vectors"],
            "snapped_count": slot_contact["snapped_count"],
            "body_points": slot_world.astype(np.float32),
            "sides": slot_sides,
            "slot_ids": slot_ids,
        }
    vis = {
        "local_verts": np.ascontiguousarray(local_verts, dtype=np.float32),
        "anchor": np.asarray(anchor, dtype=np.float64),
        "lookat": np.asarray(lookat, dtype=np.float64),
        "faces": motion["faces"],
        "body_world": np.asarray(body_world, dtype=np.float32),
        "distances": full_contact["distances"],
        "object_ids": full_contact["object_ids"],
        "object_points_world": full_contact["object_points_world"],
        "contact_threshold": float(contact_threshold),
    }
    return {"ik": ik_contact, "vis": vis, "correspondence": pairs}


def resolve_human_height(
    mode: str | float,
    meta: dict,
    bvh_height: float,
) -> tuple[float, str]:
    if isinstance(mode, float):
        return mode, f"cli={mode:.3f}m"
    if mode == "bvh":
        return float(bvh_height), f"bvh={bvh_height:.3f}m"
    height_cm = (meta.get("actor_metadata") or {}).get("height_cm")
    if height_cm:
        h = float(height_cm) / 100.0
        return h, f"metadata={h:.3f}m ({height_cm} cm)"
    return float(bvh_height), f"bvh-fallback={bvh_height:.3f}m"


# ---------------------------------------------------------------------------
#  Viewer
# ---------------------------------------------------------------------------

def _append_contact_line(
    scn,
    p_body: np.ndarray,
    p_obj: np.ndarray,
    rgba: np.ndarray,
) -> None:
    ident = np.eye(3, dtype=np.float64).flatten()
    if scn.ngeom >= scn.maxgeom:
        return
    mj.mjv_initGeom(
        scn.geoms[scn.ngeom],
        type=mj.mjtGeom.mjGEOM_SPHERE,
        size=[0.012, 0.0, 0.0],
        pos=p_body,
        mat=ident,
        rgba=rgba,
    )
    scn.ngeom += 1
    delta = p_obj - p_body
    dist = float(np.linalg.norm(delta))
    if dist < 1e-6 or scn.ngeom >= scn.maxgeom:
        return
    z_axis = delta / dist
    ref = np.array([1.0, 0.0, 0.0]) if abs(z_axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x_axis = np.cross(z_axis, ref)
    x_axis /= np.linalg.norm(x_axis) + 1e-12
    y_axis = np.cross(z_axis, x_axis)
    bone_mat = np.column_stack([x_axis, y_axis, z_axis]).reshape(-1)
    mj.mjv_initGeom(
        scn.geoms[scn.ngeom],
        type=mj.mjtGeom.mjGEOM_CAPSULE,
        size=[0.004, dist * 0.5, 0.0],
        pos=(p_body + p_obj) * 0.5,
        mat=bone_mat,
        rgba=rgba,
    )
    scn.ngeom += 1


def _draw_robot_contact_pairs(viewer, robot_pts, obj_pts) -> None:
    if robot_pts is None or obj_pts is None:
        return
    rgba = np.array([0.15, 0.85, 0.35, 0.95], dtype=np.float64)
    n = min(len(robot_pts), len(obj_pts))
    for i in range(n):
        _append_contact_line(
            viewer.user_scn,
            np.asarray(robot_pts[i], dtype=np.float64),
            np.asarray(obj_pts[i], dtype=np.float64),
            rgba,
        )


def _draw_smplx_object_contacts(
    viewer,
    body_pts: np.ndarray,
    obj_pts: np.ndarray,
    distances: np.ndarray,
    threshold: float,
    max_points: int,
    offset: np.ndarray,
) -> None:
    active = np.where(np.asarray(distances) <= threshold)[0]
    if active.size == 0:
        return
    if max_points > 0 and active.size > max_points:
        active = active[np.argsort(distances[active])[:max_points]]
    shift = np.asarray(offset, dtype=np.float64).reshape(3)
    for idx in active:
        t = float(np.clip(distances[idx] / max(threshold, 1e-8), 0.0, 1.0))
        rgba = np.array([1.0, 0.15 + 0.7 * t, 0.10, 0.95], dtype=np.float64)
        _append_contact_line(
            viewer.user_scn,
            np.asarray(body_pts[idx], dtype=np.float64) + shift,
            np.asarray(obj_pts[idx], dtype=np.float64) + shift,
            rgba,
        )


def _upload_mesh_verts(viewer, model: mj.MjModel, mesh_id: int, verts: np.ndarray) -> None:
    adr = int(model.mesh_vertadr[mesh_id])
    nvert = int(model.mesh_vertnum[mesh_id])
    if verts.shape[0] != nvert:
        raise ValueError(f"SMPL-X vert count {verts.shape[0]} != mesh {nvert}")
    model.mesh_vert[adr:adr + nvert] = np.asarray(verts, dtype=model.mesh_vert.dtype)
    viewer.update_mesh(int(mesh_id))


def launch_viewer(
    model: mj.MjModel,
    qpos: np.ndarray,
    objects: list[HiphiObject],
    fps: float,
    follow_camera: bool,
    title: str,
    smplx: SmplxOverlay | None = None,
    contact_pairs: list | None = None,
    smplx_offset: np.ndarray | None = None,
    max_vis_contacts: int = 128,
) -> None:
    data = mj.MjData(model)
    n_frames = int(qpos.shape[0])
    if n_frames == 0:
        raise RuntimeError("no retargeted frames to play")
    if qpos.shape[1] != model.nq:
        raise RuntimeError(
            f"qpos width {qpos.shape[1]} != model.nq {model.nq} "
            "(robot XML used for IK must match the viewer robot)"
        )

    try:
        base_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "base_link")
    except Exception:
        base_id = -1

    ident_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    offset = np.zeros(3, dtype=np.float64)
    if smplx_offset is not None:
        offset = np.asarray(smplx_offset, dtype=np.float64).reshape(3)
    state = {
        "current": 0,
        "playing": True,
        "speed": 1.0,
        "smplx": smplx is not None,
    }

    def key_callback(keycode: int) -> None:
        if keycode == 263:  # LEFT
            state["current"] = max(state["current"] - 1, 0)
        elif keycode == 262:  # RIGHT
            state["current"] = min(state["current"] + 1, n_frames - 1)
        elif keycode == 265:  # UP
            state["current"] = min(state["current"] + 10, n_frames - 1)
        elif keycode == 264:  # DOWN
            state["current"] = max(state["current"] - 10, 0)
        elif keycode == 268:  # HOME
            state["current"] = 0
        elif keycode == 269:  # END
            state["current"] = n_frames - 1
        elif keycode == 32:  # SPACE
            state["playing"] = not state["playing"]
        elif keycode == 49:
            state["speed"] = 0.25
        elif keycode == 50:
            state["speed"] = 0.5
        elif keycode == 51:
            state["speed"] = 1.0
        elif keycode == 52:
            state["speed"] = 2.0
        elif keycode == 53:
            state["speed"] = 4.0
        elif keycode in (82, 114):  # R / r
            state["current"] = 0
        elif keycode in (77, 109) and smplx is not None:  # M / m
            state["smplx"] = not state["smplx"]
            print(f"  SMPL-X mesh {'on' if state['smplx'] else 'off'}")
            return
        else:
            return
        print(f"  frame {state['current'] + 1}/{n_frames}  "
              f"{'play' if state['playing'] else 'pause'}  x{state['speed']:g}")

    print(f"\n{title}")
    print("  Controls: ← → frame   ↑ ↓ ±10   Space play/pause   1-5 speed   "
          "M SMPL-X   R restart   Esc close")
    obj_names = ", ".join(f"{o.object_id}({o.category})" for o in objects) or "(none)"
    print(f"  Robot column: optimized robot + objects (Hips XY + object_robot)  ({obj_names})")
    if smplx is not None:
        print(
            f"  SMPL-X column: stretched mesh + unscaled objects (cm→m), "
            f"offset={offset.tolist()}  ({smplx.verts.shape[1]} verts, M toggles)"
        )
        print("  Contact: red SMPL-X→object  |  green robot-thigh→object")
    elif contact_pairs is not None:
        print("  Contact overlay: green robot-thigh → object")

    with mjv.launch_passive(
        model, data,
        show_left_ui=True,
        show_right_ui=False,
        key_callback=key_callback,
    ) as viewer:
        viewer.opt.frame = mj.mjtFrame.mjFRAME_NONE
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 135
        viewer.cam.distance = 5.5 if smplx is not None else 3.5

        prev = -1
        last_time = time.time()
        while viewer.is_running():
            now = time.time()
            if state["playing"]:
                dt = now - last_time
                advance = int(dt * fps * state["speed"])
                if advance > 0:
                    advance = min(advance, max(1, int(fps * 0.5)))
                    state["current"] = (state["current"] + advance) % n_frames
                    last_time = now
            else:
                last_time = now

            fi = state["current"]
            viewer.opt.geomgroup[2] = 1 if state["smplx"] else 0
            viewer.opt.geomgroup[3] = 1 if state["smplx"] else 0
            if fi != prev:
                prev = fi
                if smplx is not None and fi < len(smplx.verts):
                    _upload_mesh_verts(viewer, model, smplx.mesh_id, smplx.verts[fi])
                with viewer.lock():
                    data.qpos[:] = qpos[fi]
                    for obj in objects:
                        if obj.mocap_id < 0 or fi >= len(obj.pos_zup):
                            continue
                        data.mocap_pos[obj.mocap_id] = obj.pos_zup[fi]
                        data.mocap_quat[obj.mocap_id] = obj.quat_wxyz[fi]
                    if smplx is not None:
                        if fi < len(smplx.anchor):
                            data.mocap_pos[smplx.mocap_id] = smplx.anchor[fi] + offset
                            data.mocap_quat[smplx.mocap_id] = ident_quat
                        for obj in objects:
                            if obj.ref_mocap_id < 0:
                                continue
                            src = obj.pos_smplx if obj.pos_smplx.size else obj.pos_zup
                            if fi >= len(src):
                                continue
                            data.mocap_pos[obj.ref_mocap_id] = src[fi] + offset
                            data.mocap_quat[obj.ref_mocap_id] = obj.quat_wxyz[fi]
                    mj.mj_forward(model, data)

                viewer.user_scn.ngeom = 0
                if (
                    state["smplx"]
                    and smplx is not None
                    and smplx.distances is not None
                    and smplx.object_ids is not None
                    and smplx.object_points_world is not None
                    and smplx.body_world is not None
                    and fi < len(smplx.distances)
                ):
                    ids = smplx.object_ids[fi]
                    _draw_smplx_object_contacts(
                        viewer,
                        smplx.body_world[fi],
                        smplx.object_points_world[fi, ids],
                        smplx.distances[fi],
                        smplx.contact_threshold,
                        max_vis_contacts,
                        offset,
                    )
                if contact_pairs is not None and fi < len(contact_pairs):
                    robot_pts, obj_pts = contact_pairs[fi]
                    _draw_robot_contact_pairs(viewer, robot_pts, obj_pts)

                if follow_camera and base_id >= 0:
                    lookat = np.asarray(data.xpos[base_id], dtype=np.float64).copy()
                    if smplx is not None and fi < len(smplx.anchor):
                        lookat = 0.5 * (lookat + smplx.anchor[fi] + offset)
                    viewer.cam.lookat[:] = lookat

            viewer.sync()
            if not state["playing"]:
                time.sleep(0.02)


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def _parse_height(value: str):
    lowered = value.strip().lower()
    if lowered in ("auto", "meta", "bvh"):
        return "meta" if lowered == "auto" else lowered
    return float(value)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Retarget HiPHI motion and eval robot vs stretched SMPL-X contact in MuJoCo.",
    )
    p.add_argument("--hiphi-root", type=Path, default=DEFAULT_HIPHI_ROOT,
                   help="Root of HiPHI data (default: datas/hiphi)")
    p.add_argument("--clip", type=str, default=None,
                   help=f"motion_id or path substring (default: {DEFAULT_CLIP} if present)")
    p.add_argument("--bvh", type=Path, default=None,
                   help="Direct path to motion_actor.bvh (overrides --clip)")
    p.add_argument("--ik-config", type=Path, default=DEFAULT_IK,
                   help="IK JSON (default: ik_configs/hiphi_sit.json)")
    p.add_argument("--xml", type=Path, default=DEFAULT_XML,
                   help="Robot MJCF (default: assets/D20_29dof/mjcf/D20.xml)")
    p.add_argument("--object-category", type=str, default=None,
                   help="Keep objects of this category (e.g. chair). Default: all in the clip.")
    p.add_argument("--list", action="store_true",
                   help="List discovered clips and exit")
    p.add_argument("--start", type=int, default=None, help="First BVH frame (inclusive)")
    p.add_argument("--end", type=int, default=None, help="Last BVH frame (exclusive)")
    p.add_argument("--max-frames", type=int, default=None, help="Cap number of frames after --start/--end")
    p.add_argument("--human-height", type=_parse_height, default="meta",
                   help="'meta' (actor height_cm), 'bvh' (detected), or a metre value")
    p.add_argument("--bvh-scale", type=float, default=None,
                   help="Override IK config bvh_scale (default: from JSON, usually 0.01)")
    p.add_argument("--no-y-up", action="store_true",
                   help="Do not apply BVHParser zxy axis remap to object tracks")
    p.add_argument("--smplx-model", type=Path, default=DEFAULT_SMPLX,
                   help="SMPL-X folder or SMPLX_*.pkl (default: assets/smpl-x)")
    p.add_argument("--gender", type=str, default="auto",
                   choices=("auto", "neutral", "female", "male"),
                   help="SMPL-X gender. auto uses clip metadata, then Neutral.")
    p.add_argument("--no-smplx", action="store_true",
                   help="Do not show the stretched SMPL-X + object eval column")
    p.add_argument("--smplx-offset", type=float, nargs=3, default=(2.5, 0.0, 0.0),
                   metavar=("X", "Y", "Z"),
                   help="World translation of the SMPL-X+object column (default: 2.5 0 0)")
    p.add_argument("--no-follow-camera", action="store_true",
                   help="Do not keep the camera on the robot / SMPL-X midpoint")
    p.add_argument("--no-viewer", action="store_true",
                   help="Retarget (and optionally save) without opening MuJoCo")
    p.add_argument("--save-pkl", type=Path, default=None,
                   help="Write retargeted qpos + object poses to a pickle")
    p.add_argument("--object-contact-cost", type=float, default=8.0,
                   help="Mink cost for thigh–object contact (0 disables)")
    p.add_argument("--object-contact-threshold", type=float, default=0.10,
                   help="SMPL-X thigh contact distance threshold in metres")
    p.add_argument("--object-contact-snap", type=float, default=0.005,
                   help="Snap SMPL-X–object pairs closer than this to zero offset")
    p.add_argument("--object-contact-max-points", type=int, default=32,
                   help="Max active thigh contacts per IK frame")
    p.add_argument("--object-contact-samples", type=int, default=1024,
                   help="Object surface samples used for contact queries")
    p.add_argument("--correspondence", type=Path, default=DEFAULT_CORRESPONDENCE,
                   help="SMPL-X↔robot correspondence_contact npz (AE slot pairs)")
    p.add_argument("--max-vis-contacts", type=int, default=128,
                   help="Max SMPL-X↔object contact lines drawn per frame")
    p.add_argument("--no-object-contact", action="store_true",
                   help="Do not add the mink object-contact task")
    p.add_argument("--no-show-contact", action="store_true",
                   help="Do not draw contact lines in the viewer")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    hiphi_root = args.hiphi_root.resolve()
    ik_path = args.ik_config.resolve()
    xml_path = args.xml.resolve()

    if not hiphi_root.is_dir():
        print(f"ERROR: HiPHI root not found: {hiphi_root}")
        return 1
    if not ik_path.is_file():
        print(f"ERROR: IK config not found: {ik_path}")
        return 1
    if not xml_path.is_file():
        print(f"ERROR: robot XML not found: {xml_path}")
        return 1

    print(f"Scanning {hiphi_root} …")
    clips = discover_clips(hiphi_root)
    if args.list:
        print_clip_table(clips)
        return 0
    if not clips and args.bvh is None:
        print(f"ERROR: no clips with motion_actor.bvh + metadata.json under {hiphi_root}")
        return 1

    clip = resolve_clip(clips, args.clip, args.bvh, args.object_category, hiphi_root)
    objects = filter_objects(clip.objects, args.object_category)
    smplx_model = args.smplx_model.resolve()
    gender = args.gender
    if gender == "auto":
        gender = str((clip.meta.get("actor_metadata") or {}).get("gender") or "neutral")
    if not args.no_smplx and not args.no_viewer and not smplx_model.exists():
        print(f"ERROR: SMPL-X model not found: {smplx_model}")
        return 1

    with open(ik_path) as fh:
        ik_config = json.load(fh)
    bvh_scale = args.bvh_scale if args.bvh_scale is not None else float(ik_config.get("bvh_scale", 0.01))

    print(f"Clip       : {clip.motion_id}")
    print(f"  dir      : {clip.clip_dir}")
    print(f"  bvh      : {clip.bvh_path}")
    print(f"  text     : {clip.meta.get('text_annotation', '')}")
    print(f"IK config  : {ik_path}")
    print(f"Robot XML  : {xml_path}")
    if args.no_smplx:
        print("SMPL-X     : (off)")
    else:
        print(f"SMPL-X     : {smplx_model}  gender={gender}")
    obj_desc = ", ".join(
        f"{o.object_id} mesh={o.mesh_path.name} ({o.category}) track={o.track_path.name}"
        for o in objects
    ) or "(none)"
    print(f"Objects    : {obj_desc}")
    print(f"  metadata : {clip.meta_path}")

    print("\n[1/4] Loading BVH …")
    human_frames, bvh_height, frame_time, _parser = load_bvh_all_frames(
        clip.bvh_path,
        start=args.start,
        end=args.end,
        scale=bvh_scale,
    )
    frame_origin = int(args.start or 0)
    if args.max_frames and args.max_frames < len(human_frames):
        human_frames = human_frames[: args.max_frames]
        print(f"  limited to {len(human_frames)} frames (--max-frames)")
    fps = (1.0 / frame_time) if frame_time > 0 else float(clip.meta.get("fps") or 90.0)
    print(f"  {len(human_frames)} frames, fps≈{fps:.1f}, bvh height={bvh_height:.2f}m")

    use_height, height_label = resolve_human_height(args.human_height, clip.meta, bvh_height)
    assumed = float(ik_config.get("human_height_assumption", 1.8))
    print(f"  human height {height_label}  assumed={assumed:.2f}m  "
          f"ratio={use_height / assumed:.4f}")

    print("[2/4] Loading object tracks …")
    for obj in objects:
        obj.pos_zup, obj.quat_wxyz = load_object_track(
            obj.track_path,
            n_frames=len(human_frames),
            frame_origin=frame_origin,
            convert_y_up=not args.no_y_up,
        )
        print(f"  {obj.object_id}: mesh={obj.mesh_id} ({obj.mesh_path.name})  "
              f"track={obj.track_path}  "
              f"pos0_raw={np.round(obj.pos_zup[0], 3).tolist()}")
        obj.pos_smplx = np.asarray(obj.pos_zup, dtype=np.float64).copy()
        obj.mesh_scale_xyz = np.full(3, MESH_SCALE, dtype=np.float64)
        obj.mesh_scale_smplx = np.full(3, MESH_SCALE, dtype=np.float64)

    print("[3/4] Retargeting …")
    robot_key = register_custom_robot(xml_path)
    from general_motion_retargeting import GeneralMotionRetargeting as GMR

    retargeter = GMR(
        src_human="bvh_xsens",
        tgt_robot=robot_key,
        actual_human_height=use_height,
        ik_config_path=str(ik_path),
        verbose=False,
    )
    sh, sv = hips_object_scales(ik_config, use_height)
    obj_cfg = parse_object_robot_config(ik_config)
    assumed = float(ik_config.get("human_height_assumption", 1.8))
    ratio = use_height / assumed if assumed > 1e-8 else 1.0
    root = str(ik_config.get("human_root_name") or "Hips")
    json_h = float((ik_config.get("human_scale_table_horizontal") or {}).get(root, sh / max(ratio, 1e-8)))
    json_v = float((ik_config.get("human_scale_table_vertical") or {}).get(root, sv / max(ratio, 1e-8)))
    print(
        f"  JSON {root} (GMR Mixamo IK)  horizontal={json_h:.4f}  "
        f"vertical={json_v:.4f}  height_ratio={ratio:.4f}  → sh={sh:.4f} sv={sv:.4f}"
    )
    print(
        f"  object_robot  mesh_scale={np.round(obj_cfg.mesh_scale, 4).tolist()}  "
        f"pos_scale={np.round(obj_cfg.pos_scale, 4).tolist()}  "
        f"pos_offset={np.round(obj_cfg.pos_offset, 4).tolist()}"
    )
    print(f"  SMPL-X objects  pos×[1, 1, 1]  mesh×{MESH_SCALE}  (stretch world)")

    object_contact = None
    smplx_vis = None
    use_object_contact = (
        not args.no_object_contact
        and float(args.object_contact_cost) > 0.0
        and len(objects) > 0
        and smplx_model.exists()
    )
    need_smplx_prep = (
        len(objects) > 0
        and smplx_model.exists()
        and (use_object_contact or (not args.no_viewer and not args.no_smplx))
    )
    if need_smplx_prep:
        print("[3b/4] SMPL-X thigh–object contact (IK reference + eval mesh) …")
        try:
            payload = prepare_thigh_object_contact(
                human_frames,
                objects,
                smplx_model,
                gender,
                sh,
                sv,
                align_to_head=(args.human_height == "bvh"),
                use_height=use_height,
                correspondence_path=args.correspondence if use_object_contact else None,
                object_samples=args.object_contact_samples,
                snap_threshold=args.object_contact_snap,
                contact_threshold=args.object_contact_threshold,
            )
        except FileNotFoundError as exc:
            print(f"ERROR: {exc}")
            return 1
        smplx_vis = payload["vis"]
        if use_object_contact:
            from general_motion_retargeting.utils.smplx_robot_correspondence import (
                bind_robot_slots,
            )

            object_contact = payload["ik"]
            bound = bind_robot_slots(
                retargeter.model,
                payload["correspondence"],
                object_contact["slot_ids"],
            )
            err = np.asarray(bound["bind_error"], dtype=np.float64)
            print(
                f"  robot slot bind  center={bound['center_body']}  "
                f"mean={float(err.mean()):.4f}m  p95={float(np.percentile(err, 95)):.4f}m"
            )
            retargeter.enable_object_contact(
                bound["geom_ids"],
                bound["local_pos"],
                object_contact["sides"],
                cost=args.object_contact_cost,
                threshold=args.object_contact_threshold,
                max_points=args.object_contact_max_points,
            )
        else:
            print("  object-contact IK off; SMPL-X mesh is still shown for eval")
    elif args.no_object_contact:
        print("  object contact: off (--no-object-contact)")
    elif not objects:
        print("  object contact: off (no objects in clip)")
    elif float(args.object_contact_cost) <= 0.0:
        print("  object contact: off (cost=0)")

    s_pos, pos_offset, s_mesh = scale_objects_to_robot(objects, sh, obj_cfg)
    print(
        f"  robot objects   origin×{np.round(s_pos, 4).tolist()}  "
        f"+ {np.round(pos_offset, 4).tolist()}  mesh×{np.round(s_mesh, 5).tolist()}"
    )
    for obj in objects:
        print(
            f"  {obj.object_id} robot pos0={np.round(obj.pos_zup[0], 3).tolist()}  "
            f"mesh={np.round(obj.mesh_scale_xyz, 5).tolist()}  "
            f"smplx pos0={np.round(obj.pos_smplx[0], 3).tolist()}"
        )
    if object_contact is not None:
        counts = object_contact.get("object_point_counts")
        if counts:
            object_contact["object_points_world"] = map_smplx_object_points_to_robot(
                object_contact["object_points_world"],
                objects,
                counts,
                obj_cfg.mesh_scale,
            )
        else:
            pts = np.asarray(object_contact["object_points_world"], dtype=np.float64)
            object_contact["object_points_world"] = pts * s_pos + pos_offset

    qpos, contact_pairs = retarget_frames(
        retargeter, human_frames, object_contact=object_contact
    )
    print(f"  qpos {qpos.shape}")

    overlay: SmplxOverlay | None = None
    tmp_obj_path: Path | None = None
    if not args.no_viewer and not args.no_smplx and smplx_vis is not None:
        tmp_obj = tempfile.NamedTemporaryFile(
            prefix="smplx_eval_", suffix=".obj", delete=False,
        )
        tmp_obj_path = Path(tmp_obj.name)
        tmp_obj.close()
        write_obj(tmp_obj_path, smplx_vis["local_verts"][0], smplx_vis["faces"])
        overlay = SmplxOverlay(
            verts=smplx_vis["local_verts"],
            anchor=smplx_vis["anchor"],
            obj_path=tmp_obj_path,
            lookat=smplx_vis["lookat"],
            body_world=None if args.no_show_contact else smplx_vis["body_world"],
            distances=None if args.no_show_contact else smplx_vis["distances"],
            object_ids=None if args.no_show_contact else smplx_vis["object_ids"],
            object_points_world=(
                None if args.no_show_contact else smplx_vis["object_points_world"]
            ),
            contact_threshold=float(smplx_vis["contact_threshold"]),
        )

    if args.save_pkl is not None:
        import pickle
        payload = {
            "motion_id": clip.motion_id,
            "fps": fps,
            "qpos": qpos,
            "ik_config": str(ik_path),
            "robot_xml": str(xml_path),
            "human_height": use_height,
            "hips_scale_horizontal": sh,
            "hips_scale_vertical": sv,
            "object_pos_scale": s_pos,
            "object_pos_offset": pos_offset,
            "object_mesh_scale": s_mesh,
            "objects": [
                {
                    "object_id": o.object_id,
                    "category": o.category,
                    "mesh_path": str(o.mesh_path),
                    "mesh_scale_xyz": o.mesh_scale_xyz,
                    "mesh_scale_smplx": o.mesh_scale_smplx,
                    "pos": o.pos_zup,
                    "pos_smplx": o.pos_smplx,
                    "quat_wxyz": o.quat_wxyz,
                }
                for o in objects
            ],
        }
        args.save_pkl.parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_pkl, "wb") as fh:
            pickle.dump(payload, fh)
        print(f"  saved {args.save_pkl}")

    if args.no_viewer:
        print("Done (--no-viewer).")
        return 0

    print("[4/4] Opening MuJoCo viewer …")
    xml_str = build_scene_xml(
        xml_path, objects,
        smplx_obj=overlay.obj_path if overlay is not None else None,
    )
    model, tmp_xml = compile_scene(xml_str)
    try:
        bind_mocap_ids(model, objects)
        if overlay is not None:
            bind_smplx_overlay(model, overlay)
            bind_ref_mocap_ids(model, objects)
        launch_viewer(
            model=model,
            qpos=qpos,
            objects=objects,
            fps=fps,
            follow_camera=not args.no_follow_camera,
            title=f"{clip.motion_id} → {xml_path.parent.name}  [{ik_path.name}]",
            smplx=overlay,
            contact_pairs=None if args.no_show_contact else (
                contact_pairs if object_contact is not None else None
            ),
            smplx_offset=np.asarray(args.smplx_offset, dtype=np.float64),
            max_vis_contacts=args.max_vis_contacts,
        )
    finally:
        tmp_xml.unlink(missing_ok=True)
        if tmp_obj_path is not None:
            tmp_obj_path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        raise SystemExit(130)
