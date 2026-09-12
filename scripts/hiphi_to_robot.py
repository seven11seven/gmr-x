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

Object position and mesh size then use the same Hips horizontal/vertical
factors as ``GMR.scale_human_data`` (including the human-height ratio):
world XY × ``sh``, world Z × ``sv``. Mesh local Y (authored up) maps to
world Z after ``zxy``, so local scale is ``(sh, sv, sh)`` times centimetres→metres.

Examples::

    python scripts/hiphi_to_robot.py --list
    python scripts/hiphi_to_robot.py
    python scripts/hiphi_to_robot.py --clip Change_posture-sit_down_0002
    python scripts/hiphi_to_robot.py --clip Change_posture-sit_down_0002__mirror
    python scripts/hiphi_to_robot.py --object-category chair --max-frames 300
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
if str(_PROJECT_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "tools"))

import mujoco as mj
import mujoco.viewer as mjv

from retargeting_visualizer.visualize_retargeting import load_bvh_all_frames

DEFAULT_HIPHI_ROOT = _PROJECT_ROOT / "datas" / "hiphi"
DEFAULT_MESH_DIR = DEFAULT_HIPHI_ROOT / "object_meshes"
DEFAULT_IK = _PROJECT_ROOT / "general_motion_retargeting" / "ik_configs" / "hiphi.json"
DEFAULT_XML = _PROJECT_ROOT / "assets" / "D20_29dof" / "mjcf" / "D20.xml"
DEFAULT_CLIP = "Posture-sit_0015"
MESH_SCALE = 0.01  # OBJ centimetres → metres
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
    quat_wxyz: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))
    mesh_scale_xyz: np.ndarray = field(
        default_factory=lambda: np.full(3, MESH_SCALE, dtype=np.float64)
    )
    mocap_id: int = -1


@dataclass
class HiphiClip:
    motion_id: str
    clip_dir: Path
    bvh_path: Path
    meta_path: Path
    meta: dict
    objects: list[HiphiObject]


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


def hips_object_scales(retargeter) -> tuple[float, float]:
    """Hips horizontal / vertical factors actually used on the BVH human."""
    root = retargeter.human_root_name
    sh = float(retargeter.human_scale_table_horizontal[root])
    sv = float(retargeter.human_scale_table_vertical[root])
    return sh, sv


def scale_objects_like_hips(objects: list[HiphiObject], sh: float, sv: float) -> None:
    """Match GMR Hips world scaling: ``pos * [sh, sh, sv]``, size accordingly.

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


# ---------------------------------------------------------------------------
#  Combined MuJoCo scene: robot XML + mocap object meshes
# ---------------------------------------------------------------------------

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

    used_names: set[str] = set()
    for obj in objects:
        body_name = _unique_name(f"obj_{obj.object_id}", used_names)
        mesh_name = _unique_name(f"mesh_{obj.mesh_id}", used_names)
        sx, sy, sz = (float(x) for x in obj.mesh_scale_xyz)
        ET.SubElement(
            asset, "mesh",
            name=mesh_name,
            file=str(obj.mesh_path.resolve()),
            scale=f"{sx} {sy} {sz}",
        )
        body = ET.SubElement(world, "body", name=body_name, mocap="true")
        ET.SubElement(
            body, "geom",
            name=f"{body_name}_geom",
            type="mesh",
            mesh=mesh_name,
            rgba=_rgba_str(rgba_of(obj)),
            contype="0",
            conaffinity="0",
            group="1",
        )
        obj._body_name = body_name  # type: ignore[attr-defined]

    if smplx_obj is not None:
        mesh_name = _unique_name("smplx_mesh", used_names)
        ET.SubElement(
            asset, "mesh",
            name=mesh_name,
            file=str(Path(smplx_obj).resolve()),
            scale="1 1 1",
        )
        body = ET.SubElement(world, "body", name="smplx_human")
        ET.SubElement(
            body, "geom",
            name="smplx_human_geom",
            type="mesh",
            mesh=mesh_name,
            rgba="0.92 0.72 0.58 0.55",
            contype="0",
            conaffinity="0",
            group="2",
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


# ---------------------------------------------------------------------------
#  Retarget
# ---------------------------------------------------------------------------

def register_custom_robot(xml_path: Path) -> str:
    from general_motion_retargeting.params import ROBOT_BASE_DICT, ROBOT_XML_DICT

    key = "_hiphi_robot"
    ROBOT_XML_DICT[key] = str(xml_path)
    ROBOT_BASE_DICT[key] = "base_link"
    return key


def retarget_frames(retargeter, human_frames: list) -> tuple[np.ndarray, list[dict]]:
    qpos_list = []
    human_scaled = []
    for human_data in tqdm(human_frames, desc="retarget", unit="frame"):
        qpos = retargeter.retarget(human_data)
        qpos_list.append(np.asarray(qpos, dtype=np.float64).copy())
        human_scaled.append({
            name: (val[0].copy(), val[1].copy())
            for name, val in retargeter.scaled_human_data.items()
        })
    return np.stack(qpos_list, axis=0), human_scaled


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

def _draw_human_overlay(viewer, human_data: dict) -> None:
    viewer.user_scn.ngeom = 0
    mat = np.eye(3, dtype=np.float64).flatten()
    for _name, (pos, _quat) in human_data.items():
        if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
            break
        geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
        mj.mjv_initGeom(
            geom,
            type=mj.mjtGeom.mjGEOM_SPHERE,
            size=[0.025, 0.0, 0.0],
            pos=np.asarray(pos, dtype=np.float64),
            mat=mat,
            rgba=[1.0, 0.45, 0.12, 0.85],
        )
        viewer.user_scn.ngeom += 1


def _upload_mesh_verts(viewer, model: mj.MjModel, mesh_id: int, verts: np.ndarray) -> None:
    adr = int(model.mesh_vertadr[mesh_id])
    nvert = int(model.mesh_vertnum[mesh_id])
    if verts.shape[0] != nvert:
        raise ValueError(f"SMPL-X vert count {verts.shape[0]} != mesh {nvert}")
    model.mesh_vert[adr:adr + nvert] = np.asarray(verts, dtype=model.mesh_vert.dtype)
    sim = viewer._sim() if hasattr(viewer, "_sim") else None
    con = None
    if sim is not None:
        con = getattr(sim, "con", None) or getattr(sim, "_con", None)
    if con is not None:
        mj.mjr_uploadMesh(model, con, int(mesh_id))


def launch_viewer(
    model: mj.MjModel,
    qpos: np.ndarray,
    objects: list[HiphiObject],
    human_scaled: list[dict],
    fps: float,
    show_human: bool,
    follow_camera: bool,
    title: str,
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

    state = {
        "current": 0,
        "playing": True,
        "speed": 1.0,
        "last_switch": time.time(),
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
        else:
            return
        print(f"  frame {state['current'] + 1}/{n_frames}  "
              f"{'play' if state['playing'] else 'pause'}  x{state['speed']:g}")

    print(f"\n{title}")
    print("  Controls: ← → frame   ↑ ↓ ±10   Space play/pause   1-5 speed   R restart   Esc close")
    obj_names = ", ".join(f"{o.object_id}({o.category})" for o in objects) or "(none)"
    print(f"  Objects: {obj_names}")

    with mjv.launch_passive(
        model, data,
        show_left_ui=True,
        show_right_ui=False,
        key_callback=key_callback,
    ) as viewer:
        viewer.opt.frame = mj.mjtFrame.mjFRAME_NONE
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 135
        viewer.cam.distance = 3.5

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
            if fi != prev:
                prev = fi
                data.qpos[:] = qpos[fi]
                mj.mj_forward(model, data)
                for obj in objects:
                    if obj.mocap_id < 0 or fi >= len(obj.pos_zup):
                        continue
                    data.mocap_pos[obj.mocap_id] = obj.pos_zup[fi]
                    data.mocap_quat[obj.mocap_id] = obj.quat_wxyz[fi]
                mj.mj_forward(model, data)

                if show_human and fi < len(human_scaled):
                    _draw_human_overlay(viewer, human_scaled[fi])
                else:
                    viewer.user_scn.ngeom = 0

                if follow_camera and base_id >= 0:
                    viewer.cam.lookat[:] = data.xpos[base_id]

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
        description="Retarget HiPHI motion with ik_configs and view robot + objects in MuJoCo.",
    )
    p.add_argument("--hiphi-root", type=Path, default=DEFAULT_HIPHI_ROOT,
                   help="Root of HiPHI data (default: datas/hiphi)")
    p.add_argument("--clip", type=str, default=None,
                   help=f"motion_id or path substring (default: {DEFAULT_CLIP} if present)")
    p.add_argument("--bvh", type=Path, default=None,
                   help="Direct path to motion_actor.bvh (overrides --clip)")
    p.add_argument("--ik-config", type=Path, default=DEFAULT_IK,
                   help="IK JSON (default: ik_configs/hiphi.json)")
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
    p.add_argument("--show-human", action="store_true",
                   help="Overlay scaled human joint spheres")
    p.add_argument("--no-follow-camera", action="store_true",
                   help="Do not keep the camera on the robot base")
    p.add_argument("--no-viewer", action="store_true",
                   help="Retarget (and optionally save) without opening MuJoCo")
    p.add_argument("--save-pkl", type=Path, default=None,
                   help="Write retargeted qpos + object poses to a pickle")
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

    with open(ik_path) as fh:
        ik_config = json.load(fh)
    bvh_scale = args.bvh_scale if args.bvh_scale is not None else float(ik_config.get("bvh_scale", 0.01))

    print(f"Clip       : {clip.motion_id}")
    print(f"  dir      : {clip.clip_dir}")
    print(f"  bvh      : {clip.bvh_path}")
    print(f"  text     : {clip.meta.get('text_annotation', '')}")
    print(f"IK config  : {ik_path}")
    print(f"Robot XML  : {xml_path}")
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
    sh, sv = hips_object_scales(retargeter)
    scale_objects_like_hips(objects, sh, sv)
    print(f"  object scale from Hips: horizontal={sh:.4f}  vertical={sv:.4f}")
    print(f"  mesh local scale (cm→m × Hips): "
          f"{MESH_SCALE * sh:.5f} {MESH_SCALE * sv:.5f} {MESH_SCALE * sh:.5f}")
    for obj in objects:
        print(f"  {obj.object_id} pos0_scaled={np.round(obj.pos_zup[0], 3).tolist()}")

    qpos, human_scaled = retarget_frames(retargeter, human_frames)
    print(f"  qpos {qpos.shape}")

    if args.save_pkl is not None:
        import pickle
        payload = {
            "motion_id": clip.motion_id,
            "fps": fps,
            "qpos": qpos,
            "ik_config": str(ik_path),
            "robot_xml": str(xml_path),
            "human_height": use_height,
            "object_scale_horizontal": sh,
            "object_scale_vertical": sv,
            "objects": [
                {
                    "object_id": o.object_id,
                    "category": o.category,
                    "mesh_path": str(o.mesh_path),
                    "mesh_scale_xyz": o.mesh_scale_xyz,
                    "pos": o.pos_zup,
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
    xml_str = build_scene_xml(xml_path, objects)
    model, tmp_xml = compile_scene(xml_str)
    try:
        bind_mocap_ids(model, objects)
        launch_viewer(
            model=model,
            qpos=qpos,
            objects=objects,
            human_scaled=human_scaled,
            fps=fps,
            show_human=args.show_human,
            follow_camera=not args.no_follow_camera,
            title=f"{clip.motion_id} → {xml_path.parent.name}  [{ik_path.name}]",
        )
    finally:
        tmp_xml.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        raise SystemExit(130)
