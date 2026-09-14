#!/usr/bin/env python3
"""Drive an SMPL-X mesh from a HiPHI Mixamo BVH clip and play it in MuJoCo.

The clip is not SMPL-X motion parameters. Joint globals from ``motion_actor.bvh``
are mapped onto the SMPL-X kinematic tree (see
``general_motion_retargeting.utils.smplx_from_mixamo``), then the mesh is
uploaded each frame.

Examples::

    python scripts/bvh_to_smplx_viewer.py
    python scripts/bvh_to_smplx_viewer.py --clip Change_posture-sit_down_0002
    python scripts/bvh_to_smplx_viewer.py --smplx-model assets/smpl-x --max-frames 300
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
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

from general_motion_retargeting.utils.smplx_from_mixamo import (
    SMPLX_JOINT_LEFT_ANKLE,
    SMPLX_JOINT_RIGHT_ANKLE,
    MixamoToSMPLX,
    write_obj,
)
from retargeting_visualizer.visualize_retargeting import get_human_edges, load_bvh_all_frames
from hiphi_to_robot import (
    DEFAULT_HIPHI_ROOT,
    HiphiObject,
    bind_mocap_ids,
    filter_objects,
    load_object_track,
    resolve_clip,
    rgba_of,
    discover_clips,
    _rgba_str,
    _unique_name,
)

DEFAULT_CLIP = "Change_posture-sit_down_0002"
DEFAULT_SMPLX = _PROJECT_ROOT / "assets" / "smpl-x"
MESH_SCALE = 0.01
_LEFT_FOOT_NAMES = ("LeftFoot", "LeftAnkle")
_RIGHT_FOOT_NAMES = ("RightFoot", "RightAnkle")
_FINGER_MARKERS = ("HandIndex", "HandMiddle", "HandRing", "HandPinky", "HandThumb")
# Extra world length on top of Mixamo bone-fit (thigh = hip→knee, shin = knee→ankle).
LEG_EXTRA_THIGH_M = 0.04
LEG_EXTRA_SHIN_M = 0.03


def _is_body_joint(name: str) -> bool:
    return not any(token in name for token in _FINGER_MARKERS)


def _draw_bvh_skeleton(viewer, positions: dict, edges: list[tuple[str, str]]) -> None:
    """Overlay Mixamo/BVH joints (spheres) and bones (capsules) in world frame."""
    scn = viewer.user_scn
    scn.ngeom = 0
    ident = np.eye(3, dtype=np.float64).flatten()
    maxgeom = int(scn.maxgeom)

    def _add(geom_type, size, pos, mat, rgba) -> bool:
        if scn.ngeom >= maxgeom:
            return False
        mj.mjv_initGeom(
            scn.geoms[scn.ngeom],
            type=geom_type,
            size=size,
            pos=np.asarray(pos, dtype=np.float64),
            mat=mat,
            rgba=rgba,
        )
        scn.ngeom += 1
        return True

    for name, pos in positions.items():
        if name == "LeftFoot":
            rgba = [1.0, 0.25, 0.15, 0.95]
            radius = 0.022
        elif name == "Hips":
            rgba = [0.20, 0.75, 1.00, 0.95]
            radius = 0.020
        else:
            rgba = [0.35, 0.95, 0.40, 0.90]
            radius = 0.014
        if not _add(mj.mjtGeom.mjGEOM_SPHERE, [radius, 0.0, 0.0], pos, ident, rgba):
            return

    for parent, child in edges:
        if parent not in positions or child not in positions:
            continue
        p1 = np.asarray(positions[parent], dtype=np.float64)
        p2 = np.asarray(positions[child], dtype=np.float64)
        delta = p2 - p1
        dist = float(np.linalg.norm(delta))
        if dist < 1e-6:
            continue
        z_axis = delta / dist
        ref = np.array([1.0, 0.0, 0.0]) if abs(z_axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        x_axis = np.cross(z_axis, ref)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        bone_mat = np.column_stack([x_axis, y_axis, z_axis]).reshape(-1)
        if not _add(
            mj.mjtGeom.mjGEOM_CAPSULE,
            [0.006, dist * 0.5, 0.0],
            (p1 + p2) * 0.5,
            bone_mat,
            [1.0, 0.55, 0.10, 0.55],
        ):
            return


def _bvh_left_foot(human_data: dict) -> np.ndarray:
    for name in _LEFT_FOOT_NAMES:
        pose = human_data.get(name)
        if pose is not None:
            return np.asarray(pose[0], dtype=np.float64)
    raise KeyError("BVH frame has no LeftFoot / LeftAnkle")


def _bvh_right_foot(human_data: dict) -> np.ndarray:
    for name in _RIGHT_FOOT_NAMES:
        pose = human_data.get(name)
        if pose is not None:
            return np.asarray(pose[0], dtype=np.float64)
    raise KeyError("BVH frame has no RightFoot / RightAnkle")


def _bvh_head(human_data: dict) -> np.ndarray:
    for name in ("Head", "Neck"):
        pose = human_data.get(name)
        if pose is not None:
            return np.asarray(pose[0], dtype=np.float64)
    raise KeyError("BVH frame has no Head / Neck")


def scale_mesh_to_skeleton_head(driver: MixamoToSMPLX, human_data: dict) -> tuple[float, float, float]:
    """Uniform scale so the posed mesh crown hits BVH Head after ankle-pin.

    Height is Head_z − LeftFoot_z on the skeleton, and mesh-crown_z −
    SMPL-X left-ankle_z on the mesh. ``betas`` are left unchanged.
    """
    driver.set_scale(1.0, 1.0)
    verts, joints = driver.drive(human_data)
    target_h = float(_bvh_head(human_data)[2] - _bvh_left_foot(human_data)[2])
    posed_h = float(verts[:, 2].max() - joints[SMPLX_JOINT_LEFT_ANKLE][2])
    if posed_h <= 1e-6:
        raise RuntimeError("SMPL-X posed crown-to-ankle height is degenerate")
    scale = target_h / posed_h
    driver.set_scale(scale, scale)
    return scale, target_h, posed_h


def build_smplx_scene_xml(
    smplx_obj: Path,
    objects: list[HiphiObject],
) -> str:
    root = ET.Element("mujoco", model="smplx_bvh")
    ET.SubElement(
        root, "compiler",
        angle="radian", autolimits="true",
        inertiafromgeom="false",
    )
    ET.SubElement(root, "option", gravity="0 0 0", timestep="0.002")

    visual = ET.SubElement(root, "visual")
    ET.SubElement(
        visual, "headlight",
        diffuse="0.65 0.65 0.65",
        ambient="0.30 0.30 0.30",
        specular="0.20 0.20 0.20",
    )
    ET.SubElement(visual, "rgba", haze="0.15 0.25 0.35 1")

    asset = ET.SubElement(root, "asset")
    ET.SubElement(
        asset, "texture",
        name="grid", type="2d", builtin="checker",
        rgb1="0.22 0.22 0.22", rgb2="0.18 0.18 0.18",
        width="512", height="512",
    )
    ET.SubElement(
        asset, "material",
        name="grid", texture="grid", texrepeat="8 8", reflectance="0.08",
    )
    ET.SubElement(
        asset, "mesh",
        name="smplx_mesh",
        file=str(Path(smplx_obj).resolve()),
        scale="1 1 1",
        maxhullvert="32",
    )
    used_names = {"smplx_mesh", "smplx_human", "floor", "grid"}

    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="0 0 4", dir="0 0 -1", directional="true")
    ET.SubElement(
        world, "geom",
        name="floor", type="plane", size="20 20 0.05",
        material="grid",
    )
    body = ET.SubElement(world, "body", name="smplx_human", mocap="true")
    ET.SubElement(
        body, "inertial",
        pos="0 0 0", mass="1", diaginertia="1 1 1",
    )
    ET.SubElement(
        body, "geom",
        name="smplx_human_geom",
        type="mesh",
        mesh="smplx_mesh",
        rgba="0.92 0.72 0.58 1",
        contype="0",
        conaffinity="0",
        group="1",
    )

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
        mocap = ET.SubElement(world, "body", name=body_name, mocap="true")
        ET.SubElement(
            mocap, "geom",
            name=f"{body_name}_geom",
            type="mesh",
            mesh=mesh_name,
            rgba=_rgba_str(rgba_of(obj)),
            contype="0",
            conaffinity="0",
            group="1",
        )
        obj._body_name = body_name  # type: ignore[attr-defined]

    return ET.tostring(root, encoding="unicode")


def compile_scene(xml_str: str) -> tuple[mj.MjModel, Path]:
    tmp = tempfile.NamedTemporaryFile(
        prefix="smplx_bvh_", suffix=".xml", delete=False,
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


def _pin_visual_mesh_frame(model: mj.MjModel, geom_name: str, verts: np.ndarray) -> None:
    """Undo MuJoCo's compile-time mesh inertia frame.

    The compiler recenters/rotates mesh vertices and stores the inverse on the
    geom. Updating ``mesh_vert`` with our original coords then shows a ~90°
    yaw and a ~1 m lift. Force the geom frame back to identity.
    """
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


def _upload_mesh_verts(viewer, model: mj.MjModel, mesh_id: int, verts: np.ndarray) -> None:
    adr = int(model.mesh_vertadr[mesh_id])
    nvert = int(model.mesh_vertnum[mesh_id])
    if verts.shape[0] != nvert:
        raise ValueError(f"SMPL-X vert count {verts.shape[0]} != mesh {nvert}")
    model.mesh_vert[adr:adr + nvert] = np.asarray(verts, dtype=model.mesh_vert.dtype)
    # launch_passive renders from a GPU copy; update_mesh() uploads on the viewer thread.
    viewer.update_mesh(int(mesh_id))


def launch_viewer(
    model: mj.MjModel,
    verts: np.ndarray,
    objects: list[HiphiObject],
    anchor: np.ndarray,
    lookat: np.ndarray,
    skel_pos: list[dict] | None,
    skel_edges: list[tuple[str, str]],
    fps: float,
    title: str,
    show_skeleton: bool = True,
) -> None:
    data = mj.MjData(model)
    n_frames = int(verts.shape[0])
    if n_frames == 0:
        raise RuntimeError("no SMPL-X frames to play")

    mesh_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_MESH, "smplx_mesh")
    if mesh_id < 0:
        raise RuntimeError("compiled scene is missing smplx_mesh")
    human_body = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "smplx_human")
    if human_body < 0:
        raise RuntimeError("compiled scene is missing smplx_human")
    human_mocap = int(model.body_mocapid[human_body])
    if human_mocap < 0:
        raise RuntimeError("smplx_human is not a mocap body")
    ident_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    state = {
        "current": 0,
        "playing": True,
        "speed": 1.0,
        "skeleton": bool(show_skeleton and skel_pos),
    }

    def key_callback(keycode: int) -> None:
        if keycode == 263:
            state["current"] = max(state["current"] - 1, 0)
        elif keycode == 262:
            state["current"] = min(state["current"] + 1, n_frames - 1)
        elif keycode == 265:
            state["current"] = min(state["current"] + 10, n_frames - 1)
        elif keycode == 264:
            state["current"] = max(state["current"] - 10, 0)
        elif keycode == 268:
            state["current"] = 0
        elif keycode == 269:
            state["current"] = n_frames - 1
        elif keycode == 32:
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
        elif keycode in (82, 114):
            state["current"] = 0
        elif keycode in (72, 104) and skel_pos:  # H / h
            state["skeleton"] = not state["skeleton"]
            state["redraw"] = True
            print(f"  skeleton {'on' if state['skeleton'] else 'off'}")
            return
        else:
            return
        print(f"  frame {state['current'] + 1}/{n_frames}  "
              f"{'play' if state['playing'] else 'pause'}  x{state['speed']:g}")

    print(f"\n{title}")
    print("  Controls: ← → frame   ↑ ↓ ±10   Space play/pause   1-5 speed   H skeleton   R restart   Esc close")
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
        viewer.cam.distance = 3.2

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
            if fi != prev or state.pop("redraw", False):
                prev = fi
                _upload_mesh_verts(viewer, model, mesh_id, verts[fi])
                with viewer.lock():
                    data.mocap_pos[human_mocap] = anchor[fi]
                    data.mocap_quat[human_mocap] = ident_quat
                    for obj in objects:
                        if obj.mocap_id < 0 or fi >= len(obj.pos_zup):
                            continue
                        data.mocap_pos[obj.mocap_id] = obj.pos_zup[fi]
                        data.mocap_quat[obj.mocap_id] = obj.quat_wxyz[fi]
                    mj.mj_forward(model, data)
                if state["skeleton"] and skel_pos is not None:
                    _draw_bvh_skeleton(viewer, skel_pos[fi], skel_edges)
                else:
                    viewer.user_scn.ngeom = 0
                viewer.cam.lookat[:] = lookat[fi]

            viewer.sync()
            if not state["playing"]:
                time.sleep(0.02)


def _parse_height(value: str):
    lowered = value.strip().lower()
    if lowered in ("auto", "meta", "bvh"):
        return "meta" if lowered == "auto" else lowered
    return float(value)


def resolve_human_height(mode, meta: dict, bvh_height: float) -> tuple[float, str]:
    if isinstance(mode, float):
        return mode, f"cli={mode:.3f}m"
    if mode == "bvh":
        return float(bvh_height), f"bvh={bvh_height:.3f}m"
    height_cm = (meta.get("actor_metadata") or {}).get("height_cm")
    if height_cm:
        h = float(height_cm) / 100.0
        return h, f"metadata={h:.3f}m ({height_cm} cm)"
    return float(bvh_height), f"bvh-fallback={bvh_height:.3f}m"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Skin a HiPHI BVH clip onto SMPL-X and view it in MuJoCo.",
    )
    p.add_argument("--hiphi-root", type=Path, default=DEFAULT_HIPHI_ROOT)
    p.add_argument("--clip", type=str, default=DEFAULT_CLIP,
                   help=f"motion_id or path substring (default: {DEFAULT_CLIP})")
    p.add_argument("--bvh", type=Path, default=None,
                   help="Direct path to motion_actor.bvh (overrides --clip)")
    p.add_argument("--smplx-model", type=Path, default=DEFAULT_SMPLX,
                   help="SMPL-X folder or SMPLX_*.pkl/npz file (default: assets/smpl-x)")
    p.add_argument("--gender", type=str, default="auto",
                   choices=("auto", "neutral", "female", "male"),
                   help="SMPL-X gender. auto uses metadata, then the file on disk.")
    p.add_argument("--object-category", type=str, default=None)
    p.add_argument("--list", action="store_true")
    p.add_argument("--start", type=int, default=None)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--human-height", type=_parse_height, default="bvh",
                   help="'bvh' = mesh crown matches skeleton Head; "
                        "'meta' / metres = scale T-pose AABB to that height")
    p.add_argument("--bvh-scale", type=float, default=0.01)
    p.add_argument("--no-objects", action="store_true")
    p.add_argument("--no-skeleton", action="store_true",
                   help="Do not overlay the Mixamo/BVH skeleton")
    p.add_argument("--no-y-up", action="store_true")
    p.add_argument("--no-fit-bones", action="store_true",
                   help="Do not stretch SMPL-X bones to Mixamo lengths")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    hiphi_root = args.hiphi_root.resolve()
    smplx_model = args.smplx_model.resolve()

    if not hiphi_root.is_dir():
        print(f"ERROR: HiPHI root not found: {hiphi_root}")
        return 1
    if not smplx_model.exists():
        print(f"ERROR: SMPL-X model not found: {smplx_model}")
        return 1

    print(f"Scanning {hiphi_root} …")
    clips = discover_clips(hiphi_root)
    if args.list:
        from hiphi_to_robot import print_clip_table
        print_clip_table(clips)
        return 0
    if not clips and args.bvh is None:
        print(f"ERROR: no clips with motion_actor.bvh + metadata.json under {hiphi_root}")
        return 1

    clip = resolve_clip(clips, args.clip, args.bvh, args.object_category, hiphi_root)
    objects = [] if args.no_objects else filter_objects(clip.objects, args.object_category)

    gender = args.gender
    if gender == "auto":
        gender = str((clip.meta.get("actor_metadata") or {}).get("gender") or "neutral")

    print(f"Clip       : {clip.motion_id}")
    print(f"  dir      : {clip.clip_dir}")
    print(f"  bvh      : {clip.bvh_path}")
    print(f"  text     : {clip.meta.get('text_annotation', '')}")
    print(f"SMPL-X     : {smplx_model}  gender={gender}")
    obj_desc = ", ".join(
        f"{o.object_id} mesh={o.mesh_path.name} ({o.category})"
        for o in objects
    ) or "(none)"
    print(f"Objects    : {obj_desc}")

    print("\n[1/4] Loading BVH …")
    human_frames, bvh_height, frame_time, parser = load_bvh_all_frames(
        clip.bvh_path,
        start=args.start,
        end=args.end,
        scale=args.bvh_scale,
    )
    frame_origin = int(args.start or 0)
    if args.max_frames and args.max_frames < len(human_frames):
        human_frames = human_frames[: args.max_frames]
        print(f"  limited to {len(human_frames)} frames (--max-frames)")
    fps = (1.0 / frame_time) if frame_time > 0 else float(clip.meta.get("fps") or 90.0)
    print(f"  {len(human_frames)} frames, fps≈{fps:.1f}, bvh height={bvh_height:.2f}m")

    use_height, height_label = resolve_human_height(args.human_height, clip.meta, bvh_height)
    print(f"  human height {height_label}")

    print("[2/4] Loading SMPL-X and object tracks …")
    driver = MixamoToSMPLX(smplx_model, gender=gender, ext="pkl")
    if not args.no_fit_bones:
        fit = driver.fit_bone_scales(human_frames[0])
        print(
            f"  bone-fit ({fit['mode']}) from frame 0  "
            f"leg×{fit['leg']:.3f}  torso×{fit['torso']:.3f}  arm×{fit['arm']:.3f}"
        )
        for j, name in enumerate(
            ("Hips", "LUpLeg", "RUpLeg", "Spine", "LLeg", "RLeg", "Spine1",
             "LFoot", "RFoot", "Spine2", "LToe", "RToe", "Neck",
             "LShoulder", "RShoulder", "Head", "LArm", "RArm",
             "LForeArm", "RForeArm", "LHand", "RHand")
        ):
            if j == 0:
                continue
            print(
                f"    {name:10s}  ×{fit['scale'][j]:.3f}  "
                f"mixamo={fit['mixamo_len'][j]:.3f}m  smplx={fit['smplx_len'][j]:.3f}m"
            )
    rest_h = driver.rest_height()
    align_to_head = args.human_height == "bvh"
    if align_to_head:
        scale, target_h, posed_h = scale_mesh_to_skeleton_head(driver, human_frames[0])
        print(f"  model gender={driver.gender}  rest_height={rest_h:.3f}m")
        print(f"  scale={scale:.4f}  so mesh crown matches Head "
              f"(Head−ankle={target_h:.3f}m / posed crown−ankle={posed_h:.3f}m)")
    elif rest_h > 1e-6:
        scale = use_height / rest_h
        driver.set_scale(scale, scale)
        print(f"  model gender={driver.gender}  rest_height={rest_h:.3f}m  "
              f"uniform scale={scale:.4f}  (T-pose AABB → {height_label})")
    else:
        print(f"  model gender={driver.gender}  (could not measure rest height)")

    if LEG_EXTRA_THIGH_M or LEG_EXTRA_SHIN_M:
        driver.lengthen_leg_segments(
            human_frames[0],
            thigh_m=LEG_EXTRA_THIGH_M,
            shin_m=LEG_EXTRA_SHIN_M,
        )
        print(
            f"  leg extra  thigh+{LEG_EXTRA_THIGH_M * 100:.0f}cm  "
            f"shin+{LEG_EXTRA_SHIN_M * 100:.0f}cm  "
            f"(after height scale, world metres)",
            flush=True,
        )

    for obj in objects:
        obj.mesh_scale_xyz = np.full(3, MESH_SCALE, dtype=np.float64)
        obj.pos_zup, obj.quat_wxyz = load_object_track(
            obj.track_path,
            n_frames=len(human_frames),
            frame_origin=frame_origin,
            convert_y_up=not args.no_y_up,
        )
        print(f"  {obj.object_id}: mesh={obj.mesh_path.name}  "
              f"pos0={np.round(obj.pos_zup[0], 3).tolist()}")

    print("[3/4] Skinning SMPL-X …")
    first_verts, first_joints = driver.drive(human_frames[0])
    world_verts = np.empty((len(human_frames), first_verts.shape[0], 3), dtype=np.float32)
    smplx_lfoot = np.zeros((len(human_frames), 3), dtype=np.float64)
    smplx_rfoot = np.zeros((len(human_frames), 3), dtype=np.float64)
    bvh_lfoot = np.zeros((len(human_frames), 3), dtype=np.float64)
    bvh_rfoot = np.zeros((len(human_frames), 3), dtype=np.float64)
    lookat = np.zeros((len(human_frames), 3), dtype=np.float64)
    world_verts[0] = first_verts
    smplx_lfoot[0] = first_joints[SMPLX_JOINT_LEFT_ANKLE]
    smplx_rfoot[0] = first_joints[SMPLX_JOINT_RIGHT_ANKLE]
    bvh_lfoot[0] = _bvh_left_foot(human_frames[0])
    bvh_rfoot[0] = _bvh_right_foot(human_frames[0])
    lookat[0] = np.asarray(human_frames[0]["Hips"][0], dtype=np.float64)
    for i in tqdm(range(1, len(human_frames)), desc="smplx", unit="frame"):
        verts_i, joints_i = driver.drive(human_frames[i])
        world_verts[i] = verts_i
        smplx_lfoot[i] = joints_i[SMPLX_JOINT_LEFT_ANKLE]
        smplx_rfoot[i] = joints_i[SMPLX_JOINT_RIGHT_ANKLE]
        bvh_lfoot[i] = _bvh_left_foot(human_frames[i])
        bvh_rfoot[i] = _bvh_right_foot(human_frames[i])
        lookat[i] = np.asarray(human_frames[i]["Hips"][0], dtype=np.float64)

    # Horizontal: midpoint of the two ankles. Vertical: left ankle Z.
    smplx_mid = 0.5 * (smplx_lfoot + smplx_rfoot)
    bvh_mid = 0.5 * (bvh_lfoot + bvh_rfoot)
    origin = np.column_stack([smplx_mid[:, 0], smplx_mid[:, 1], smplx_lfoot[:, 2]])
    target = np.column_stack([bvh_mid[:, 0], bvh_mid[:, 1], bvh_lfoot[:, 2]])
    offset = target - origin
    lookat = lookat + offset
    local_verts = np.ascontiguousarray(world_verts - origin[:, None, :], dtype=np.float32)
    print(
        f"  pin XY=ankle-midpoint  Z=LeftFoot  "
        f"offset0={np.round(offset[0], 3).tolist()}",
        flush=True,
    )
    print(
        f"  bvh_mid0={np.round(bvh_mid[0], 3).tolist()}  "
        f"smplx_mid0={np.round(smplx_mid[0], 3).tolist()}",
        flush=True,
    )
    print(
        f"  bvh_lfoot0={np.round(bvh_lfoot[0], 3).tolist()}  "
        f"smplx_lfoot0={np.round(smplx_lfoot[0], 3).tolist()}",
        flush=True,
    )
    print(
        f"  local z=[{local_verts[0, :, 2].min():.3f}, {local_verts[0, :, 2].max():.3f}]  "
        f"aligned world z=[{(local_verts[0, :, 2] + target[0, 2]).min():.3f}, "
        f"{(local_verts[0, :, 2] + target[0, 2]).max():.3f}]",
        flush=True,
    )

    print("  writing rest mesh …", flush=True)
    tmp_obj = tempfile.NamedTemporaryFile(
        prefix="smplx_rest_", suffix=".obj", delete=False,
    )
    tmp_obj_path = Path(tmp_obj.name)
    tmp_obj.close()
    write_obj(tmp_obj_path, local_verts[0], driver.faces)

    print("[4/4] Compiling MuJoCo scene and opening viewer …", flush=True)
    xml_str = build_smplx_scene_xml(tmp_obj_path, objects)
    model, tmp_xml = compile_scene(xml_str)
    try:
        bind_mocap_ids(model, objects)
        _pin_visual_mesh_frame(model, "smplx_human_geom", local_verts[0])
        skel_edges = [
            (p, c) for p, c in get_human_edges(parser)
            if _is_body_joint(p) and _is_body_joint(c)
        ]
        skel_pos = [
            {
                name: np.asarray(val[0], dtype=np.float64)
                for name, val in frame.items()
                if _is_body_joint(name)
            }
            for frame in human_frames
        ]
        launch_viewer(
            model=model,
            verts=local_verts,
            objects=objects,
            anchor=target,
            lookat=lookat,
            skel_pos=None if args.no_skeleton else skel_pos,
            skel_edges=skel_edges,
            fps=fps,
            title=f"{clip.motion_id} → SMPL-X ({driver.gender})",
            show_skeleton=not args.no_skeleton,
        )
    finally:
        tmp_xml.unlink(missing_ok=True)
        tmp_obj_path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        raise SystemExit(130)
