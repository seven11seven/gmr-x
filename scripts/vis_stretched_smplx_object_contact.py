#!/usr/bin/env python3
"""Human–object surface contact on stretched SMPL-X (no GMR Hips second scale).

Pipeline
--------
1. Drive SMPL-X from HiPHI BVH with the same stretch as
   ``compute_stretched_smplx_points.py`` (bone-fit, Head scale, extra thigh/shin,
   ankle pin).
2. Keep that mesh in the stretch-only world. Objects stay centimetres→metres
   (``0.01``) with unscaled tracks — the SMPL-X↔object relation from before
   the GMR skeleton scale.
3. For each frame, nearest-surface contact (UMR-style): query object points from
   each body point, store distance / index / offset, snap below a threshold.

The MuJoCo viewer is the test: scaled SMPL-X + objects, with contact lines.

Examples::

    python scripts/vis_stretched_smplx_object_contact.py
    python scripts/vis_stretched_smplx_object_contact.py --clip Change_posture-sit_down_0002
    python scripts/vis_stretched_smplx_object_contact.py --max-frames 200 --no-viewer --out output/contact.npz
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
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

from general_motion_retargeting.utils.smplx_from_mixamo import MixamoToSMPLX, write_obj
from general_motion_retargeting.utils.smplx_object_contact import (
    compute_object_contact,
    sample_object_local_points,
)
from retargeting_visualizer.visualize_retargeting import load_bvh_all_frames

from bvh_to_smplx_viewer import (
    DEFAULT_CLIP,
    DEFAULT_SMPLX,
    LEG_EXTRA_SHIN_M,
    LEG_EXTRA_THIGH_M,
    _parse_height,
    _pin_visual_mesh_frame,
    _upload_mesh_verts,
    build_smplx_scene_xml,
    compile_scene,
    compute_stretched_smplx_motion,
    resolve_human_height,
)
from hiphi_to_robot import (
    DEFAULT_HIPHI_ROOT,
    DEFAULT_IK,
    MESH_SCALE,
    bind_mocap_ids,
    discover_clips,
    filter_objects,
    hips_object_scales,
    load_object_track,
    resolve_clip,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hiphi-root", type=Path, default=DEFAULT_HIPHI_ROOT)
    p.add_argument("--clip", type=str, default=DEFAULT_CLIP)
    p.add_argument("--bvh", type=Path, default=None)
    p.add_argument("--smplx-model", type=Path, default=DEFAULT_SMPLX)
    p.add_argument("--gender", type=str, default="auto", choices=("auto", "neutral", "female", "male"))
    p.add_argument("--ik-config", type=Path, default=DEFAULT_IK)
    p.add_argument("--start", type=int, default=None)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--human-height", type=_parse_height, default="bvh")
    p.add_argument("--bvh-scale", type=float, default=0.01)
    p.add_argument("--no-fit-bones", action="store_true")
    p.add_argument("--object-category", type=str, default=None)
    p.add_argument("--object-samples", type=int, default=1024)
    p.add_argument("--body-points", type=int, default=0, help="Subsample SMPL verts; 0 uses all vertices.")
    p.add_argument("--contact-threshold", type=float, default=0.10)
    p.add_argument("--snap-threshold", type=float, default=0.005)
    p.add_argument("--max-vis-contacts", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--no-viewer", action="store_true")
    return p.parse_args()


def _draw_contacts(viewer, body_pts, obj_pts, distances, threshold, max_points) -> None:
    scn = viewer.user_scn
    scn.ngeom = 0
    ident = np.eye(3, dtype=np.float64).flatten()
    active = np.where(distances <= threshold)[0]
    if active.size == 0:
        return
    if max_points > 0 and active.size > max_points:
        active = active[np.argsort(distances[active])[:max_points]]
    for idx in active:
        t = float(np.clip(distances[idx] / max(threshold, 1e-8), 0.0, 1.0))
        rgba = np.array([1.0, 0.15 + 0.7 * t, 0.10, 0.95], dtype=np.float64)
        p_body = np.asarray(body_pts[idx], dtype=np.float64)
        p_obj = np.asarray(obj_pts[idx], dtype=np.float64)
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
            continue
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


def launch_contact_viewer(
    model: mj.MjModel,
    local_verts: np.ndarray,
    objects,
    anchor: np.ndarray,
    lookat: np.ndarray,
    body_world: np.ndarray,
    contact: dict,
    threshold: float,
    max_vis: int,
    fps: float,
    title: str,
) -> None:
    data = mj.MjData(model)
    n_frames = int(local_verts.shape[0])
    mesh_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_MESH, "smplx_mesh")
    human_body = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "smplx_human")
    human_mocap = int(model.body_mocapid[human_body])
    ident_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    state = {"current": 0, "playing": True, "speed": 1.0}

    def key_callback(keycode: int) -> None:
        if keycode == 263:
            state["current"] = max(state["current"] - 1, 0)
        elif keycode == 262:
            state["current"] = min(state["current"] + 1, n_frames - 1)
        elif keycode == 265:
            state["current"] = min(state["current"] + 10, n_frames - 1)
        elif keycode == 264:
            state["current"] = max(state["current"] - 10, 0)
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
        elif keycode in (82, 114):
            state["current"] = 0
        else:
            return
        print(f"  frame {state['current'] + 1}/{n_frames}  "
              f"{'play' if state['playing'] else 'pause'}  x{state['speed']:g}")

    print(f"\n{title}")
    print("  Controls: ← → frame   ↑ ↓ ±10   Space play/pause   1-4 speed   R restart   Esc close")
    print(f"  Contact lines: body→object where dist ≤ {threshold:.3f} m")

    obj_world = contact["object_points_world"]
    distances = contact["distances"]
    object_ids = contact["object_ids"]

    with mjv.launch_passive(model, data, show_left_ui=True, show_right_ui=False, key_callback=key_callback) as viewer:
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
            if fi != prev:
                prev = fi
                _upload_mesh_verts(viewer, model, mesh_id, local_verts[fi])
                with viewer.lock():
                    data.mocap_pos[human_mocap] = anchor[fi]
                    data.mocap_quat[human_mocap] = ident_quat
                    for obj in objects:
                        if obj.mocap_id < 0 or fi >= len(obj.pos_zup):
                            continue
                        data.mocap_pos[obj.mocap_id] = obj.pos_zup[fi]
                        data.mocap_quat[obj.mocap_id] = obj.quat_wxyz[fi]
                    mj.mj_forward(model, data)
                ids = object_ids[fi]
                _draw_contacts(
                    viewer,
                    body_world[fi],
                    obj_world[fi, ids],
                    distances[fi],
                    threshold,
                    max_vis,
                )
                viewer.cam.lookat[:] = lookat[fi]
            viewer.sync()
            if not state["playing"]:
                time.sleep(0.02)


def main() -> int:
    args = parse_args()
    hiphi_root = args.hiphi_root.resolve()
    smplx_model = args.smplx_model.resolve()
    ik_path = args.ik_config.resolve()
    if not hiphi_root.is_dir():
        print(f"ERROR: HiPHI root not found: {hiphi_root}")
        return 1
    if not smplx_model.exists():
        print(f"ERROR: SMPL-X model not found: {smplx_model}")
        return 1
    if not ik_path.is_file():
        print(f"ERROR: IK config not found: {ik_path}")
        return 1

    with open(ik_path) as fh:
        ik_config = json.load(fh)

    clips = discover_clips(hiphi_root)
    clip = resolve_clip(clips, args.clip, args.bvh, args.object_category, hiphi_root)
    objects = filter_objects(clip.objects, args.object_category)
    gender = args.gender
    if gender == "auto":
        gender = str((clip.meta.get("actor_metadata") or {}).get("gender") or "neutral")

    human_frames, bvh_height, frame_time, _parser = load_bvh_all_frames(
        clip.bvh_path,
        start=args.start,
        end=args.end,
        scale=args.bvh_scale,
    )
    frame_origin = int(args.start or 0)
    if args.max_frames and args.max_frames < len(human_frames):
        human_frames = human_frames[: args.max_frames]
    fps = (1.0 / frame_time) if frame_time > 0 else float(clip.meta.get("fps") or 90.0)
    use_height, height_label = resolve_human_height(args.human_height, clip.meta, bvh_height)
    align_to_head = args.human_height == "bvh"
    sh, sv = hips_object_scales(ik_config, use_height)
    assumed = float(ik_config.get("human_height_assumption", 1.8))
    ratio = use_height / assumed if assumed > 1e-8 else 1.0

    print(f"Clip   : {clip.motion_id}")
    print(f"SMPL-X : {smplx_model}  gender={gender}")
    print(f"Frames : {len(human_frames)}  fps≈{fps:.1f}  height {height_label}")
    print(
        f"GMR Mixamo Hips (not applied)  height_ratio={ratio:.4f}  "
        f"sh={sh:.4f}  sv={sv:.4f}"
    )

    if not objects:
        print("ERROR: this clip has no object meshes/tracks; cannot run contact.")
        return 1

    driver = MixamoToSMPLX(smplx_model, gender=gender, ext="pkl")
    motion = compute_stretched_smplx_motion(
        driver,
        human_frames,
        fit_bones=not args.no_fit_bones,
        align_to_head=align_to_head,
        use_height=None if align_to_head else use_height,
        extra_thigh_m=LEG_EXTRA_THIGH_M,
        extra_shin_m=LEG_EXTRA_SHIN_M,
        verbose=True,
    )

    body_world = np.asarray(motion["vertices"], dtype=np.float32)
    joints_world = np.asarray(motion["joints"], dtype=np.float32)
    local_verts = np.asarray(motion["vertices_local"], dtype=np.float32)
    anchor = np.asarray(motion["target"], dtype=np.float64)
    lookat = np.asarray(motion["lookat"], dtype=np.float64)
    n_body = body_world.shape[1]
    if args.body_points and 0 < args.body_points < n_body:
        rng = np.random.default_rng(args.seed)
        pick = np.sort(rng.choice(n_body, size=int(args.body_points), replace=False))
        body_query = body_world[:, pick]
    else:
        pick = np.arange(n_body, dtype=np.int32)
        body_query = body_world

    for obj in objects:
        obj.pos_zup, obj.quat_wxyz = load_object_track(
            obj.track_path,
            n_frames=len(human_frames),
            frame_origin=frame_origin,
            convert_y_up=True,
        )
        obj.mesh_scale_xyz = np.full(3, MESH_SCALE, dtype=np.float64)
    s_pos = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    s_mesh = np.full(3, MESH_SCALE, dtype=np.float64)
    print(
        f"Object (stretch world)  pos×{s_pos.tolist()}  "
        f"mesh×{np.round(s_mesh, 5).tolist()}"
    )

    obj_local_parts = []
    for i, obj in enumerate(objects):
        pts = sample_object_local_points(
            obj.mesh_path,
            obj.mesh_scale_xyz,
            args.object_samples,
            args.seed + 17 * (i + 1),
        )
        obj_local_parts.append(pts)
        print(f"  {obj.object_id}: {len(pts)} local surface pts  "
              f"pos0={np.round(obj.pos_zup[0], 3).tolist()}")
    object_local = np.concatenate(obj_local_parts, axis=0)
    primary = objects[0]
    contact = compute_object_contact(
        body_query,
        objects,
        obj_local_parts,
        args.snap_threshold,
    )
    distances = contact["distances"]
    active = (distances <= args.contact_threshold).sum(axis=1)
    print(
        f"[ObjectContact] object={primary.object_id}  "
        f"body_points={body_query.shape[1]}  object_points={len(object_local)}  "
        f"frames={len(human_frames)}  "
        f"source_mesh_scale={np.round(s_mesh, 5).tolist()}  "
        f"min={float(distances.min()):.4f}  p50={float(np.percentile(distances, 50)):.4f}  "
        f"max={float(distances.max()):.4f}  "
        f"snap<{args.snap_threshold:.4f}m={contact['snapped_count']}/{distances.size}  "
        f"active min/mean/max={int(active.min())}/{float(active.mean()):.2f}/{int(active.max())}"
    )

    if args.out is not None:
        out = args.out.resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out,
            body_points=body_query.astype(np.float32),
            body_vertex_index=pick.astype(np.int32),
            joints=joints_world.astype(np.float32),
            faces=motion["faces"],
            object_points_local=object_local,
            object_points_world=contact["object_points_world"],
            distances=contact["distances"],
            object_ids=contact["object_ids"],
            pair_vectors=contact["pair_vectors"],
            motion_positions=np.asarray(primary.pos_zup, dtype=np.float32),
            motion_quats_wxyz=np.asarray(primary.quat_wxyz, dtype=np.float32),
            scale_horizontal=np.asarray(sh, dtype=np.float32),
            scale_vertical=np.asarray(sv, dtype=np.float32),
            object_pos_scale=np.asarray(s_pos, dtype=np.float32),
            contact_threshold=np.asarray(args.contact_threshold, dtype=np.float32),
            snap_threshold=np.asarray(args.snap_threshold, dtype=np.float32),
            fps=np.asarray(fps, dtype=np.float32),
            clip_id=np.asarray(clip.motion_id),
            object_id=np.asarray(primary.object_id),
        )
        print(f"Saved {out}")

    if args.no_viewer:
        return 0

    tmp_obj = tempfile.NamedTemporaryFile(prefix="smplx_contact_", suffix=".obj", delete=False)
    tmp_obj_path = Path(tmp_obj.name)
    tmp_obj.close()
    write_obj(tmp_obj_path, local_verts[0], driver.faces)
    xml_str = build_smplx_scene_xml(tmp_obj_path, objects)
    model, tmp_xml = compile_scene(xml_str)
    try:
        bind_mocap_ids(model, objects)
        _pin_visual_mesh_frame(model, "smplx_human_geom", local_verts[0])
        launch_contact_viewer(
            model=model,
            local_verts=local_verts,
            objects=objects,
            anchor=np.asarray(anchor, dtype=np.float64),
            lookat=np.asarray(lookat, dtype=np.float64),
            body_world=body_query,
            contact=contact,
            threshold=args.contact_threshold,
            max_vis=args.max_vis_contacts,
            fps=fps,
            title=f"{clip.motion_id}  contact  sh={sh:.3f} sv={sv:.3f}",
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
