#!/usr/bin/env python3
"""Replay a SitChair 50 Hz export in MuJoCo (evaluation only).

Reads nothing except the clip folder written by ``hiphi_to_robot`` / the
standard exporter:

    data/SitChair/data_000/
        robot_50hz.npz
        obj_motion_global_50hz.pkl
        contact_labels_50hz.npy
        object.obj
        metadata.json

Robot qpos is rebuilt from ``body_pos_w[0]`` / ``body_quat_w[0]`` (base_link)
and ``joint_pos``. The chair mesh is already metres with frame-0 world pose
baked in; later frames apply the relative transform implied by ``obj_trans`` /
``obj_rot``. The D20 XML is only the visual robot asset, not motion.

Examples::

    python scripts/vis_sitchair_export.py
    python scripts/vis_sitchair_export.py --clip data/SitChair/data_000
    python scripts/vis_sitchair_export.py --clip data/SitChair/data_000 --check
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

import mujoco as mj
import mujoco.viewer as mjv

DEFAULT_CLIP = _PROJECT / "data" / "SitChair" / "data_000"
DEFAULT_XML = _PROJECT / "assets" / "D20_29dof" / "mjcf" / "D20.xml"
RGBA_FREE = (0.72, 0.45, 0.28, 1.0)
RGBA_CONTACT = (0.20, 0.82, 0.35, 1.0)


def load_clip(clip_dir: Path) -> dict:
    clip_dir = Path(clip_dir)
    robot_path = clip_dir / "robot_50hz.npz"
    obj_path = clip_dir / "obj_motion_global_50hz.pkl"
    label_path = clip_dir / "contact_labels_50hz.npy"
    mesh_path = clip_dir / "object.obj"
    missing = [p for p in (robot_path, obj_path, label_path, mesh_path) if not p.is_file()]
    if missing:
        names = ", ".join(p.name for p in missing)
        raise FileNotFoundError(f"{clip_dir} is missing: {names}")

    robot = dict(np.load(robot_path))
    with open(obj_path, "rb") as fh:
        obj = pickle.load(fh)
    labels = np.asarray(np.load(label_path), dtype=bool).reshape(-1)
    n = int(robot["joint_pos"].shape[0])
    n = min(n, int(np.asarray(obj["obj_trans"]).shape[0]), int(labels.shape[0]))
    if n < 1:
        raise ValueError(f"{clip_dir} has no frames")
    fps = float(np.asarray(robot["fps"]).reshape(-1)[0])
    meta_path = clip_dir / "metadata.json"
    metadata = json.loads(meta_path.read_text()) if meta_path.is_file() else None
    return {
        "dir": clip_dir,
        "n": n,
        "fps": fps if fps > 1e-6 else 50.0,
        "joint_pos": np.asarray(robot["joint_pos"][:n], dtype=np.float64),
        "body_pos_w": np.asarray(robot["body_pos_w"][:n], dtype=np.float64),
        "body_quat_w": np.asarray(robot["body_quat_w"][:n], dtype=np.float64),
        "obj_trans": np.asarray(obj["obj_trans"][:n], dtype=np.float64),
        "obj_rot": np.asarray(obj["obj_rot"][:n], dtype=np.float64),
        "labels": labels[:n],
        "mesh_path": mesh_path,
        "metadata": metadata,
    }


def _base_link_index(clip: dict) -> int:
    meta = clip.get("metadata") or {}
    for row in meta.get("body_links", {}).get("index", []):
        if row.get("name") == "base_link":
            return int(row["index"])
    return 0


def _hinge_qpos_slice(clip: dict) -> slice:
    meta = clip.get("metadata") or {}
    qslice = meta.get("qpos_layout", {}).get("hinges", {}).get("qpos_slice")
    if qslice and len(qslice) == 2:
        return slice(int(qslice[0]), int(qslice[1]))
    return slice(7, 36)


def reconstruct_qpos(clip: dict, nq: int) -> np.ndarray:
    if nq < 36:
        raise ValueError(f"robot nq={nq} is too small for 7+29 qpos")
    n = clip["n"]
    base_i = _base_link_index(clip)
    hinge = _hinge_qpos_slice(clip)
    qpos = np.zeros((n, nq), dtype=np.float64)
    qpos[:, :3] = clip["body_pos_w"][:, base_i]
    quat = clip["body_quat_w"][:, base_i].copy()
    nrm = np.linalg.norm(quat, axis=1, keepdims=True)
    nrm = np.maximum(nrm, 1e-12)
    qpos[:, 3:7] = quat / nrm
    qpos[:, hinge] = clip["joint_pos"]
    return qpos


def baked_mesh_mocap_pose(obj_trans: np.ndarray, obj_rot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mocap pose for an OBJ that already contains frame-0 world transform."""
    rot0 = np.asarray(obj_rot[0], dtype=np.float64)
    trans0 = np.asarray(obj_trans[0], dtype=np.float64)
    rot_t = np.asarray(obj_rot, dtype=np.float64)
    trans_t = np.asarray(obj_trans, dtype=np.float64)
    rel_rot = np.einsum("tij,jk->tik", rot_t, rot0.T)
    rel_trans = trans_t - np.einsum("tij,j->ti", rel_rot, trans0)
    quat = R.from_matrix(rel_rot).as_quat(scalar_first=True)
    return rel_trans, quat


def build_eval_xml(robot_xml: Path, object_obj: Path) -> str:
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
    ET.SubElement(
        asset,
        "mesh",
        name="export_object",
        file=str(object_obj.resolve()),
        scale="1 1 1",
        maxhullvert="32",
    )
    body = ET.SubElement(world, "body", name="export_object", mocap="true")
    ET.SubElement(body, "inertial", pos="0 0 0", mass="1", diaginertia="1 1 1")
    rgba = " ".join(f"{c:.4f}" for c in RGBA_FREE)
    ET.SubElement(
        body,
        "geom",
        name="export_object_geom",
        type="mesh",
        mesh="export_object",
        rgba=rgba,
        contype="0",
        conaffinity="0",
        condim="1",
        group="1",
    )
    return ET.tostring(root, encoding="unicode")


def compile_xml(xml_str: str) -> tuple[mj.MjModel, Path]:
    tmp = tempfile.NamedTemporaryFile(prefix="sitchair_eval_", suffix=".xml", delete=False)
    path = Path(tmp.name)
    tmp.write(xml_str.encode("utf-8"))
    tmp.close()
    try:
        model = mj.MjModel.from_xml_path(str(path))
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return model, path


def set_geom_rgba(model: mj.MjModel, geom_id: int, rgba: tuple[float, float, float, float]) -> None:
    model.geom_rgba[geom_id] = np.asarray(rgba, dtype=np.float64)


def check_reconstruction(model: mj.MjModel, clip: dict, qpos: np.ndarray) -> None:
    from general_motion_retargeting.utils.standard_motion_export import d20_link_body_ids

    body_ids, names = d20_link_body_ids(model)
    data = mj.MjData(model)
    n = min(32, clip["n"])
    step = max(1, clip["n"] // n)
    pos_err = []
    quat_err = []
    for i in range(0, clip["n"], step):
        data.qpos[: qpos.shape[1]] = qpos[i]
        mj.mj_forward(model, data)
        delta = data.xpos[body_ids] - clip["body_pos_w"][i]
        pos_err.append(float(np.linalg.norm(delta, axis=1).max()))
        rec = data.xquat[body_ids]
        tgt = clip["body_quat_w"][i]
        rec = rec / np.maximum(np.linalg.norm(rec, axis=1, keepdims=True), 1e-12)
        tgt = tgt / np.maximum(np.linalg.norm(tgt, axis=1, keepdims=True), 1e-12)
        # quat double-cover
        dots = np.abs(np.einsum("ij,ij->i", rec, tgt))
        quat_err.append(float(np.degrees(2.0 * np.arccos(np.clip(dots, 0.0, 1.0))).max()))
    print(
        f"  FK vs export  pos max={max(pos_err):.5f}m  "
        f"ori max={max(quat_err):.3f}deg  "
        f"checked {len(pos_err)} frames"
    )
    print(f"  link0={names[0]}  contact True {int(clip['labels'].sum())}/{clip['n']}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualize a SitChair export folder (robot + object + contact labels).",
    )
    p.add_argument("--clip", type=Path, default=DEFAULT_CLIP, help="Export folder (default: data/SitChair/data_000)")
    p.add_argument("--xml", type=Path, default=DEFAULT_XML, help="D20 MJCF used only as the visual robot")
    p.add_argument("--check", action="store_true", help="Print FK reconstruction error vs body_pos_w")
    p.add_argument("--no-viewer", action="store_true", help="Only load/check, do not open MuJoCo")
    p.add_argument("--no-follow-camera", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    clip_dir = args.clip.resolve()
    xml_path = args.xml.resolve()
    if not xml_path.is_file():
        print(f"ERROR: robot XML not found: {xml_path}")
        return 1
    try:
        clip = load_clip(clip_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1

    xml_str = build_eval_xml(xml_path, clip["mesh_path"])
    model, tmp_xml = compile_xml(xml_str)
    try:
        qpos = reconstruct_qpos(clip, model.nq)
        obj_pos, obj_quat = baked_mesh_mocap_pose(clip["obj_trans"], clip["obj_rot"])
        body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "export_object")
        geom_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "export_object_geom")
        mocap_id = int(model.body_mocapid[body_id]) if body_id >= 0 else -1
        if mocap_id < 0 or geom_id < 0:
            print("ERROR: compiled scene is missing the export object mocap geom")
            return 1
        base_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "base_link")
        print(f"Clip  : {clip_dir}")
        print(f"  frames={clip['n']}  fps={clip['fps']:.1f}  "
              f"contact={int(clip['labels'].sum())}/{clip['n']}")
        print(f"  object.obj baked at frame 0; mocap uses relative world pose")
        if args.check:
            check_reconstruction(model, clip, qpos)
        if args.no_viewer:
            return 0

        data = mj.MjData(model)
        n_frames = clip["n"]
        fps = clip["fps"]
        state = {"current": 0, "playing": True, "speed": 1.0}

        def apply_frame(i: int) -> None:
            data.qpos[:] = qpos[i]
            data.mocap_pos[mocap_id] = obj_pos[i]
            data.mocap_quat[mocap_id] = obj_quat[i]
            set_geom_rgba(model, geom_id, RGBA_CONTACT if clip["labels"][i] else RGBA_FREE)
            mj.mj_forward(model, data)

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
            else:
                return
            print(
                f"  frame {state['current'] + 1}/{n_frames}  "
                f"{'play' if state['playing'] else 'pause'}  "
                f"x{state['speed']:g}  contact={bool(clip['labels'][state['current']])}"
            )

        print("  Controls: ← → frame   ↑ ↓ ±10   Space play/pause   1-5 speed   R restart")
        print("  Chair green = contact_labels True, brown = False")
        apply_frame(0)
        with mjv.launch_passive(model, data, key_callback=key_callback) as viewer:
            viewer.cam.distance = 3.5
            viewer.cam.azimuth = 140
            viewer.cam.elevation = -15
            last = time.perf_counter()
            while viewer.is_running():
                now = time.perf_counter()
                dt = now - last
                last = now
                if state["playing"]:
                    step = max(int(round(dt * fps * state["speed"])), 0)
                    if step:
                        nxt = state["current"] + step
                        state["current"] = 0 if nxt >= n_frames else nxt
                fi = state["current"]
                apply_frame(fi)
                if not args.no_follow_camera and base_id >= 0:
                    viewer.cam.lookat[:] = data.xpos[base_id]
                viewer.sync()
                if not state["playing"]:
                    time.sleep(0.02)
        return 0
    finally:
        tmp_xml.unlink(missing_ok=True)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        raise SystemExit(130)
