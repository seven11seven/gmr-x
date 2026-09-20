"""Export retargeted HiPHI clips in the SitChair 50 Hz layout.

See ``reference/output_data_format.md``.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

EXPORT_FPS = 50.0
HIP_CONTACT_TOKENS = ("hip_pitch", "hip_roll")
MIN_CONTACT_SLOTS = 10


def allocate_clip_dir(root: Path, slot: int | None = None) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if slot is not None:
        out = root / f"data_{int(slot):03d}"
        out.mkdir(parents=True, exist_ok=True)
        return out
    index = 0
    while True:
        out = root / f"data_{index:03d}"
        if not out.exists():
            out.mkdir(parents=True)
            return out
        index += 1


def d20_link_body_ids(model: mj.MjModel) -> tuple[np.ndarray, list[str]]:
    ids: list[int] = []
    names: list[str] = []
    for body_id in range(model.nbody):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, body_id) or ""
        if name.endswith("_link"):
            ids.append(int(body_id))
            names.append(name)
    if len(ids) != 35:
        raise ValueError(
            f"expected 35 D20 *_link bodies for body_pos_w, got {len(ids)}: {names}"
        )
    return np.asarray(ids, dtype=np.int32), names


def hinge_qpos_slice(model: mj.MjModel) -> slice:
    n_hinge = int(model.nq - 7)
    if n_hinge != 29:
        raise ValueError(f"expected 29 hinge DoF after freejoint, got {n_hinge}")
    return slice(7, 7 + n_hinge)


def hinge_joint_index(model: mj.MjModel) -> list[dict]:
    """``joint_pos[:, i]`` is this hinge, in MuJoCo qpos order after the freejoint."""
    rows: list[dict] = []
    for joint_id in range(model.njnt):
        if int(model.jnt_type[joint_id]) == int(mj.mjtJoint.mjJNT_FREE):
            continue
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, joint_id) or f"joint_{joint_id}"
        rows.append(
            {
                "index": len(rows),
                "name": name,
                "joint_id": int(joint_id),
                "qposadr": int(model.jnt_qposadr[joint_id]),
                "dofadr": int(model.jnt_dofadr[joint_id]),
            }
        )
    if len(rows) != 29:
        raise ValueError(f"expected 29 hinge joints, got {len(rows)}")
    return rows


def link_body_index(model: mj.MjModel) -> list[dict]:
    """``body_pos_w[:, i]`` / ``body_quat_w[:, i]`` for each ``*_link`` body."""
    body_ids, names = d20_link_body_ids(model)
    return [
        {"index": int(i), "name": names[i], "body_id": int(body_ids[i])}
        for i in range(len(names))
    ]


def write_export_metadata(
    out_dir: Path,
    model: mj.MjModel,
    *,
    fps: float,
    n_frames: int,
    contact_threshold: float,
    object_name: str | None = None,
) -> Path:
    free_name = None
    for joint_id in range(model.njnt):
        if int(model.jnt_type[joint_id]) == int(mj.mjtJoint.mjJNT_FREE):
            free_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, joint_id)
            break
    payload = {
        "fps": float(fps),
        "n_frames": int(n_frames),
        "robot": "d20_29dof",
        "root_body": "base_link",
        "quat_order": "wxyz",
        "units": "meters, radians",
        "qpos_layout": {
            "freejoint": {
                "name": free_name or "floating_base_joint",
                "qpos_slice": [0, 7],
                "fields": ["x", "y", "z", "qw", "qx", "qy", "qz"],
                "note": "Not stored in joint_pos; recovered from body_pos_w/body_quat_w of base_link",
            },
            "hinges": {"qpos_slice": [7, 36], "count": 29},
        },
        "joint_pos": {
            "array": "robot_50hz.npz::joint_pos",
            "shape": [int(n_frames), 29],
            "definition": "MuJoCo hinge qpos after the freejoint, increasing qposadr",
            "index": hinge_joint_index(model),
        },
        "body_links": {
            "array": "robot_50hz.npz::body_pos_w / body_quat_w",
            "shape": [int(n_frames), 35, 3],
            "definition": "MuJoCo bodies whose names end with '_link', increasing body_id; world frame",
            "index": link_body_index(model),
        },
        "contact_labels": {
            "array": "contact_labels_50hz.npy",
            "shape": [int(n_frames)],
            "robot_body_tokens": list(HIP_CONTACT_TOKENS),
            "min_slots": int(MIN_CONTACT_SLOTS),
            "threshold_m": float(contact_threshold),
            "rule": f"True if more than {MIN_CONTACT_SLOTS} hip_pitch/roll thigh slots have distance <= threshold",
        },
        "object": {
            "mesh": "object.obj",
            "motion": "obj_motion_global_50hz.pkl",
            "name": object_name,
            "obj_scale": 1.0,
            "mesh_note": "Metres; frame-0 world pose is baked into vertices",
            "obj_trans_note": "World translation used in retargeting (pos_scale, ground-z, pos_offset)",
        },
    }
    path = Path(out_dir) / "metadata.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def resample_times(n_src: int, fps_src: float, fps_dst: float = EXPORT_FPS) -> tuple[np.ndarray, np.ndarray]:
    n_src = int(n_src)
    if n_src < 1:
        raise ValueError("no frames to resample")
    fps_src = float(fps_src)
    fps_dst = float(fps_dst)
    t_src = np.arange(n_src, dtype=np.float64) / max(fps_src, 1e-8)
    duration = float(t_src[-1])
    n_dst = int(np.floor(duration * fps_dst + 1e-9)) + 1
    n_dst = max(n_dst, 1)
    t_dst = np.arange(n_dst, dtype=np.float64) / max(fps_dst, 1e-8)
    t_dst = np.clip(t_dst, t_src[0], t_src[-1])
    return t_src, t_dst


def interp_linear(t_src: np.ndarray, values: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.shape[0] != len(t_src):
        raise ValueError(f"interp length {values.shape[0]} != {len(t_src)}")
    if values.ndim == 1:
        return np.interp(t_dst, t_src, values).astype(np.float64)
    tail = values.shape[1:]
    src = values.reshape(len(t_src), -1)
    out = np.empty((len(t_dst), src.shape[1]), dtype=np.float64)
    for col in range(src.shape[1]):
        out[:, col] = np.interp(t_dst, t_src, src[:, col])
    return out.reshape((len(t_dst),) + tail)


def interp_quat_wxyz(t_src: np.ndarray, quat_wxyz: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64).reshape(-1, 4)
    if len(quat_wxyz) == 1 or len(t_dst) == 1:
        return np.repeat(quat_wxyz[:1], len(t_dst), axis=0)
    rots = R.from_quat(quat_wxyz, scalar_first=True)
    slerp = Slerp(t_src, rots)
    return slerp(t_dst).as_quat(scalar_first=True)


def finite_difference(series: np.ndarray, fps: float) -> np.ndarray:
    series = np.asarray(series, dtype=np.float64)
    out = np.zeros_like(series)
    if len(series) == 1:
        return out
    fps = float(fps)
    out[1:-1] = (series[2:] - series[:-2]) * (0.5 * fps)
    out[0] = (series[1] - series[0]) * fps
    out[-1] = (series[-1] - series[-2]) * fps
    return out


def angular_velocity_world(rot_mats: np.ndarray, fps: float) -> np.ndarray:
    rot_mats = np.asarray(rot_mats, dtype=np.float64)
    n = len(rot_mats)
    out = np.zeros((n, 3), dtype=np.float64)
    if n < 2:
        return out
    rots = R.from_matrix(rot_mats)
    for i in range(n - 1):
        out[i] = (rots[i + 1] * rots[i].inv()).as_rotvec() * float(fps)
    out[-1] = out[-2]
    return out


def resample_qpos(qpos: np.ndarray, fps_src: float, fps_dst: float = EXPORT_FPS) -> np.ndarray:
    qpos = np.asarray(qpos, dtype=np.float64)
    t_src, t_dst = resample_times(len(qpos), fps_src, fps_dst)
    root = interp_linear(t_src, qpos[:, :3], t_dst)
    quat = interp_quat_wxyz(t_src, qpos[:, 3:7], t_dst)
    joints = interp_linear(t_src, qpos[:, 7:], t_dst)
    return np.concatenate([root, quat, joints], axis=1)


def fk_link_motion(
    model: mj.MjModel,
    qpos: np.ndarray,
    body_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    data = mj.MjData(model)
    n_frames = int(qpos.shape[0])
    n_bodies = int(len(body_ids))
    pos = np.empty((n_frames, n_bodies, 3), dtype=np.float64)
    quat = np.empty((n_frames, n_bodies, 4), dtype=np.float64)
    for i in range(n_frames):
        data.qpos[:] = qpos[i]
        mj.mj_forward(model, data)
        pos[i] = data.xpos[body_ids]
        quat[i] = data.xquat[body_ids]
    return pos, quat


def hip_pitch_roll_contact_labels(
    distances: np.ndarray,
    body_names: np.ndarray,
    threshold: float,
    min_slots: int = MIN_CONTACT_SLOTS,
) -> np.ndarray:
    """True iff more than ``min_slots`` hip_pitch/roll thigh slots are in contact."""
    distances = np.asarray(distances, dtype=np.float64)
    names = np.asarray(body_names).astype(str).reshape(-1)
    if distances.ndim != 2 or distances.shape[1] != len(names):
        raise ValueError(
            f"contact distances {distances.shape} vs body names {len(names)}"
        )
    mask = np.array(
        [any(tok in name.lower() for tok in HIP_CONTACT_TOKENS) for name in names],
        dtype=bool,
    )
    if not np.any(mask):
        return np.zeros(len(distances), dtype=bool)
    n_hit = (distances[:, mask] <= float(threshold)).sum(axis=1)
    return n_hit > int(min_slots)


def write_scaled_object_obj(
    src_path: Path,
    dst_path: Path,
    scale_xyz: np.ndarray,
    translation: np.ndarray | None = None,
    rotation: np.ndarray | None = None,
) -> None:
    """Rewrite OBJ in metres: ``R @ (v * scale) + translation``.

    ``scale_xyz`` is the MuJoCo mesh scale already in metres/file-unit
    (``0.01 * mesh_scale``). Default R=I, translation=0 keeps the retarget
    body origin. Pass the actual retarget pose to bake a world-frame mesh.
    """
    import trimesh

    mesh = trimesh.load(src_path, process=False, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        geoms = [g for g in mesh.dump() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            raise ValueError(f"no triangle mesh in {src_path}")
        mesh = trimesh.util.concatenate(geoms)
    verts = np.asarray(mesh.vertices, dtype=np.float64) * np.asarray(scale_xyz, dtype=np.float64).reshape(3)
    if rotation is not None:
        rot = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
        verts = verts @ rot.T
    if translation is not None:
        verts = verts + np.asarray(translation, dtype=np.float64).reshape(3)
    out = trimesh.Trimesh(
        vertices=verts.astype(np.float64),
        faces=np.asarray(mesh.faces, dtype=np.int32),
        process=False,
    )
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    out.export(dst_path)


def export_standard_clip(
    out_dir: Path,
    *,
    model: mj.MjModel,
    qpos: np.ndarray,
    fps_src: float,
    objects: list,
    object_contact: dict | None,
    contact_threshold: float,
    bake_object_pose: bool = True,
) -> Path:
    """Write ``robot_50hz.npz``, ``obj_motion_global_50hz.pkl``, labels, OBJ."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    qpos50 = resample_qpos(qpos, fps_src, EXPORT_FPS)
    joint_slice = hinge_qpos_slice(model)
    joint_pos = qpos50[:, joint_slice]
    joint_vel = finite_difference(joint_pos, EXPORT_FPS)
    body_ids, _body_names = d20_link_body_ids(model)
    body_pos_w, body_quat_w = fk_link_motion(model, qpos50, body_ids)
    body_lin_vel_w = finite_difference(body_pos_w, EXPORT_FPS)
    body_rots = R.from_quat(body_quat_w.reshape(-1, 4), scalar_first=True).as_matrix().reshape(
        len(qpos50), len(body_ids), 3, 3
    )
    body_ang_vel_w = np.stack(
        [angular_velocity_world(body_rots[:, b], EXPORT_FPS) for b in range(len(body_ids))],
        axis=1,
    )
    np.savez_compressed(
        out_dir / "robot_50hz.npz",
        joint_pos=joint_pos.astype(np.float32),
        joint_vel=joint_vel.astype(np.float32),
        body_pos_w=body_pos_w.astype(np.float32),
        body_quat_w=body_quat_w.astype(np.float32),
        body_lin_vel_w=body_lin_vel_w.astype(np.float32),
        body_ang_vel_w=body_ang_vel_w.astype(np.float32),
        fps=np.asarray([EXPORT_FPS], dtype=np.float32),
    )

    t_src, t_dst = resample_times(len(qpos), fps_src, EXPORT_FPS)
    labels_src = np.zeros(len(qpos), dtype=bool)
    if object_contact is not None and "distances" in object_contact:
        names = object_contact.get("robot_body_names")
        if names is not None:
            labels_src = hip_pitch_roll_contact_labels(
                object_contact["distances"],
                names,
                contact_threshold,
            )
    labels50 = interp_linear(t_src, labels_src.astype(np.float64), t_dst) >= 0.5
    np.save(out_dir / "contact_labels_50hz.npy", labels50.astype(bool))

    if objects:
        obj = objects[0]
        trans_src = np.asarray(obj.pos_zup, dtype=np.float64)
        quat_src = np.asarray(obj.quat_wxyz, dtype=np.float64)
        n_obj = min(len(trans_src), len(quat_src), len(qpos))
        trans50 = interp_linear(t_src[:n_obj], trans_src[:n_obj], t_dst)
        quat50 = interp_quat_wxyz(t_src[:n_obj], quat_src[:n_obj], t_dst)
        rot50 = R.from_quat(quat50, scalar_first=True).as_matrix()
        payload = {
            "obj_trans": trans50.astype(np.float32),
            "obj_rot": rot50.astype(np.float32),
            "obj_lin_vel": finite_difference(trans50, EXPORT_FPS).astype(np.float32),
            "obj_ang_vel": angular_velocity_world(rot50, EXPORT_FPS).astype(np.float32),
            "obj_scale": 1.0,
        }
        with open(out_dir / "obj_motion_global_50hz.pkl", "wb") as fh:
            pickle.dump(payload, fh, protocol=4)
        # Mesh is metres. Bake the retarget scale and the frame-0 world pose
        # actually used after pos_scale / ground-z / pos_offset.
        write_scaled_object_obj(
            obj.mesh_path,
            out_dir / "object.obj",
            obj.mesh_scale_xyz,
            translation=trans50[0] if bake_object_pose else None,
            rotation=rot50[0] if bake_object_pose else None,
        )
        print(
            f"  object.obj  metres  scale={np.asarray(obj.mesh_scale_xyz).tolist()}  "
            f"baked_frame0_pose={bake_object_pose}  "
            f"pkl obj_trans is retarget world pose ({len(trans50)} @ {EXPORT_FPS:.0f}Hz)"
        )
    obj_name = None
    if objects:
        obj_name = getattr(objects[0], "name", None) or Path(objects[0].mesh_path).stem
    write_export_metadata(
        out_dir,
        model,
        fps=EXPORT_FPS,
        n_frames=len(qpos50),
        contact_threshold=float(contact_threshold),
        object_name=obj_name,
    )
    return out_dir
