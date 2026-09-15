"""SMPL-X ↔ robot AE slot correspondence for object-contact IK.

Slot ``i`` in ``correspondence_contact_*.npz`` is one semantic surface location
on both meshes. Contact is queried on the skinned SMPL-X slot, then transferred
to the same robot slot (geom + local offset) without posed nearest-neighbour.
"""
from __future__ import annotations

from pathlib import Path

import mujoco as mj
import numpy as np
from scipy.spatial import cKDTree

from general_motion_retargeting.object_contact_task import is_thigh_body
from general_motion_retargeting.utils.smplx_object_contact import (
    SMPLX_JOINT_LEFT_UPLEG,
    SMPLX_JOINT_RIGHT_UPLEG,
)

DEFAULT_CORRESPONDENCE = (
    Path(__file__).resolve().parents[2]
    / "output"
    / "correspondence_d20"
    / "correspondence_contact_smplx_neutral_d20.npz"
)


def load_correspondence_contact(path: Path | str) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"SMPL-X↔robot correspondence not found: {path}\n"
            "Export it with:\n"
            "  python general_motion_retargeting/points_contact/export_correspondence_contact.py \\\n"
            "    --slots output/correspondence_d20/correspondence_slots_final.npz \\\n"
            "    --data data/correspondence_d20.npz --robot d20"
        )
    data = np.load(path, allow_pickle=True)
    required = (
        "source_vertex_ids",
        "source_bary",
        "robot_geom_ids",
        "robot_geom_names",
        "robot_body_names",
        "robot_points",
    )
    missing = [key for key in required if key not in data.files]
    if missing:
        raise KeyError(f"{path} is missing correspondence fields: {missing}")
    payload = {key: data[key] for key in data.files}
    payload["path"] = path
    n = int(np.asarray(payload["source_vertex_ids"]).shape[0])
    payload["n_slots"] = n
    return payload


def _majority_upleg(lbs_weights: np.ndarray, vertex_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per slot: (is_thigh, side) from LBS argmax of the bound triangle."""
    nearest = np.argmax(np.asarray(lbs_weights, dtype=np.float64)[:, :22], axis=1)
    joints = nearest[np.asarray(vertex_ids, dtype=np.int32)]
    left_votes = (joints == SMPLX_JOINT_LEFT_UPLEG).sum(axis=1)
    right_votes = (joints == SMPLX_JOINT_RIGHT_UPLEG).sum(axis=1)
    is_thigh = (left_votes >= 2) | (right_votes >= 2)
    sides = np.where(left_votes >= right_votes, 0, 1).astype(np.int32)
    return is_thigh.astype(bool), sides


def thigh_slot_index(
    pairs: dict,
    lbs_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep correspondence slots whose SMPL triangle is thigh and robot body is thigh.

    ``side`` is 0/1 from SMPL-X left/right up-leg LBS.
    """
    smpl_thigh, sides = _majority_upleg(lbs_weights, pairs["source_vertex_ids"])
    robot_names = np.asarray(pairs["robot_body_names"]).astype(str).reshape(-1)
    robot_thigh = np.array([is_thigh_body(name) for name in robot_names], dtype=bool)
    mask = smpl_thigh & robot_thigh
    idx = np.where(mask)[0].astype(np.int32)
    if idx.size == 0:
        raise ValueError(
            "No thigh correspondence slots (SMPL-X up-leg LBS ∩ robot hip/thigh bodies)."
        )
    return idx, sides[idx]


def skin_source_slots(
    vertices: np.ndarray,
    pairs: dict,
    slot_ids: np.ndarray,
) -> np.ndarray:
    """Barycentric SMPL-X slot positions from posed mesh vertices.

    ``vertices`` is ``(F, V, 3)``; result is ``(F, S, 3)``.
    """
    verts = np.asarray(vertices, dtype=np.float64)
    vids = np.asarray(pairs["source_vertex_ids"], dtype=np.int32)[slot_ids]
    bary = np.asarray(pairs["source_bary"], dtype=np.float64)[slot_ids]
    if int(vids.max()) >= verts.shape[1]:
        raise ValueError(
            f"Correspondence vertex id {int(vids.max())} exceeds mesh V={verts.shape[1]}"
        )
    tris = verts[:, vids]  # (F, S, 3, 3)
    return np.einsum("fsvc,sv->fsc", tris, bary).astype(np.float32)


def _yup_to_zup(points: np.ndarray) -> np.ndarray:
    p = np.asarray(points, dtype=np.float64)
    return np.column_stack([p[:, 2], p[:, 0], p[:, 1]])


def _resolve_geom_id(model: mj.MjModel, name: str, hint: int) -> int:
    if name:
        gid = int(mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, str(name)))
        if gid >= 0:
            return gid
    hint = int(hint)
    if 0 <= hint < model.ngeom:
        return hint
    return -1


def _geom_mesh_local(model: mj.MjModel, geom_id: int) -> np.ndarray:
    mesh_id = int(model.geom_dataid[geom_id])
    if mesh_id < 0:
        return np.zeros((0, 3), dtype=np.float64)
    vadr = int(model.mesh_vertadr[mesh_id])
    vnum = int(model.mesh_vertnum[mesh_id])
    return np.asarray(model.mesh_vert[vadr : vadr + vnum], dtype=np.float64)


def _center_xpos(model: mj.MjModel, data: mj.MjData, body_name: str) -> np.ndarray:
    bid = int(mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name))
    if bid < 0:
        bid = int(mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "base_link"))
    if bid < 0:
        return np.zeros(3, dtype=np.float64)
    return np.asarray(data.xpos[bid], dtype=np.float64)


def bind_robot_slots(
    model: mj.MjModel,
    pairs: dict,
    slot_ids: np.ndarray,
    center_body: str = "waist_yaw_link",
) -> dict:
    """Project correspondence robot points onto the live MJCF visual geoms.

    Returns geom-local offsets so FK is ``geom_xpos + geom_xmat @ local``.
    """
    slot_ids = np.asarray(slot_ids, dtype=np.int32).reshape(-1)
    geom_names = np.asarray(pairs["robot_geom_names"]).astype(str).reshape(-1)[slot_ids]
    geom_hints = np.asarray(pairs["robot_geom_ids"], dtype=np.int32).reshape(-1)[slot_ids]
    body_names = np.asarray(pairs["robot_body_names"]).astype(str).reshape(-1)[slot_ids]
    points_yup = np.asarray(pairs["robot_points"], dtype=np.float64).reshape(-1, 3)[slot_ids]

    data = mj.MjData(model)
    mj.mj_forward(model, data)
    center = _center_xpos(model, data, center_body)
    query_world = _yup_to_zup(points_yup) + center

    geom_ids = np.empty(len(slot_ids), dtype=np.int32)
    local_pos = np.empty((len(slot_ids), 3), dtype=np.float64)
    bind_err = np.empty(len(slot_ids), dtype=np.float64)
    sides = np.full(len(slot_ids), -1, dtype=np.int32)
    for i, name in enumerate(body_names):
        n = str(name).lower()
        if "left" in n or n.startswith("l_"):
            sides[i] = 0
        elif "right" in n or n.startswith("r_"):
            sides[i] = 1

    missing = 0
    for i, (name, hint) in enumerate(zip(geom_names, geom_hints)):
        gid = _resolve_geom_id(model, name, int(hint))
        if gid < 0:
            missing += 1
            geom_ids[i] = 0
            local_pos[i] = 0.0
            bind_err[i] = np.inf
            continue
        geom_ids[i] = gid

    if missing:
        raise ValueError(f"{missing} correspondence robot geoms were not in the MJCF")

    for gid in np.unique(geom_ids):
        mask = geom_ids == gid
        verts_local = _geom_mesh_local(model, int(gid))
        if len(verts_local) == 0:
            raise ValueError(f"geom {int(gid)} has no mesh vertices")
        rot = data.geom_xmat[int(gid)].reshape(3, 3)
        xpos = data.geom_xpos[int(gid)]
        world = xpos + verts_local @ rot.T
        dist, nn = cKDTree(world).query(query_world[mask], k=1)
        local_pos[mask] = verts_local[np.asarray(nn, dtype=np.int32)]
        bind_err[mask] = np.asarray(dist, dtype=np.float64)

    return {
        "geom_ids": geom_ids,
        "local_pos": local_pos,
        "sides": sides,
        "body_names": body_names,
        "geom_names": geom_names,
        "bind_error": bind_err,
        "center_body": center_body,
        "center": center,
    }
