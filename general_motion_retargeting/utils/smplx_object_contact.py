"""SMPL-X ↔ object nearest-surface contact, in the GMR robot-world scale.

Contact is the UMR-style map: each body point queries the closest object
sample, storing distance / object index / offset vector. Snap below a
threshold zeros the residual so near-touch counts as true contact.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

# SMPL-X body joints: 1 = left hip / up-leg, 2 = right hip / up-leg.
SMPLX_JOINT_LEFT_UPLEG = 1
SMPLX_JOINT_RIGHT_UPLEG = 2
THIGH_JOINTS = (SMPLX_JOINT_LEFT_UPLEG, SMPLX_JOINT_RIGHT_UPLEG)


def robot_world_scale(sh: float, sv: float) -> np.ndarray:
    return np.array([sh, sh, sv], dtype=np.float64)


def scale_points(points: np.ndarray, scale_xyz: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=np.float32) * np.asarray(scale_xyz, dtype=np.float32)


def smplx_thigh_vertex_index(lbs_weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (vertex_index, side) for left/right thigh verts.

    ``side`` is 0 for left (joint 1) and 1 for right (joint 2), using the
    rest-pose LBS argmax so the mask is constant across frames.
    """
    nearest = np.argmax(np.asarray(lbs_weights, dtype=np.float64)[:, :22], axis=1)
    mask = (nearest == SMPLX_JOINT_LEFT_UPLEG) | (nearest == SMPLX_JOINT_RIGHT_UPLEG)
    idx = np.where(mask)[0].astype(np.int32)
    sides = np.where(nearest[idx] == SMPLX_JOINT_LEFT_UPLEG, 0, 1).astype(np.int32)
    return idx, sides


def sample_object_local_points(mesh_path: Path, scale_xyz: np.ndarray, n: int, seed: int) -> np.ndarray:
    import trimesh

    mesh = trimesh.load(mesh_path, process=False, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        geoms = [g for g in mesh.dump() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            raise ValueError(f"no triangle mesh in {mesh_path}")
        mesh = trimesh.util.concatenate(geoms)
    verts = np.asarray(mesh.vertices, dtype=np.float64) * np.asarray(scale_xyz, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    scaled = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    pts, _fid = trimesh.sample.sample_surface(scaled, int(n), seed=int(seed))
    return np.asarray(pts, dtype=np.float32)


def transform_local_points(local: np.ndarray, quat_wxyz: np.ndarray, pos: np.ndarray) -> np.ndarray:
    rot = R.from_quat(np.asarray(quat_wxyz, dtype=np.float64), scalar_first=True).as_matrix()
    return np.asarray(local, dtype=np.float32) @ rot.T.astype(np.float32) + np.asarray(pos, dtype=np.float32)


def compute_object_contact(
    body_points: np.ndarray,
    objects,
    object_local_parts: list[np.ndarray],
    snap_threshold: float,
) -> dict:
    """UMR nearest-surface contact: body slot → closest object sample."""
    n_frames, n_body, _ = body_points.shape
    n_obj = int(sum(len(p) for p in object_local_parts))
    obj_world = np.empty((n_frames, n_obj, 3), dtype=np.float32)
    distances = np.empty((n_frames, n_body), dtype=np.float32)
    object_ids = np.empty((n_frames, n_body), dtype=np.int32)
    pair_vectors = np.empty((n_frames, n_body, 3), dtype=np.float32)
    snapped = 0
    for i in tqdm(range(n_frames), desc="contact", unit="frame"):
        parts = [
            transform_local_points(local, obj.quat_wxyz[i], obj.pos_zup[i])
            for obj, local in zip(objects, object_local_parts)
        ]
        world = np.concatenate(parts, axis=0)
        obj_world[i] = world
        dist, ids = cKDTree(world).query(body_points[i], k=1)
        ids = np.asarray(ids, dtype=np.int32)
        vec = body_points[i] - world[ids]
        dist = np.asarray(dist, dtype=np.float32)
        if snap_threshold > 0.0:
            snap = dist < snap_threshold
            snapped += int(snap.sum())
            vec[snap] = 0.0
            dist[snap] = 0.0
        distances[i] = dist
        object_ids[i] = ids
        pair_vectors[i] = vec.astype(np.float32)
    return {
        "object_points_world": obj_world,
        "distances": distances,
        "object_ids": object_ids,
        "pair_vectors": pair_vectors,
        "snapped_count": snapped,
    }


def contact_frame_dict(
    contact: dict,
    frame_index: int,
    body_points: np.ndarray,
    sides: np.ndarray,
) -> dict:
    """Slice a full-sequence contact map into the per-frame IK payload.

    Row ``i`` is one correspondence slot: SMPL-X id, object id, and pair vector.
    The robot task uses the same row as its slot index.
    """
    ids = np.asarray(contact["object_ids"][frame_index], dtype=np.int32).reshape(-1)
    obj_world = np.asarray(contact["object_points_world"][frame_index], dtype=np.float64)
    return {
        "distances": np.asarray(contact["distances"][frame_index], dtype=np.float64).reshape(-1),
        "object_ids": ids,
        "pair_vectors": np.asarray(contact["pair_vectors"][frame_index], dtype=np.float64).reshape(-1, 3),
        "object_points": obj_world,
        "body_points": np.asarray(body_points[frame_index], dtype=np.float64).reshape(-1, 3),
        "sides": np.asarray(sides, dtype=np.int32).reshape(-1),
        "mode": "world_points",
    }
