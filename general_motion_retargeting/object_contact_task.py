"""Mink task: match robot thigh-mesh points to SMPL-X object-contact vectors.

UMR residual (``retarget_smpl_to_humanoid_surface_vector.py`` 1835–1879)::

    current = p_robot - p_object
    e = cost * strength * (current - target_vector)

``target_vector`` comes from stretched SMPL-X thigh verts vs the scaled object.
Robot points are samples on the thigh visual meshes, expressed in geom frames
so ``mj_jac`` of the world point is the task Jacobian. Mink already stacks
custom tasks (see ``FrameJointDamperTask``); this is the same QP path.
"""
from __future__ import annotations

import numpy as np
import mink
import mujoco as mj

_THIGH_TOKENS = ("hip", "upleg", "thigh")
_THIGH_EXCLUDE = ("knee", "ankle", "foot", "toe", "pelvis", "waist")


def _body_side(name: str) -> int | None:
    n = str(name).lower()
    if "left" in n or n.startswith("l_"):
        return 0
    if "right" in n or n.startswith("r_"):
        return 1
    return None


def is_thigh_body(name: str) -> bool:
    n = str(name).lower()
    if any(tok in n for tok in _THIGH_EXCLUDE):
        return False
    return any(tok in n for tok in _THIGH_TOKENS)


def thigh_visual_geom_ids(model: mj.MjModel) -> list[int]:
    ids: list[int] = []
    for geom_id in range(model.ngeom):
        if int(model.geom_type[geom_id]) != int(mj.mjtGeom.mjGEOM_MESH):
            continue
        geom_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if "collision" in geom_name.lower():
            continue
        body_id = int(model.geom_bodyid[geom_id])
        body_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, body_id) or ""
        if is_thigh_body(body_name):
            ids.append(int(geom_id))
    return ids


def _geom_mesh_local_verts(model: mj.MjModel, geom_id: int) -> np.ndarray:
    mesh_id = int(model.geom_dataid[geom_id])
    if mesh_id < 0:
        return np.zeros((0, 3), dtype=np.float64)
    vadr = int(model.mesh_vertadr[mesh_id])
    vnum = int(model.mesh_vertnum[mesh_id])
    return np.asarray(model.mesh_vert[vadr : vadr + vnum], dtype=np.float64)


def sample_robot_thigh_cloud(
    model: mj.MjModel,
    samples_per_side: int = 128,
    seed: int = 0,
) -> dict:
    """Sample visual-mesh vertices on left/right thigh geoms.

    Points are stored in each geom's local frame so FK is
    ``geom_xpos + geom_xmat @ local``.
    """
    rng = np.random.default_rng(int(seed))
    geom_ids: list[int] = []
    local_pos: list[np.ndarray] = []
    sides: list[int] = []
    names: list[str] = []
    per_side: dict[int, list[tuple[int, np.ndarray, str]]] = {0: [], 1: []}
    for geom_id in thigh_visual_geom_ids(model):
        body_id = int(model.geom_bodyid[geom_id])
        body_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, body_id) or ""
        side = _body_side(body_name)
        if side is None:
            continue
        verts = _geom_mesh_local_verts(model, geom_id)
        if len(verts) == 0:
            continue
        geom_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, geom_id) or body_name
        per_side[side].append((int(geom_id), verts, geom_name))

    n_per = max(int(samples_per_side), 1)
    for side in (0, 1):
        chunks = per_side[side]
        if not chunks:
            continue
        all_verts = np.concatenate([c[1] for c in chunks], axis=0)
        all_gids = np.concatenate(
            [np.full(len(c[1]), c[0], dtype=np.int32) for c in chunks], axis=0
        )
        all_names = []
        for gid, verts, name in chunks:
            all_names.extend([name] * len(verts))
        if len(all_verts) > n_per:
            pick = rng.choice(len(all_verts), size=n_per, replace=False)
            pick.sort()
            all_verts = all_verts[pick]
            all_gids = all_gids[pick]
            all_names = [all_names[i] for i in pick]
        geom_ids.extend(int(g) for g in all_gids)
        local_pos.append(all_verts)
        sides.extend([side] * len(all_verts))
        names.extend(all_names)

    if not geom_ids:
        raise ValueError(
            "No thigh visual meshes found (expected body names containing "
            "'hip' / 'thigh' / 'upleg', excluding knee/ankle)."
        )
    return {
        "geom_ids": np.asarray(geom_ids, dtype=np.int32),
        "local_pos": np.concatenate(local_pos, axis=0).astype(np.float64),
        "sides": np.asarray(sides, dtype=np.int32),
        "geom_names": np.asarray(names, dtype=object),
    }


def robot_cloud_world_points(
    data: mj.MjData,
    geom_ids: np.ndarray,
    local_pos: np.ndarray,
) -> np.ndarray:
    pts = np.empty((len(geom_ids), 3), dtype=np.float64)
    for i, geom_id in enumerate(geom_ids):
        gid = int(geom_id)
        rot = data.geom_xmat[gid].reshape(3, 3)
        pts[i] = data.geom_xpos[gid] + rot @ local_pos[i]
    return pts


class ObjectContactTask(mink.Task):
    """Drive ``(p_robot - p_object)`` toward the SMPL-X contact offset."""

    def __init__(
        self,
        model: mj.MjModel,
        geom_ids: np.ndarray,
        local_pos: np.ndarray,
        sides: np.ndarray,
        cost: float = 8.0,
        threshold: float = 0.10,
        max_points: int = 32,
        max_match_distance: float = 0.30,
        lm_damping: float = 1.0,
        gain: float = 1.0,
    ) -> None:
        super().__init__(cost=np.zeros(0), gain=gain, lm_damping=lm_damping)
        self.model = model
        self.geom_ids = np.asarray(geom_ids, dtype=np.int32).reshape(-1)
        self.local_pos = np.asarray(local_pos, dtype=np.float64).reshape(-1, 3)
        self.sides = np.asarray(sides, dtype=np.int32).reshape(-1)
        if len(self.geom_ids) != len(self.local_pos) or len(self.geom_ids) != len(self.sides):
            raise ValueError("geom_ids, local_pos, and sides must have the same length")
        self.body_ids = np.asarray(model.geom_bodyid[self.geom_ids], dtype=np.int32)
        self.base_cost = float(cost)
        self.threshold = float(threshold)
        self.max_points = int(max_points)
        self.max_match_distance = float(max_match_distance)
        self._side_index = {
            0: np.where(self.sides == 0)[0].astype(np.int32),
            1: np.where(self.sides == 1)[0].astype(np.int32),
        }
        self.active_slot_ids = np.zeros(0, dtype=np.int32)
        self.object_points = np.zeros((0, 3), dtype=np.float64)
        self.target_vectors = np.zeros((0, 3), dtype=np.float64)
        self.row_costs = np.zeros(0, dtype=np.float64)
        self._jacp = np.zeros((3, model.nv), dtype=np.float64)
        self.n_active = 0

    def clear(self) -> None:
        self.active_slot_ids = np.zeros(0, dtype=np.int32)
        self.object_points = np.zeros((0, 3), dtype=np.float64)
        self.target_vectors = np.zeros((0, 3), dtype=np.float64)
        self.row_costs = np.zeros(0, dtype=np.float64)
        self.cost = np.zeros(0)
        self.n_active = 0

    def set_from_smpl_frame(self, configuration, frame: dict | None) -> int:
        """Match active SMPL thigh contacts to nearest same-side robot points."""
        if frame is None or self.base_cost <= 0.0 or self.threshold < 0.0:
            self.clear()
            return 0
        distances = np.asarray(frame["distances"], dtype=np.float64).reshape(-1)
        object_ids = np.asarray(frame["object_ids"], dtype=np.int32).reshape(-1)
        target_vectors = np.asarray(frame["pair_vectors"], dtype=np.float64).reshape(-1, 3)
        object_points = np.asarray(frame["object_points"], dtype=np.float64).reshape(-1, 3)
        body_points = np.asarray(frame["body_points"], dtype=np.float64).reshape(-1, 3)
        smpl_sides = np.asarray(frame["sides"], dtype=np.int32).reshape(-1)
        if len(distances) != len(body_points) or len(smpl_sides) != len(distances):
            raise ValueError(
                "Object contact frame length mismatch: "
                f"distances={len(distances)} body={len(body_points)} sides={len(smpl_sides)}"
            )
        active = np.where(distances <= self.threshold)[0].astype(np.int32)
        if active.size == 0 or len(object_points) == 0:
            self.clear()
            return 0
        if self.max_points > 0 and active.size > self.max_points:
            active = active[np.argsort(distances[active])[: self.max_points]]
        paired_ids = object_ids[active]
        valid = (paired_ids >= 0) & (paired_ids < len(object_points))
        active = active[valid]
        paired_ids = paired_ids[valid]
        if active.size == 0:
            self.clear()
            return 0

        robot_pts = robot_cloud_world_points(
            configuration.data, self.geom_ids, self.local_pos
        )
        used = np.zeros(len(self.geom_ids), dtype=bool)
        slot_ids: list[int] = []
        obj_pts: list[np.ndarray] = []
        tgt_vecs: list[np.ndarray] = []
        strengths: list[float] = []
        max_d2 = self.max_match_distance * self.max_match_distance
        # Tightest SMPL contacts first, so they claim the nearest robot slots.
        order = np.argsort(distances[active])
        for row in order:
            src = int(active[row])
            side = int(smpl_sides[src])
            candidates = self._side_index.get(side)
            if candidates is None or candidates.size == 0:
                continue
            free = candidates[~used[candidates]]
            if free.size == 0:
                continue
            delta = robot_pts[free] - body_points[src]
            d2 = np.einsum("ij,ij->i", delta, delta)
            best = int(np.argmin(d2))
            if max_d2 > 0.0 and float(d2[best]) > max_d2:
                continue
            slot = int(free[best])
            used[slot] = True
            slot_ids.append(slot)
            obj_pts.append(object_points[int(paired_ids[row])])
            tgt_vecs.append(target_vectors[src])
            strengths.append(
                float(
                    np.clip(
                        (self.threshold - distances[src]) / max(self.threshold, 1e-8),
                        0.5,
                        1.0,
                    )
                )
            )

        if not slot_ids:
            self.clear()
            return 0
        self.active_slot_ids = np.asarray(slot_ids, dtype=np.int32)
        self.object_points = np.stack(obj_pts, axis=0)
        self.target_vectors = np.stack(tgt_vecs, axis=0)
        self.row_costs = self.base_cost * np.asarray(strengths, dtype=np.float64)
        self.cost = np.repeat(self.row_costs, 3)
        self.n_active = int(len(slot_ids))
        return self.n_active

    def _active_world_points(self, configuration) -> np.ndarray:
        if self.n_active == 0:
            return np.zeros((0, 3), dtype=np.float64)
        return robot_cloud_world_points(
            configuration.data,
            self.geom_ids[self.active_slot_ids],
            self.local_pos[self.active_slot_ids],
        )

    def compute_error(self, configuration) -> np.ndarray:
        if self.n_active == 0:
            return np.zeros(0, dtype=np.float64)
        current = self._active_world_points(configuration) - self.object_points
        return (current - self.target_vectors).reshape(-1)

    def compute_jacobian(self, configuration) -> np.ndarray:
        nv = configuration.nv
        if self.n_active == 0:
            return np.zeros((0, nv), dtype=np.float64)
        jac = np.zeros((3 * self.n_active, nv), dtype=np.float64)
        data = configuration.data
        points = self._active_world_points(configuration)
        jacp = self._jacp
        for row, slot_id in enumerate(self.active_slot_ids):
            mj.mj_jac(
                self.model,
                data,
                jacp,
                None,
                points[row],
                int(self.body_ids[int(slot_id)]),
            )
            jac[3 * row : 3 * row + 3] = jacp
        return jac
