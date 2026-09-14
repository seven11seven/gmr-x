"""Drive an SMPL-X mesh from Mixamo-style global (pos, quat) frames.

HiPHI BVH is Mixamo, not SMPL-X parameters. After GMR scaling we only have
joint globals. This module maps those rotations onto the SMPL-X kinematic tree,
optionally stretches each body bone to the matching Mixamo length, places the
pelvis on Hips, and applies a leftover uniform Hips scale (Y-up: XZ × sh, Y × sv).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

_P = np.array(
    [[0.0, 0.0, 1.0],
     [1.0, 0.0, 0.0],
     [0.0, 1.0, 0.0]],
    dtype=np.float64,
)
_P_ROT = R.from_matrix(_P)

# SMPL-X body joint indices we expose after the Y-up → MuJoCo remap.
SMPLX_JOINT_LEFT_ANKLE = 7   # maps to Mixamo LeftFoot
SMPLX_JOINT_LEFT_FOOT = 10   # maps to Mixamo LeftToeBase
_THIGH_BONES = (4, 5)        # hip → knee (LeftLeg / RightLeg)
_SHIN_BONES = (7, 8)         # knee → ankle (LeftFoot / RightFoot)

# SMPL-X body parents (22 joints, no hands/face).
_SMPLX_PARENTS = (
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19
)

# Mixamo name for each SMPL-X body joint index.
_MIXAMO_FOR_SMPLX = (
    "Hips",
    "LeftUpLeg",
    "RightUpLeg",
    "Spine",
    "LeftLeg",
    "RightLeg",
    "Spine1",
    "LeftFoot",
    "RightFoot",
    "Spine2",
    "LeftToeBase",
    "RightToeBase",
    "Neck",
    "LeftShoulder",
    "RightShoulder",
    "Head",
    "LeftArm",
    "RightArm",
    "LeftForeArm",
    "RightForeArm",
    "LeftHand",
    "RightHand",
)

_SPINE3_FALLBACKS = ("Spine4", "Spine3", "Spine2")

# Left/right pairs among the 22 body joints (for symmetric bone-length fit).
_SYMMETRIC_BONES = (
    (1, 2), (4, 5), (7, 8), (10, 11),
    (13, 14), (16, 17), (18, 19), (20, 21),
)
_LEG_BONES = (1, 2, 4, 5, 7, 8, 10, 11)
_TORSO_BONES = (3, 6, 9, 12, 15)
# Mixamo clavicles (Shoulder) are not the SMPL-X Shoulder joints; skip them.
_ARM_BONES = (16, 17, 18, 19, 20, 21)
# SMPL-X LeftForeArm / RightForeArm. Local X is twist along the humerus
# (does not change the elbow included angle); Y/Z carry flexion.
_ELBOW_JOINTS = (18, 19)
# Mixamo bind: LeftForeArm along +X, RightForeArm along −X (Y-up).
_MIXAMO_BIND_HUMERUS = (
    np.array([1.0, 0.0, 0.0], dtype=np.float64),
    np.array([-1.0, 0.0, 0.0], dtype=np.float64),
)


def _from_to_rotation(src: np.ndarray, dst: np.ndarray) -> R:
    """Smallest rotation that takes ``src`` onto ``dst``."""
    a = np.asarray(src, dtype=np.float64)
    b = np.asarray(dst, dtype=np.float64)
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    v = np.cross(a, b)
    s = float(np.linalg.norm(v))
    c = float(np.dot(a, b))
    if s < 1e-8:
        return R.identity()
    return R.from_rotvec((v / s) * np.arctan2(s, c))


def _as_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def write_obj(path: Path, verts: np.ndarray, faces: np.ndarray) -> None:
    with open(path, "w") as fh:
        for v in verts:
            fh.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for a, b, c in faces:
            fh.write(f"f {int(a) + 1} {int(b) + 1} {int(c) + 1}\n")


def _mj_quat_to_yup(quat_wxyz: np.ndarray) -> R:
    # v_mj = P @ v_yup, so R_yup = P^{-1} R_mj P.
    R_mj = R.from_quat(np.asarray(quat_wxyz, dtype=np.float64), scalar_first=True)
    return _P_ROT.inv() * R_mj * _P_ROT


def _mj_pos_to_yup(pos: np.ndarray) -> np.ndarray:
    p = np.asarray(pos, dtype=np.float64)
    return np.array([p[1], p[2], p[0]], dtype=np.float64)


def _yup_pos_to_mj(pos: np.ndarray) -> np.ndarray:
    p = np.asarray(pos, dtype=np.float64)
    return p[[2, 0, 1]]


def _yup_verts_to_mj(verts: np.ndarray) -> np.ndarray:
    v = np.asarray(verts, dtype=np.float64)
    return np.column_stack([v[:, 2], v[:, 0], v[:, 1]])


def _lookup_mixamo(human_data: dict, smplx_idx: int):
    name = _MIXAMO_FOR_SMPLX[smplx_idx]
    if name in human_data:
        return human_data[name]
    if smplx_idx == 9:
        for alt in _SPINE3_FALLBACKS:
            if alt in human_data:
                return human_data[alt]
    return None


def resolve_smplx_source(
    model_dir: str | Path,
    gender: str = "neutral",
    ext: str = "npz",
) -> tuple[Path, str, str]:
    """Locate an SMPL-X file.

    Accepts a ``.pkl``/``.npz`` file, a folder that *contains* those files
    (``assets/smpl-x/SMPLX_NEUTRAL.pkl``), or the official
    ``<root>/smplx/SMPLX_*.{pkl,npz}`` layout used by ``smplx.create``.
    """
    p = Path(model_dir)
    gender = (gender or "neutral").lower()
    if gender not in ("male", "female", "neutral"):
        gender = "neutral"
    exts = [e for e in (ext, "pkl", "npz") if e]
    genders = [g for g in (gender, "neutral", "female", "male")]

    if p.is_file():
        return p, gender, (p.suffix.lstrip(".") or ext)

    if p.is_dir():
        for g in genders:
            for e in exts:
                cand = p / f"SMPLX_{g.upper()}.{e}"
                if cand.is_file():
                    return cand, g, e
        nested = p / "smplx"
        if nested.is_dir():
            for g in genders:
                for e in exts:
                    cand = nested / f"SMPLX_{g.upper()}.{e}"
                    if cand.is_file():
                        return cand, g, e
            return nested, gender, ext

    return p, gender, ext


def _load_smplx_model(model_dir: str | Path, gender: str, ext: str):
    import smplx
    from smplx.body_models import SMPLX

    source, gender_used, ext_used = resolve_smplx_source(model_dir, gender, ext)
    if source.is_file():
        model = SMPLX(str(source), gender=gender_used, use_pca=False, ext=ext_used)
    else:
        model = smplx.create(
            str(source),
            model_type="smplx",
            gender=gender_used,
            use_pca=False,
            ext=ext_used,
        )
    model.eval()
    return model, gender_used, ext_used


class MixamoToSMPLX:
    def __init__(
        self,
        model_dir: str | Path,
        gender: str = "neutral",
        sh: float = 1.0,
        sv: float = 1.0,
        ext: str = "npz",
    ) -> None:
        import torch

        self._torch = torch
        gender = (gender or "neutral").lower()
        if gender not in ("male", "female", "neutral"):
            gender = "neutral"
        self.model, self.gender, self.ext = _load_smplx_model(model_dir, gender, ext)
        self.sh = float(sh)
        self.sv = float(sv)
        self._scale_yup = np.array([self.sh, self.sv, self.sh], dtype=np.float64)
        self.faces = np.asarray(self.model.faces, dtype=np.int32)
        self.model_dir = Path(model_dir)
        self.bone_scale = np.ones(22, dtype=np.float64)
        self._lbs = _as_numpy(self.model.lbs_weights).astype(np.float64)
        self._parents_all = _as_numpy(self.model.parents).astype(np.int64)
        self._arm_rest_corr = self._compute_arm_rest_corr()

    def set_scale(self, sh: float, sv: float) -> None:
        self.sh = float(sh)
        self.sv = float(sv)
        self._scale_yup = np.array([self.sh, self.sv, self.sh], dtype=np.float64)

    def _compute_arm_rest_corr(self) -> tuple[R, R]:
        """Align SMPL-X rest humeri to Mixamo bind (±X).

        Neutral SMPL-X rest arms already sit ~15–18° down and in. Copying
        Mixamo locals on top of that makes hanging arms sag into the torso.
        """
        torch = self._torch
        with torch.no_grad():
            rest = self.model().joints[0].cpu().numpy().astype(np.float64)
        left = _from_to_rotation(rest[18] - rest[16], _MIXAMO_BIND_HUMERUS[0])
        right = _from_to_rotation(rest[19] - rest[17], _MIXAMO_BIND_HUMERUS[1])
        return left, right

    def rest_height(self) -> float:
        """SMPL-X T-pose height in the model's Y-up frame (metres)."""
        torch = self._torch
        with torch.no_grad():
            out = self.model()
            verts = out.vertices[0].cpu().numpy().astype(np.float64)
            joints_all = out.joints[0].cpu().numpy().astype(np.float64)
        verts, _ = self._apply_bone_scales(verts, joints_all)
        return float(verts[:, 1].max() - verts[:, 1].min())

    def _rotvecs_from_mixamo(self, human_data: dict) -> np.ndarray:
        rotvecs = np.zeros((22, 3), dtype=np.float64)
        R_global: list[R | None] = [None] * 22
        for j in range(22):
            pose = _lookup_mixamo(human_data, j)
            if pose is None:
                continue
            R_global[j] = _mj_quat_to_yup(pose[1])

        if R_global[0] is None:
            raise KeyError("scaled human data has no Hips; cannot drive SMPL-X")

        for j in range(22):
            if R_global[j] is None:
                p = _SMPLX_PARENTS[j]
                R_global[j] = R_global[p] if p >= 0 and R_global[p] is not None else R.identity()
            if j == 0:
                rotvecs[0] = R_global[0].as_rotvec()
            else:
                rotvecs[j] = (R_global[_SMPLX_PARENTS[j]].inv() * R_global[j]).as_rotvec()
        # Mixamo elbows are ball joints. Keep swing (flexion), drop twist
        # around the upper arm so the mesh is not wrung at the elbow.
        for j in _ELBOW_JOINTS:
            rotvecs[j, 0] = 0.0
        rotvecs[16] = (R.from_rotvec(rotvecs[16]) * self._arm_rest_corr[0]).as_rotvec()
        rotvecs[17] = (R.from_rotvec(rotvecs[17]) * self._arm_rest_corr[1]).as_rotvec()
        return rotvecs

    def _forward_smplx(self, rotvecs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        torch = self._torch
        with torch.no_grad():
            out = self.model(
                global_orient=torch.tensor(rotvecs[0][None], dtype=torch.float32),
                body_pose=torch.tensor(rotvecs[1:].reshape(1, -1), dtype=torch.float32),
            )
            verts = out.vertices[0].cpu().numpy().astype(np.float64)
            joints_all = out.joints[0].cpu().numpy().astype(np.float64)
        return self._apply_bone_scales(verts, joints_all)

    def _apply_bone_scales(
        self, verts: np.ndarray, joints_all: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Lengthen/shorten the posed kinematic tree, then drag vertices by LBS."""
        if np.allclose(self.bone_scale, 1.0):
            return verts, joints_all

        n_j = int(joints_all.shape[0])
        new_j = joints_all.copy()
        n_body = min(22, n_j)
        for j in range(1, n_body):
            p = int(self._parents_all[j]) if j < len(self._parents_all) else _SMPLX_PARENTS[j]
            if p < 0:
                continue
            new_j[j] = new_j[p] + (joints_all[j] - joints_all[p]) * float(self.bone_scale[j])
        for j in range(n_body, n_j):
            p = int(self._parents_all[j]) if j < len(self._parents_all) else -1
            if p < 0:
                continue
            new_j[j] = new_j[p] + (joints_all[j] - joints_all[p])

        n_w = min(self._lbs.shape[1], n_j)
        verts = verts + self._lbs[:, :n_w] @ (new_j[:n_w] - joints_all[:n_w])
        return verts, new_j

    def fit_bone_scales(
        self,
        human_data: dict,
        *,
        mode: str = "chain",
        symmetric: bool = True,
        clamp: tuple[float, float] = (0.5, 1.8),
    ) -> dict:
        """Match Mixamo bone lengths. ``betas`` stay at zero.

        Default ``mode="chain"`` uses one scale per region (leg / torso) from
        the summed bone lengths. Arms stay at 1.0 so the elbow mesh is not
        pinched by translation-only LBS. Mixamo and SMPL-X split the spine
        and clavicle differently, so per-bone ratios overstretch Neck / Shoulder.

        ``mode="per_bone"`` uses ``||child − parent||`` on each mapped joint.
        Left/right pairs are averaged when ``symmetric`` is set.
        """
        saved = self.bone_scale.copy()
        self.bone_scale = np.ones(22, dtype=np.float64)
        _verts, joints_all = self._forward_smplx(self._rotvecs_from_mixamo(human_data))
        joints = joints_all[:22]
        self.bone_scale = saved

        lo, hi = clamp
        mixamo_len = np.zeros(22, dtype=np.float64)
        smplx_len = np.zeros(22, dtype=np.float64)
        for j in range(1, 22):
            child = _lookup_mixamo(human_data, j)
            parent = _lookup_mixamo(human_data, _SMPLX_PARENTS[j])
            if child is None or parent is None:
                continue
            mixamo_len[j] = float(np.linalg.norm(
                np.asarray(child[0]) - np.asarray(parent[0])
            ))
            smplx_len[j] = float(np.linalg.norm(joints[j] - joints[_SMPLX_PARENTS[j]]))

        def _ratio(idxs: tuple[int, ...]) -> float:
            num = float(sum(mixamo_len[j] for j in idxs if mixamo_len[j] > 1e-8))
            den = float(sum(smplx_len[j] for j in idxs if mixamo_len[j] > 1e-8))
            if den <= 1e-8:
                return 1.0
            return float(np.clip(num / den, lo, hi))

        scales = np.ones(22, dtype=np.float64)
        if mode == "per_bone":
            for j in range(1, 22):
                if smplx_len[j] > 1e-8:
                    scales[j] = float(np.clip(mixamo_len[j] / smplx_len[j], lo, hi))
            if symmetric:
                for a, b in _SYMMETRIC_BONES:
                    mid = 0.5 * (scales[a] + scales[b])
                    scales[a] = scales[b] = mid
            for j in _ARM_BONES:
                scales[j] = 1.0
            s_leg = float(np.mean(scales[list(_LEG_BONES)]))
            s_torso = float(np.mean(scales[list(_TORSO_BONES)]))
            s_arm = 1.0
        else:
            s_leg = _ratio(_LEG_BONES)
            s_torso = _ratio(_TORSO_BONES)
            # Leave arms unscaled. The translation-only LBS used in
            # ``_apply_bone_scales`` pinches/twists the elbow when the
            # upper-arm and forearm joints move by different amounts.
            s_arm = 1.0
            for j in _LEG_BONES:
                scales[j] = s_leg
            for j in _TORSO_BONES:
                scales[j] = s_torso

        self.bone_scale = scales
        return {
            "mode": mode,
            "scale": scales,
            "mixamo_len": mixamo_len,
            "smplx_len": smplx_len,
            "leg": s_leg,
            "torso": s_torso,
            "arm": s_arm,
        }

    def lengthen_leg_segments(
        self,
        human_data: dict,
        *,
        thigh_m: float = 0.04,
        shin_m: float = 0.04,
    ) -> dict:
        """Add metres to thigh (hip→knee) and shin (knee→ankle) in world units.

        Applied on top of the current ``bone_scale`` and uniform ``sh/sv``, so
        call this after height alignment if the extra length should stay visible.
        """
        saved = self.bone_scale.copy()
        self.bone_scale = np.ones(22, dtype=np.float64)
        _verts, joints_all = self._forward_smplx(self._rotvecs_from_mixamo(human_data))
        self.bone_scale = saved
        joints = joints_all[:22]
        s = 0.5 * (abs(self.sh) + abs(self.sv))
        if s <= 1e-8:
            s = 1.0
        extras = {j: float(thigh_m) for j in _THIGH_BONES}
        extras.update({j: float(shin_m) for j in _SHIN_BONES})
        for j, extra in extras.items():
            raw_L = float(np.linalg.norm(joints[j] - joints[_SMPLX_PARENTS[j]]))
            if raw_L <= 1e-8:
                continue
            self.bone_scale[j] = saved[j] + extra / (raw_L * s)
        return {
            "thigh_m": float(thigh_m),
            "shin_m": float(shin_m),
            "bone_scale": self.bone_scale.copy(),
        }

    def drive(self, human_data: dict) -> tuple[np.ndarray, np.ndarray]:
        """Return (verts, joints) in the BVHParser / MuJoCo frame.

        Joints are the first 22 SMPL-X body joints. Pelvis is placed on Mixamo
        Hips; callers may then translate so an ankle matches ``LeftFoot``.
        """
        verts, joints_all = self._forward_smplx(self._rotvecs_from_mixamo(human_data))
        joints = joints_all[:22]
        pelvis = joints[0].copy()
        hips_yup = _mj_pos_to_yup(human_data["Hips"][0])
        verts = (verts - pelvis) * self._scale_yup + hips_yup
        joints = (joints - pelvis) * self._scale_yup + hips_yup
        return (
            np.ascontiguousarray(_yup_verts_to_mj(verts)),
            np.ascontiguousarray(_yup_verts_to_mj(joints)),
        )

    def verts_mujoco(self, human_data: dict) -> np.ndarray:
        """Return SMPL-X vertices in the BVHParser / MuJoCo frame."""
        verts, _joints = self.drive(human_data)
        return verts
