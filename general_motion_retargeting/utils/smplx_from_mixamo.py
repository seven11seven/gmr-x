"""Drive an SMPL-X mesh from Mixamo-style global (pos, quat) frames.

HiPHI BVH is Mixamo, not SMPL-X parameters. After GMR scaling we only have
joint globals. This module maps those rotations onto the SMPL-X kinematic tree,
places the pelvis on scaled Hips, and stretches the mesh with the same Hips
horizontal/vertical factors (Y-up: XZ × sh, Y × sv).

SMPL-X itself cannot reproduce per-bone anisotropic GMR scales; Hips sh/sv is
the same approximation used for HiPHI objects.
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
SMPLX_JOINT_LEFT_FOOT = 10  # maps to Mixamo LeftToeBase

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

    def set_scale(self, sh: float, sv: float) -> None:
        self.sh = float(sh)
        self.sv = float(sv)
        self._scale_yup = np.array([self.sh, self.sv, self.sh], dtype=np.float64)

    def rest_height(self) -> float:
        """SMPL-X T-pose height in the model's Y-up frame (metres)."""
        torch = self._torch
        with torch.no_grad():
            verts = self.model().vertices[0].cpu().numpy()
        return float(verts[:, 1].max() - verts[:, 1].min())

    def drive(self, human_data: dict) -> tuple[np.ndarray, np.ndarray]:
        """Return (verts, joints) in the BVHParser / MuJoCo frame.

        Joints are the first 22 SMPL-X body joints. Pelvis is placed on Mixamo
        Hips; callers may then translate so a foot matches ``LeftToeBase``.
        """
        torch = self._torch
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

        hips_yup = _mj_pos_to_yup(human_data["Hips"][0])
        with torch.no_grad():
            out = self.model(
                global_orient=torch.tensor(rotvecs[0][None], dtype=torch.float32),
                body_pose=torch.tensor(rotvecs[1:].reshape(1, -1), dtype=torch.float32),
            )
            verts = out.vertices[0].cpu().numpy().astype(np.float64)
            joints = out.joints[0, :22].cpu().numpy().astype(np.float64)
            pelvis = joints[0].copy()

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
