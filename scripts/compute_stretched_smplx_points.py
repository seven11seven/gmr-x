#!/usr/bin/env python3
"""Compute stretched SMPL-X vertex / joint coordinates from a HiPHI BVH clip.

Uses the same bone-fit, crown-to-Head scale, extra thigh/shin length, and
ankle pin as ``scripts/bvh_to_smplx_viewer.py``. No MuJoCo, no objects.

Saved ``vertices`` are world-frame mesh points after stretch + pin (metres).

Examples::

    python scripts/compute_stretched_smplx_points.py
    python scripts/compute_stretched_smplx_points.py --clip Change_posture-sit_down_0002
    python scripts/compute_stretched_smplx_points.py --out output/stretched_smplx.npz --max-frames 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(_PROJECT_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "tools"))

from general_motion_retargeting.utils.smplx_from_mixamo import MixamoToSMPLX
from retargeting_visualizer.visualize_retargeting import load_bvh_all_frames

from bvh_to_smplx_viewer import (
    DEFAULT_CLIP,
    DEFAULT_SMPLX,
    LEG_EXTRA_SHIN_M,
    LEG_EXTRA_THIGH_M,
    _parse_height,
    compute_stretched_smplx_motion,
    resolve_human_height,
)
from hiphi_to_robot import DEFAULT_HIPHI_ROOT, discover_clips, resolve_clip


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Save stretched SMPL-X point-cloud coordinates (viewer-equivalent).",
    )
    p.add_argument("--hiphi-root", type=Path, default=DEFAULT_HIPHI_ROOT)
    p.add_argument("--clip", type=str, default=DEFAULT_CLIP)
    p.add_argument("--bvh", type=Path, default=None)
    p.add_argument("--smplx-model", type=Path, default=DEFAULT_SMPLX)
    p.add_argument("--gender", type=str, default="auto", choices=("auto", "neutral", "female", "male"))
    p.add_argument("--start", type=int, default=None)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--frame", type=int, default=None, help="Keep a single frame index (after start/end).")
    p.add_argument(
        "--human-height",
        type=_parse_height,
        default="bvh",
        help="'bvh' = mesh crown matches skeleton Head; 'meta' / metres = T-pose AABB scale.",
    )
    p.add_argument("--bvh-scale", type=float, default=0.01)
    p.add_argument("--no-fit-bones", action="store_true")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output .npz path (default: output/stretched_smplx/<clip>.npz).",
    )
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

    clips = discover_clips(hiphi_root)
    clip = resolve_clip(clips, args.clip, args.bvh, None, hiphi_root)
    gender = args.gender
    if gender == "auto":
        gender = str((clip.meta.get("actor_metadata") or {}).get("gender") or "neutral")

    human_frames, bvh_height, frame_time, _parser = load_bvh_all_frames(
        clip.bvh_path,
        start=args.start,
        end=args.end,
        scale=args.bvh_scale,
    )
    if args.max_frames and args.max_frames < len(human_frames):
        human_frames = human_frames[: args.max_frames]
    fps = (1.0 / frame_time) if frame_time > 0 else float(clip.meta.get("fps") or 90.0)
    use_height, height_label = resolve_human_height(args.human_height, clip.meta, bvh_height)
    align_to_head = args.human_height == "bvh"

    print(f"Clip   : {clip.motion_id}")
    print(f"SMPL-X : {smplx_model}  gender={gender}")
    print(f"Frames : {len(human_frames)}  fps≈{fps:.1f}  height {height_label}")

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

    # motion 
    # 点云 / 骨架
    # vertices：(F, V, 3) float32 — 拉伸后、脚踝钉到 BVH 后的世界坐标网格点。第 122 行用的就是这个。
    # joints：(F, J, 3) float32 — 同样变换后的 22 个 SMPL-X 关节。
    # vertices_local / joints_local：相对 origin 的局部坐标（viewer 里上传给 mesh 的就是这个）。
    # faces：(20908, 3) int32 — 三角面，不随帧变。
    # 对齐
    # origin：(F, 3) — 钉之前每帧的参考点（脚踝中点 XY + 左踝 Z）。
    # target：(F, 3) — BVH 上对应的目标点。
    # offset：(F, 3) — target - origin，世界坐标 = 局部 + target。
    # lookat：(F, 3) — 钉完后的 Hips，给相机用。
    # 拉伸参数
    # scale：float — 头顶对齐 Head 的均匀缩放。
    # rest_height：float — 未缩放 T-pose 身高。
    # bone_scale：(22,) float32 — 每个身体关节的骨长倍率。
    # fit：骨长拟合详情（mode / leg / torso / arm / 各骨 scale、mixamo_len、smplx_len）；--no-fit-bones 时为 None。
    # 关系：vertices[i] == vertices_local[i] + target[i]。

    vertices = motion["vertices"]
    joints = motion["joints"]
    if args.frame is not None:
        fi = int(args.frame)
        if not 0 <= fi < len(vertices):
            print(f"ERROR: --frame {fi} out of range [0, {len(vertices) - 1}]")
            return 1
        vertices = vertices[fi : fi + 1]
        joints = joints[fi : fi + 1]
        for key in ("vertices_local", "joints_local", "origin", "target", "offset", "lookat"):
            motion[key] = np.asarray(motion[key])[fi : fi + 1]

    out = args.out
    if out is None:
        out = _PROJECT_ROOT / "output" / "stretched_smplx" / f"{clip.motion_id}.npz"
    out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        vertices=np.asarray(vertices, dtype=np.float32),
        joints=np.asarray(joints, dtype=np.float32),
        faces=motion["faces"],
        vertices_local=np.asarray(motion["vertices_local"], dtype=np.float32),
        joints_local=np.asarray(motion["joints_local"], dtype=np.float32),
        origin=np.asarray(motion["origin"], dtype=np.float32),
        target=np.asarray(motion["target"], dtype=np.float32),
        offset=np.asarray(motion["offset"], dtype=np.float32),
        bone_scale=motion["bone_scale"],
        scale=np.asarray(motion["scale"], dtype=np.float32),
        rest_height=np.asarray(motion["rest_height"], dtype=np.float32),
        fps=np.asarray(fps, dtype=np.float32),
        clip_id=np.asarray(clip.motion_id),
        gender=np.asarray(driver.gender),
        extra_thigh_m=np.asarray(LEG_EXTRA_THIGH_M, dtype=np.float32),
        extra_shin_m=np.asarray(LEG_EXTRA_SHIN_M, dtype=np.float32),
    )
    v = vertices[0]
    print(
        f"Saved {out}\n"
        f"  vertices {vertices.shape}  joints {joints.shape}  faces {motion['faces'].shape}\n"
        f"  frame0 xyz=[{v.min(axis=0).round(3).tolist()} … {v.max(axis=0).round(3).tolist()}]"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        raise SystemExit(130)
