"""Stage-3: retarget first N frames for weight (pos/rot) tuning preview."""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path
from typing import Any

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation as R

from tools.retargeting_app.loaders import (
    BVH_UNIT_OPTIONS,
    PROJECT_ROOT,
    resolve_under_project,
)
from tools.retargeting_app.stage2_logic import extract_offset_rows


def extract_weight_rows(ik_match_table: dict) -> list[dict[str, Any]]:
    """One editable weight row per robot body in ik_match_table1."""
    rows = []
    for robot_body, entry in (ik_match_table or {}).items():
        if not entry:
            continue
        single = entry[0] if isinstance(entry[0], list) else entry
        human = single[0]
        pos_w = float(single[1]) if len(single) > 1 else 10.0
        rot_w = float(single[2]) if len(single) > 2 else 10.0
        rows.append({
            "robot_body": robot_body,
            "human_bone": human,
            "pos_weight": pos_w,
            "rot_weight": rot_w,
        })
    rows.sort(key=lambda r: (r["human_bone"], r["robot_body"]))
    return rows


def apply_weights_to_table(
    base_table: dict | None,
    weight_rows: list[dict],
) -> dict:
    """Update pos_weight / rot_weight in ik_match_table1; preserve offsets."""
    table = copy.deepcopy(base_table or {})
    by_robot = {r["robot_body"]: r for r in weight_rows if r.get("robot_body")}

    for robot_body, row in by_robot.items():
        if robot_body not in table:
            continue
        old = table[robot_body]
        pos_w = float(row.get("pos_weight", 10))
        rot_w = float(row.get("rot_weight", 10))
        if isinstance(old[0], list):
            entries = copy.deepcopy(old)
            entries[0][1] = pos_w
            entries[0][2] = rot_w
            table[robot_body] = entries
        else:
            entry = copy.deepcopy(old)
            while len(entry) < 5:
                entry.append(None)
            entry[1] = pos_w
            entry[2] = rot_w
            table[robot_body] = entry
    return table


def merge_stage3_weights(base_config: dict, weight_rows: list[dict] | None) -> dict:
    cfg = copy.deepcopy(base_config)
    if weight_rows is not None:
        cfg["ik_match_table1"] = apply_weights_to_table(
            cfg.get("ik_match_table1"), weight_rows
        )
    return cfg


def _mat_to_quat_wxyz(mat: np.ndarray) -> list[float]:
    q = R.from_matrix(mat).as_quat(scalar_first=True)
    return [float(x) for x in q]


def _nodes_from_positions(
    positions: dict[str, np.ndarray],
    xmats: dict[str, np.ndarray] | None = None,
    quats: dict[str, np.ndarray] | None = None,
) -> list[dict]:
    nodes = []
    for name, pos in positions.items():
        if xmats and name in xmats:
            quat = _mat_to_quat_wxyz(xmats[name])
        elif quats and name in quats:
            quat = [float(x) for x in quats[name]]
        else:
            quat = [1.0, 0.0, 0.0, 0.0]
        nodes.append({
            "name": name,
            "pos": [float(pos[0]), float(pos[1]), float(pos[2])],
            "quat_wxyz": quat,
        })
    return nodes


def run_preview_retarget(
    *,
    xml_path: str,
    bvh_path: str,
    config: dict,
    bvh_unit: int | None = None,
    max_frames: int = 50,
) -> dict[str, Any]:
    """Retarget the first ``max_frames`` BVH frames and return skeleton JSON."""
    from general_motion_retargeting.params import ROBOT_BASE_DICT, ROBOT_XML_DICT
    from general_motion_retargeting import GeneralMotionRetargeting as GMR
    from tools.retargeting_visualizer.visualize_retargeting import (
        get_human_edges,
        get_robot_body_info,
        get_robot_edges,
        load_bvh_all_frames,
    )

    unit = int(bvh_unit if bvh_unit is not None else config.get("bvh_unit", 1))
    if unit not in BVH_UNIT_OPTIONS:
        raise ValueError(f"Unsupported bvh_unit={unit}")
    scale = float(BVH_UNIT_OPTIONS[unit]["scale"])

    xml = resolve_under_project(xml_path)
    bvh = resolve_under_project(bvh_path)
    if not xml.exists():
        raise FileNotFoundError(f"Robot XML not found: {xml}")
    if not bvh.exists():
        raise FileNotFoundError(f"BVH file not found: {bvh}")

    human_frames, actual_human_height, frame_time, parser = load_bvh_all_frames(
        bvh, start=0, end=int(max_frames), scale=scale,
    )
    if not human_frames:
        raise RuntimeError("No BVH frames loaded")

    human_edges = get_human_edges(parser)

    # Write config to a temp file — GMR loads ik_config from path
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8",
    ) as fh:
        import json
        json.dump(config, fh, indent=2)
        tmp_ik = Path(fh.name)

    custom_key = "_retargeting_app_stage3"
    ROBOT_XML_DICT[custom_key] = str(xml)
    # Prefer config robot root as base if present
    base = config.get("robot_root_name") or "base_link"
    if base in ("aux_base", "pelvis"):
        # MuJoCo freejoint usually on base_link; keep visualizer default
        ROBOT_BASE_DICT[custom_key] = "base_link"
    else:
        ROBOT_BASE_DICT[custom_key] = base

    try:
        retargeter = GMR(
            src_human="bvh_xsens",
            tgt_robot=custom_key,
            actual_human_height=float(actual_human_height),
            ik_config_path=str(tmp_ik),
            verbose=False,
        )
        model = retargeter.model
        data0 = mj.MjData(model)
        mj.mj_forward(model, data0)
        robot_edges = get_robot_edges(get_robot_body_info(model, data0))

        human_root = config.get("human_root_name") or "Hips"
        robot_root = (
            config.get("robot_root_name")
            or ROBOT_BASE_DICT.get(custom_key)
            or "base_link"
        )

        all_frame_data = _process_frames_for_stage3(
            retargeter, human_frames, model,
            human_root=human_root,
            robot_root=robot_root,
        )
    finally:
        try:
            tmp_ik.unlink(missing_ok=True)
        except Exception:
            pass

    frames_out = []
    for fd in all_frame_data:
        human_quats = {
            name: val[1] for name, val in fd["human_data"].items()
        }
        frames_out.append({
            "robot": {
                "nodes": _nodes_from_positions(
                    fd["robot_positions"], xmats=fd["robot_xmat"],
                ),
                "edges": [[a, b] for a, b in robot_edges if a and b],
            },
            "human": {
                "nodes": _nodes_from_positions(
                    fd["human_positions"], quats=human_quats,
                ),
                "edges": [[a, b] for a, b in human_edges],
            },
        })

    fps = 1.0 / float(frame_time) if frame_time else 120.0
    return {
        "n_frames": len(frames_out),
        "max_frames": int(max_frames),
        "fps": fps,
        "actual_human_height": float(actual_human_height),
        "xml_path": xml.relative_to(PROJECT_ROOT).as_posix()
        if xml.is_relative_to(PROJECT_ROOT) else str(xml),
        "bvh_path": bvh.relative_to(PROJECT_ROOT).as_posix()
        if bvh.is_relative_to(PROJECT_ROOT) else str(bvh),
        "weight_rows": extract_weight_rows(config.get("ik_match_table1", {})),
        "offset_rows": extract_offset_rows(config.get("ik_match_table1", {})),
        "frames": frames_out,
    }


def _process_frames_for_stage3(
    retargeter,
    human_frames,
    model,
    *,
    human_root: str,
    robot_root: str,
):
    """Retarget frames; display human IK targets with roots co-located at origin.

    Differences vs ``process_all_frames`` (CLI visualizer):
    - Human overlay uses the same ground-offset poses that IK solved against.
    - Each frame is shifted so the human root sits at the world origin (robot
      gets the same delta) — matches Stage-2 comparison UX and avoids a huge
      world bbox that makes proportions look wrong.
    """
    all_data = []
    data_fk = mj.MjData(model)

    print(f"\n  Processing {len(human_frames)} frames ...")
    for fi, human_data in enumerate(human_frames):
        qpos = retargeter.retarget(human_data, offset_to_ground=True)

        data_fk.qpos = qpos
        mj.mj_forward(model, data_fk)
        robot_positions = {}
        robot_xmat = {}
        for i in range(model.nbody):
            name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i)
            if name and name != "world":
                robot_positions[name] = data_fk.xpos[i].copy()
                robot_xmat[name] = data_fk.xmat[i].copy().reshape(3, 3)

        # IK targets (scaled + offset + ground), not pre-ground scaled_human_data
        human_for_ik = retargeter.offset_human_data_to_ground(
            {
                name: [val[0].copy(), val[1].copy()]
                for name, val in retargeter.scaled_human_data.items()
            }
        )

        # Co-locate human root at origin for display (same delta for robot)
        if human_root in human_for_ik:
            origin = np.asarray(human_for_ik[human_root][0], dtype=float).copy()
        else:
            # fallback: first available joint
            origin = np.asarray(next(iter(human_for_ik.values()))[0], dtype=float).copy()

        human_positions = {
            name: np.asarray(val[0], dtype=float) - origin
            for name, val in human_for_ik.items()
        }
        human_data_out = {
            name: [human_positions[name], np.asarray(val[1], dtype=float).copy()]
            for name, val in human_for_ik.items()
        }
        robot_positions = {
            name: pos - origin for name, pos in robot_positions.items()
        }

        all_data.append({
            "qpos": qpos.copy(),
            "robot_positions": robot_positions,
            "robot_xmat": robot_xmat,
            "human_positions": human_positions,
            "human_data": human_data_out,
        })

        if (fi + 1) % 100 == 0:
            print(f"    ... {fi + 1}/{len(human_frames)} frames")

    print(f"  Done. {len(all_data)} frames processed.")
    return all_data
