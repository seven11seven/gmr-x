"""Stage-2: apply scale / offset and align robot↔human roots for visualization."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

from tools.links_mapping.visualize_links_mapping import (
    detect_bvh_human_height,
    process_human_data,
)
from tools.retargeting_app.loaders import (
    BVH_UNIT_OPTIONS,
    load_scene,
    resolve_under_project,
)


def quat_wxyz_to_matrix(quat_wxyz) -> list[list[float]]:
    mat = R.from_quat(quat_wxyz, scalar_first=True).as_matrix()
    return [[float(mat[i, j]) for j in range(3)] for i in range(3)]


def matrix_to_quat_wxyz(matrix) -> list[float]:
    """Convert a 3×3 rotation matrix to quaternion (wxyz).

    Orthonormalizes via SVD so slightly-noisy engineer inputs still work.
    """
    m = np.asarray(matrix, dtype=float).reshape(3, 3)
    u, _, vt = np.linalg.svd(m)
    m_orth = u @ vt
    if np.linalg.det(m_orth) < 0:
        u[:, -1] *= -1
        m_orth = u @ vt
    q = R.from_matrix(m_orth).as_quat(scalar_first=True)
    return [float(x) for x in q]


def extract_offset_rows(ik_match_table: dict) -> list[dict[str, Any]]:
    """One editable row per robot body in ik_match_table1."""
    rows = []
    for robot_body, entry in (ik_match_table or {}).items():
        if not entry:
            continue
        single = entry[0] if isinstance(entry[0], list) else entry
        human = single[0]
        pos_w = float(single[1]) if len(single) > 1 else 10.0
        rot_w = float(single[2]) if len(single) > 2 else 10.0
        pos_off = list(single[3]) if len(single) > 3 else [0.0, 0.0, 0.0]
        quat = list(single[4]) if len(single) > 4 else [1.0, 0.0, 0.0, 0.0]
        rows.append({
            "robot_body": robot_body,
            "human_bone": human,
            "pos_weight": pos_w,
            "rot_weight": rot_w,
            "pos_offset": [float(x) for x in pos_off],
            "rot_quat_wxyz": [float(x) for x in quat],
            "rot_matrix": quat_wxyz_to_matrix(quat),
        })
    rows.sort(key=lambda r: (r["human_bone"], r["robot_body"]))
    return rows


def apply_offset_rows_to_table(
    base_table: dict | None,
    offset_rows: list[dict],
) -> dict:
    """Write pos/rot offsets from UI rows back into ik_match_table1.

    Rotation may be provided as rot_matrix (preferred) or rot_quat_wxyz.
    Weights and multi-entry structure are preserved when possible.
    """
    table = copy.deepcopy(base_table or {})
    by_robot = {r["robot_body"]: r for r in offset_rows if r.get("robot_body")}

    for robot_body, row in by_robot.items():
        human = row.get("human_bone")
        pos_off = [float(x) for x in row.get("pos_offset", [0, 0, 0])]
        if row.get("rot_matrix") is not None:
            quat = matrix_to_quat_wxyz(row["rot_matrix"])
        else:
            quat = [float(x) for x in row.get("rot_quat_wxyz", [1, 0, 0, 0])]

        old = table.get(robot_body)
        if old is None:
            table[robot_body] = [
                human,
                float(row.get("pos_weight", 10)),
                float(row.get("rot_weight", 10)),
                pos_off,
                quat,
            ]
            continue

        if isinstance(old[0], list):
            entries = copy.deepcopy(old)
            entries[0][0] = human or entries[0][0]
            entries[0][3] = pos_off
            entries[0][4] = quat
            table[robot_body] = entries
        else:
            entry = copy.deepcopy(old)
            if human:
                entry[0] = human
            while len(entry) < 5:
                entry.append(None)
            entry[3] = pos_off
            entry[4] = quat
            table[robot_body] = entry

    return table


def _shift_nodes(nodes: list[dict], delta: np.ndarray) -> list[dict]:
    out = []
    for n in nodes:
        pos = np.asarray(n["pos"], dtype=float) + delta
        out.append({
            **n,
            "pos": [float(pos[0]), float(pos[1]), float(pos[2])],
        })
    return out


def build_stage2_scene(
    *,
    xml_path: str,
    bvh_path: str,
    config: dict,
    bvh_unit: int | None = None,
    apply_height_ratio: bool = True,
) -> dict[str, Any]:
    """Load robot + BVH, apply scale/offset, align roots to origin."""
    unit = int(bvh_unit if bvh_unit is not None else config.get("bvh_unit", 1))
    if unit not in BVH_UNIT_OPTIONS:
        raise ValueError(f"Unsupported bvh_unit={unit}")

    raw = load_scene(xml_path, bvh_path, bvh_unit=unit)

    # Rebuild human_data dict expected by process_human_data
    human_data = {
        n["name"]: [np.asarray(n["pos"], dtype=float),
                    np.asarray(n["quat_wxyz"], dtype=float)]
        for n in raw["human"]["nodes"]
    }

    ratio = 1.0
    detected_height = detect_bvh_human_height(human_data)
    assumption = float(config.get("human_height_assumption", 1.8))
    if apply_height_ratio:
        actual = detected_height if detected_height is not None else assumption
        ratio = actual / assumption if assumption else 1.0

    processed, human_to_robot = process_human_data(human_data, config, ratio=ratio)

    human_root = config.get("human_root_name", "Hips")
    robot_root = config.get("robot_root_name", "pelvis")

    if human_root not in processed:
        raise ValueError(f"human_root_name '{human_root}' not in processed skeleton")

    # Align both roots to world origin for comparison
    h_root_pos = np.asarray(processed[human_root][0], dtype=float)
    human_nodes = []
    for name, (pos, quat) in processed.items():
        p = np.asarray(pos, dtype=float) - h_root_pos
        human_nodes.append({
            "name": name,
            "pos": [float(p[0]), float(p[1]), float(p[2])],
            "quat_wxyz": [float(x) for x in quat],
        })

    robot_by_name = {n["name"]: n for n in raw["robot"]["nodes"]}
    if robot_root not in robot_by_name:
        # fallback: first body / pelvis-like
        for cand in ("aux_base", "pelvis", "base_link"):
            if cand in robot_by_name:
                robot_root = cand
                break
        else:
            robot_root = raw["robot"]["nodes"][0]["name"]

    r_root_pos = np.asarray(robot_by_name[robot_root]["pos"], dtype=float)
    robot_nodes = _shift_nodes(raw["robot"]["nodes"], -r_root_pos)

    robot_mapped = set()
    human_mapped = set(human_to_robot.keys())
    for h_name, r_names in human_to_robot.items():
        robot_mapped.update(r_names)

    return {
        "xml_path": raw["xml_path"],
        "bvh_path": raw["bvh_path"],
        "bvh_unit": unit,
        "bvh_scale": BVH_UNIT_OPTIONS[unit]["scale"],
        "detected_human_height": detected_height,
        "height_ratio": ratio,
        "human_height_assumption": assumption,
        "robot_root_name": robot_root,
        "human_root_name": human_root,
        "robot": {
            "nodes": robot_nodes,
            "edges": raw["robot"]["edges"],
        },
        "human": {
            "nodes": human_nodes,
            "edges": raw["human"]["edges"],
        },
        "robot_mapped": sorted(robot_mapped),
        "human_mapped": sorted(human_mapped),
        "offset_rows": extract_offset_rows(config.get("ik_match_table1", {})),
        "scale_vertical": dict(config.get("human_scale_table_vertical") or {}),
        "scale_horizontal": dict(config.get("human_scale_table_horizontal") or {}),
        # legacy fallback
        "scale_unified": dict(config.get("human_scale_table") or {}),
    }


def merge_stage2_edits(
    base_config: dict,
    *,
    scale_vertical: dict | None = None,
    scale_horizontal: dict | None = None,
    offset_rows: list[dict] | None = None,
    robot_root_name: str | None = None,
    human_root_name: str | None = None,
    ground_height: float | None = None,
    human_height_assumption: float | None = None,
    bvh_unit: int | None = None,
) -> dict:
    """Return a new config with Stage-2 edits applied."""
    cfg = copy.deepcopy(base_config)

    if scale_vertical is not None:
        cfg["human_scale_table_vertical"] = {
            k: float(v) for k, v in scale_vertical.items()
        }
    if scale_horizontal is not None:
        cfg["human_scale_table_horizontal"] = {
            k: float(v) for k, v in scale_horizontal.items()
        }
    if offset_rows is not None:
        cfg["ik_match_table1"] = apply_offset_rows_to_table(
            cfg.get("ik_match_table1"), offset_rows
        )
    if robot_root_name:
        cfg["robot_root_name"] = robot_root_name
    if human_root_name:
        cfg["human_root_name"] = human_root_name
    if ground_height is not None:
        cfg["ground_height"] = float(ground_height)
    if human_height_assumption is not None:
        cfg["human_height_assumption"] = float(human_height_assumption)
    if bvh_unit is not None:
        unit = int(bvh_unit)
        if unit not in BVH_UNIT_OPTIONS:
            raise ValueError(f"Unsupported bvh_unit={unit}")
        cfg["bvh_unit"] = unit
        cfg["bvh_scale"] = float(BVH_UNIT_OPTIONS[unit]["scale"])

    return cfg
