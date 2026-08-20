"""Auto-estimate Stage-2 scale tables and rotation offsets from T-pose."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

from tools.retargeting_app.loaders import load_scene
from tools.retargeting_app.stage2_logic import (
    extract_offset_rows,
    matrix_to_quat_wxyz,
)

# ---------------------------------------------------------------------------
# Landmark name candidates
# ---------------------------------------------------------------------------

ROBOT_ROOT_CANDIDATES = ("aux_base", "pelvis", "base_link", "trunk", "root")
HUMAN_ROOT_CANDIDATES = ("Hips", "pelvis", "Hip", "Root")

ROBOT_ANKLE_L = ("aux_left_ankle_end", "left_ankle_roll_link", "left_ankle_pitch_link")
ROBOT_ANKLE_R = ("aux_right_ankle_end", "right_ankle_roll_link", "right_ankle_pitch_link")
HUMAN_ANKLE_L = ("LeftFootMod", "LeftAnkle", "LeftFoot")
HUMAN_ANKLE_R = ("RightFootMod", "RightAnkle", "RightFoot")

ROBOT_SHOULDER_L = (
    "left_shoulder_pitch_link", "left_shoulder_yaw_link", "left_shoulder_roll_link",
    "aux_left_collar",
)
ROBOT_SHOULDER_R = (
    "right_shoulder_pitch_link", "right_shoulder_yaw_link", "right_shoulder_roll_link",
    "aux_right_collar",
)
HUMAN_SHOULDER_L = ("LeftShoulder", "LeftCollar")
HUMAN_SHOULDER_R = ("RightShoulder", "RightCollar")

ROBOT_ELBOW_L = ("aux_left_elbow", "left_elbow_pitch_link", "left_elbow_link")
ROBOT_ELBOW_R = ("aux_right_elbow", "right_elbow_pitch_link", "right_elbow_link")
HUMAN_ELBOW_L = ("LeftElbow",)
HUMAN_ELBOW_R = ("RightElbow",)

# Human bone groups for scale assignment
LOWER_BODY_TOKENS = (
    "hip", "knee", "ankle", "toe", "foot", "hips", "pelvis",
)
UPPER_LIMB_TOKENS = (
    "shoulder", "elbow", "wrist", "hand", "collar",
)
TORSO_TOKENS = (
    "chest", "spine", "neck", "head", "torso", "thorax",
)


def _find_name(names: set[str] | list[str], candidates: tuple[str, ...]) -> str | None:
    name_set = set(names)
    for c in candidates:
        if c in name_set:
            return c
    # case-insensitive fallback
    lower_map = {n.lower(): n for n in name_set}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def _pos_of(nodes_by_name: dict, name: str) -> np.ndarray:
    return np.asarray(nodes_by_name[name]["pos"], dtype=float)


def _quat_mat(nodes_by_name: dict, name: str) -> np.ndarray:
    return R.from_quat(nodes_by_name[name]["quat_wxyz"], scalar_first=True).as_matrix()


def snap_rotation_matrix(mat: np.ndarray) -> np.ndarray:
    """Project a rotation to the nearest matrix with entries in {0, ±1}."""
    m = np.asarray(mat, dtype=float).reshape(3, 3)
    out = np.zeros((3, 3), dtype=float)
    used_axes: set[int] = set()
    for j in range(3):
        col = m[:, j]
        best = None
        best_score = -1e9
        for axis in range(3):
            if axis in used_axes:
                continue
            for sgn in (1.0, -1.0):
                score = float(col[axis] * sgn)
                if score > best_score:
                    best_score = score
                    best = (axis, sgn)
        assert best is not None
        axis, sgn = best
        used_axes.add(axis)
        out[axis, j] = sgn
    if np.linalg.det(out) < 0:
        out[:, 2] *= -1.0
    return out


def _classify_human_bone(name: str) -> str:
    """Return 'lower' | 'upper_limb' | 'torso' | 'other'."""
    n = name.lower()
    for t in LOWER_BODY_TOKENS:
        if t in n:
            return "lower"
    for t in UPPER_LIMB_TOKENS:
        if t in n:
            return "upper_limb"
    for t in TORSO_TOKENS:
        if t in n:
            return "torso"
    return "other"


def _safe_ratio(num: float, den: float, default: float = 1.0) -> float:
    if abs(den) < 1e-8:
        return default
    r = float(num / den)
    if not np.isfinite(r) or r <= 0:
        return default
    return r


def collect_mapping_ambiguities(ik_match_table: dict) -> list[dict[str, Any]]:
    """Robot bodies that map to more than one human bone."""
    amb = []
    for robot_body, entry in (ik_match_table or {}).items():
        if not entry:
            continue
        if isinstance(entry[0], list):
            humans = []
            for e in entry:
                if e and e[0] not in humans:
                    humans.append(e[0])
            if len(humans) > 1:
                amb.append({
                    "robot_body": robot_body,
                    "candidates": humans,
                    "kind": "robot_to_many_humans",
                })
    # Also: same human targeted by many robots — ask which robot orientation to use
    human_to_robots: dict[str, list[str]] = {}
    for robot_body, entry in (ik_match_table or {}).items():
        if not entry:
            continue
        single = entry[0] if isinstance(entry[0], list) else entry
        human = single[0]
        human_to_robots.setdefault(human, []).append(robot_body)
    for human, robots in human_to_robots.items():
        if len(robots) > 1:
            amb.append({
                "human_bone": human,
                "candidates": robots,
                "kind": "human_to_many_robots",
            })
    return amb


def _resolve_pairs(
    ik_match_table: dict,
    resolutions: dict | None,
) -> tuple[list[tuple[str, str]], dict[str, str], dict[str, list[str]]]:
    """Return (pairs, human_chosen_robot, human_to_robots).

    resolutions keys:
      - robot_body -> human_bone  (for robot_to_many_humans)
      - \"human::LeftShoulder\" -> robot_body  (for human_to_many_robots)
    """
    resolutions = resolutions or {}

    robot_human: dict[str, str] = {}
    for robot_body, entry in (ik_match_table or {}).items():
        if not entry:
            continue
        if isinstance(entry[0], list):
            humans = [e[0] for e in entry if e]
            if len(humans) > 1:
                chosen = resolutions.get(robot_body)
                if chosen not in humans:
                    continue
                robot_human[robot_body] = chosen
            else:
                robot_human[robot_body] = humans[0]
        else:
            robot_human[robot_body] = entry[0]

    human_to_robots: dict[str, list[str]] = {}
    for rb, hb in robot_human.items():
        human_to_robots.setdefault(hb, []).append(rb)

    human_chosen_robot: dict[str, str] = {}
    for human, robots in human_to_robots.items():
        if len(robots) > 1:
            key = f"human::{human}"
            chosen = resolutions.get(key) or resolutions.get(human)
            if chosen in robots:
                human_chosen_robot[human] = chosen
        else:
            human_chosen_robot[human] = robots[0]

    pairs = list(robot_human.items())
    return pairs, human_chosen_robot, human_to_robots


def estimate_scales(
    robot_nodes: list[dict],
    human_nodes: list[dict],
    *,
    robot_root_name: str | None,
    human_root_name: str | None,
    bone_names: list[str],
    height_ratio: float = 1.0,
) -> tuple[dict[str, float], dict[str, float], dict[str, Any]]:
    """Estimate V/H scale tables from raw T-pose geometry.

    Important: inputs must be **unscaled / unoffset** skeletons (``load_scene``).
    Runtime / Stage-2 viz multiplies table values by
    ``height_ratio = detected_height / human_height_assumption``, so the
    returned table stores ``(robot_metric / human_metric) / height_ratio``
    such that the *effective* scale matches ``robot / human``.
    """
    r_by = {n["name"]: n for n in robot_nodes}
    h_by = {n["name"]: n for n in human_nodes}
    r_names = list(r_by.keys())
    h_names = list(h_by.keys())

    r_root = robot_root_name if robot_root_name in r_by else _find_name(r_names, ROBOT_ROOT_CANDIDATES)
    h_root = human_root_name if human_root_name in h_by else _find_name(h_names, HUMAN_ROOT_CANDIDATES)
    if r_root is None or h_root is None:
        raise ValueError(f"Cannot find roots: robot={r_root}, human={h_root}")

    r_al = _find_name(r_names, ROBOT_ANKLE_L)
    r_ar = _find_name(r_names, ROBOT_ANKLE_R)
    h_al = _find_name(h_names, HUMAN_ANKLE_L)
    h_ar = _find_name(h_names, HUMAN_ANKLE_R)
    r_sl = _find_name(r_names, ROBOT_SHOULDER_L)
    r_sr = _find_name(r_names, ROBOT_SHOULDER_R)
    h_sl = _find_name(h_names, HUMAN_SHOULDER_L)
    h_sr = _find_name(h_names, HUMAN_SHOULDER_R)
    r_el = _find_name(r_names, ROBOT_ELBOW_L)
    r_er = _find_name(r_names, ROBOT_ELBOW_R)
    h_el = _find_name(h_names, HUMAN_ELBOW_L)
    h_er = _find_name(h_names, HUMAN_ELBOW_R)

    missing = [k for k, v in {
        "robot_ankle_L": r_al, "robot_ankle_R": r_ar,
        "human_ankle_L": h_al, "human_ankle_R": h_ar,
        "robot_shoulder_L": r_sl, "robot_shoulder_R": r_sr,
        "human_shoulder_L": h_sl, "human_shoulder_R": h_sr,
        "robot_elbow_L": r_el, "robot_elbow_R": r_er,
        "human_elbow_L": h_el, "human_elbow_R": h_er,
    }.items() if v is None]
    if missing:
        raise ValueError("Missing landmarks for scale estimate: " + ", ".join(missing))

    # Vertical: ankle↔root and shoulder↔root height (Z) — RAW geometry
    r_ankle_z = 0.5 * (_pos_of(r_by, r_al)[2] + _pos_of(r_by, r_ar)[2])
    h_ankle_z = 0.5 * (_pos_of(h_by, h_al)[2] + _pos_of(h_by, h_ar)[2])
    r_root_z = _pos_of(r_by, r_root)[2]
    h_root_z = _pos_of(h_by, h_root)[2]
    r_sh_z = 0.5 * (_pos_of(r_by, r_sl)[2] + _pos_of(r_by, r_sr)[2])
    h_sh_z = 0.5 * (_pos_of(h_by, h_sl)[2] + _pos_of(h_by, h_sr)[2])

    lower_v_raw = _safe_ratio(abs(r_root_z - r_ankle_z), abs(h_root_z - h_ankle_z))
    upper_v_raw = _safe_ratio(abs(r_sh_z - r_root_z), abs(h_sh_z - h_root_z))

    def _xy_dist(a, b):
        d = _pos_of(a[0], a[1])[:2] - _pos_of(b[0], b[1])[:2]
        return float(np.linalg.norm(d))

    r_ankle_span = _xy_dist((r_by, r_al), (r_by, r_ar))
    h_ankle_span = _xy_dist((h_by, h_al), (h_by, h_ar))
    r_elbow_span = _xy_dist((r_by, r_el), (r_by, r_er))
    h_elbow_span = _xy_dist((h_by, h_el), (h_by, h_er))

    lower_h_raw = _safe_ratio(r_ankle_span, h_ankle_span)
    upper_h_raw = _safe_ratio(r_elbow_span, h_elbow_span)

    # Compensate height_ratio applied later in process_human_data / retargeting.
    hr = float(height_ratio) if height_ratio and abs(height_ratio) > 1e-8 else 1.0
    lower_v = lower_v_raw / hr
    upper_v = upper_v_raw / hr
    lower_h = lower_h_raw / hr
    upper_h = upper_h_raw / hr

    scale_v: dict[str, float] = {}
    scale_h: dict[str, float] = {}
    for name in bone_names:
        kind = _classify_human_bone(name)
        if kind == "lower":
            scale_v[name] = round(lower_v, 4)
            scale_h[name] = round(lower_h, 4)
        elif kind == "upper_limb":
            scale_v[name] = round(upper_v, 4)
            scale_h[name] = round(upper_h, 4)
        elif kind == "torso":
            scale_v[name] = round(upper_v, 4)
            scale_h[name] = 1.0  # 躯干 horizontal 保持为 1（config 值，不再补偿）
        else:
            scale_v[name] = round(upper_v, 4)
            scale_h[name] = 1.0

    if h_root in scale_v:
        scale_v[h_root] = round(lower_v, 4)
        scale_h[h_root] = round(lower_h, 4)

    meta = {
        "landmarks": {
            "robot_root": r_root, "human_root": h_root,
            "robot_ankle": [r_al, r_ar], "human_ankle": [h_al, h_ar],
            "robot_shoulder": [r_sl, r_sr], "human_shoulder": [h_sl, h_sr],
            "robot_elbow": [r_el, r_er], "human_elbow": [h_el, h_er],
        },
        "ratios_raw": {
            "lower_vertical": lower_v_raw,
            "upper_vertical": upper_v_raw,
            "lower_horizontal": lower_h_raw,
            "upper_horizontal": upper_h_raw,
        },
        "ratios": {
            "lower_vertical": lower_v,
            "upper_vertical": upper_v,
            "lower_horizontal": lower_h,
            "upper_horizontal": upper_h,
        },
        "height_ratio": hr,
        "spans": {
            "robot_ankle_span": r_ankle_span,
            "human_ankle_span": h_ankle_span,
            "robot_elbow_span": r_elbow_span,
            "human_elbow_span": h_elbow_span,
            "robot_root_ankle_dz": abs(r_root_z - r_ankle_z),
            "human_root_ankle_dz": abs(h_root_z - h_ankle_z),
            "robot_shoulder_root_dz": abs(r_sh_z - r_root_z),
            "human_shoulder_root_dz": abs(h_sh_z - h_root_z),
        },
        "used_config_scaled_skeleton": False,
    }
    return scale_v, scale_h, meta


def estimate_rot_matrix(robot_mat: np.ndarray, human_mat: np.ndarray) -> list[list[float]]:
    """Estimate panel rotation offset from T-pose world frames.

    World frames are snapped to {0,±1}, then
        R_offset = R_human_snap^T @ R_robot_snap
    so that R_human @ R_offset ≈ R_robot (matches IK offset convention).
    Uses raw (unoffset) orientations from ``load_scene``.
    """
    rs = snap_rotation_matrix(robot_mat)
    hs = snap_rotation_matrix(human_mat)
    off = hs.T @ rs
    off = snap_rotation_matrix(off)
    return [[float(off[i, j]) for j in range(3)] for i in range(3)]


def auto_estimate(
    *,
    xml_path: str,
    bvh_path: str,
    config: dict,
    bvh_unit: int | None = None,
    resolutions: dict | None = None,
) -> dict[str, Any]:
    """Estimate scales + rotation offsets from **raw** T-pose (no config scale/offset).

    Pos offsets are left unchanged. Scale tables compensate for the runtime
    ``height_ratio`` so effective scale ≈ robot/human metric ratio.
    """
    from tools.links_mapping.visualize_links_mapping import detect_bvh_human_height

    unit = int(bvh_unit if bvh_unit is not None else config.get("bvh_unit", 1))
    # RAW skeleton only — never run process_human_data here.
    raw = load_scene(xml_path, bvh_path, bvh_unit=unit)

    ambiguities = collect_mapping_ambiguities(config.get("ik_match_table1", {}))
    resolutions = dict(resolutions or {})

    unresolved = []
    for a in ambiguities:
        if a["kind"] == "robot_to_many_humans":
            if resolutions.get(a["robot_body"]) not in a["candidates"]:
                unresolved.append(a)
        elif a["kind"] == "human_to_many_robots":
            key = f"human::{a['human_bone']}"
            chosen = resolutions.get(key) or resolutions.get(a["human_bone"])
            if chosen not in a["candidates"]:
                unresolved.append(a)

    if unresolved:
        return {
            "needs_resolution": True,
            "ambiguities": unresolved,
            "all_ambiguities": ambiguities,
        }

    bone_names = sorted({
        *list((config.get("human_scale_table_vertical") or {}).keys()),
        *list((config.get("human_scale_table_horizontal") or {}).keys()),
        *[n["name"] for n in raw["human"]["nodes"]],
    })

    human_data = {
        n["name"]: [np.asarray(n["pos"], dtype=float),
                    np.asarray(n["quat_wxyz"], dtype=float)]
        for n in raw["human"]["nodes"]
    }
    detected = detect_bvh_human_height(human_data)
    assumption = float(config.get("human_height_assumption", 1.8) or 1.8)
    height_ratio = (detected / assumption) if detected is not None else 1.0

    scale_v, scale_h, scale_meta = estimate_scales(
        raw["robot"]["nodes"],
        raw["human"]["nodes"],
        robot_root_name=config.get("robot_root_name"),
        human_root_name=config.get("human_root_name"),
        bone_names=bone_names,
        height_ratio=height_ratio,
    )
    scale_meta["detected_human_height"] = detected
    scale_meta["human_height_assumption"] = assumption

    pairs, human_chosen_robot, human_to_robots = _resolve_pairs(
        config.get("ik_match_table1", {}), resolutions
    )

    r_by = {n["name"]: n for n in raw["robot"]["nodes"]}
    h_by = {n["name"]: n for n in raw["human"]["nodes"]}

    pair_mats: dict[tuple[str, str], list] = {}
    for rb, hb in pairs:
        if rb not in r_by or hb not in h_by:
            continue
        mat = estimate_rot_matrix(_quat_mat(r_by, rb), _quat_mat(h_by, hb))
        pair_mats[(rb, hb)] = mat

    chosen_human_mat: dict[str, list] = {}
    for human, robots in human_to_robots.items():
        if len(robots) <= 1:
            continue
        chosen_r = human_chosen_robot.get(human)
        if chosen_r and (chosen_r, human) in pair_mats:
            chosen_human_mat[human] = pair_mats[(chosen_r, human)]

    offset_rows = extract_offset_rows(config.get("ik_match_table1", {}))
    for row in offset_rows:
        rb = row["robot_body"]
        if rb in resolutions and resolutions[rb]:
            row["human_bone"] = resolutions[rb]
        hb = row["human_bone"]
        if hb in chosen_human_mat:
            row["rot_matrix"] = copy.deepcopy(chosen_human_mat[hb])
        elif (rb, hb) in pair_mats:
            row["rot_matrix"] = copy.deepcopy(pair_mats[(rb, hb)])
        row["rot_quat_wxyz"] = matrix_to_quat_wxyz(row["rot_matrix"])

    return {
        "needs_resolution": False,
        "ambiguities": [],
        "scale_vertical": scale_v,
        "scale_horizontal": scale_h,
        "offset_rows": offset_rows,
        "meta": {
            "scale": scale_meta,
            "pairs_estimated": [
                {"robot": rb, "human": hb} for rb, hb in pair_mats
            ],
            "human_chosen_robot": human_chosen_robot,
        },
    }
