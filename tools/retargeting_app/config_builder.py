"""Build / merge ik_config JSON for Stage-1 (mapping + BVH unit)."""

from __future__ import annotations

import copy
import json
import pathlib
from typing import Any

from .loaders import BVH_UNIT_OPTIONS, PROJECT_ROOT, resolve_under_project

# Defaults for fields not tuned in Stage-1.
DEFAULT_IK_CONFIG: dict[str, Any] = {
    "robot_root_name": "pelvis",
    "human_root_name": "Hips",
    "ground_height": 0.0,
    "human_height_assumption": 1.8,
    "bvh_unit": 1,  # Stage-1 extension: BVH unit divisor (100=cm, 1000=mm, 1=m)
    "bvh_scale": 1.0,  # derived: 1 / bvh_unit
    "use_ik_match_table1": True,
    "use_ik_match_table2": False,
    "use_frame_joint_damper": False,
    "offset_to_ground": False,
    "human_scale_table_vertical": {},
    "human_scale_table_horizontal": {},
    "ik_match_table1": {},
    "ik_match_table2": {},
}

# Default IK entry when user only picks a bone mapping in Stage-1.
DEFAULT_POS_WEIGHT = 10
DEFAULT_ROT_WEIGHT = 10
DEFAULT_POS_OFFSET = [0.0, 0.0, 0.0]
DEFAULT_ROT_QUAT = [1.0, 0.0, 0.0, 0.0]


def _default_entry(human_bone: str) -> list:
    return [
        human_bone,
        DEFAULT_POS_WEIGHT,
        DEFAULT_ROT_WEIGHT,
        list(DEFAULT_POS_OFFSET),
        list(DEFAULT_ROT_QUAT),
    ]


def infer_roots(mapping: dict[str, str], robot_names: list[str], human_names: list[str]) -> tuple[str, str]:
    """Pick reasonable robot/human roots from mapping or name heuristics."""
    robot_root = "pelvis"
    human_root = "Hips"

    # Prefer explicit mapping of common roots
    for cand in ("aux_base", "pelvis", "base_link", "root"):
        if cand in mapping:
            robot_root = cand
            human_root = mapping[cand]
            break
        if cand in robot_names:
            robot_root = cand
            break

    for cand in ("Hips", "pelvis", "Root", "Hip"):
        if cand in human_names:
            human_root = cand
            break

    return robot_root, human_root


def build_default_scale_tables(human_names: list[str]) -> tuple[dict[str, float], dict[str, float]]:
    vertical = {name: 1.0 for name in human_names}
    horizontal = {name: 1.0 for name in human_names}
    return vertical, horizontal


def mapping_from_ik_table(ik_match_table: dict) -> dict[str, str]:
    """Extract robot_body -> human_bone from ik_match_table1."""
    out: dict[str, str] = {}
    for robot_body, entry in (ik_match_table or {}).items():
        if not entry:
            continue
        # Multi-entry or single entry
        if isinstance(entry[0], list):
            human_bone = entry[0][0]
        else:
            human_bone = entry[0]
        out[robot_body] = human_bone
    return out


def apply_mapping_to_table(
    existing_table: dict | None,
    mapping: dict[str, str],
) -> dict:
    """Update ik_match_table with Stage-1 mapping; preserve weights/offsets when possible."""
    existing_table = existing_table or {}
    new_table: dict[str, list] = {}

    for robot_body, human_bone in mapping.items():
        if not human_bone:
            continue
        old = existing_table.get(robot_body)
        if old is None:
            new_table[robot_body] = _default_entry(human_bone)
            continue

        # Preserve multi-entry structure if present, but update first human name.
        if isinstance(old[0], list):
            entries = copy.deepcopy(old)
            entries[0][0] = human_bone
            new_table[robot_body] = entries
        else:
            entry = copy.deepcopy(old)
            entry[0] = human_bone
            # Ensure length / types
            if len(entry) < 5:
                entry = _default_entry(human_bone)
            new_table[robot_body] = entry

    return new_table


def build_config(
    *,
    mapping: dict[str, str],
    bvh_unit: int,
    robot_names: list[str] | None = None,
    human_names: list[str] | None = None,
    base_config: dict | None = None,
    robot_root_name: str | None = None,
    human_root_name: str | None = None,
) -> dict[str, Any]:
    """Assemble a full ik_config dict; fill unspecified fields with defaults."""
    if bvh_unit not in BVH_UNIT_OPTIONS:
        raise ValueError(f"Unsupported bvh_unit={bvh_unit}")

    cfg = copy.deepcopy(DEFAULT_IK_CONFIG)
    if base_config:
        # Shallow-merge known keys from an existing config
        for k, v in base_config.items():
            cfg[k] = copy.deepcopy(v)

    robot_names = robot_names or []
    human_names = human_names or list(
        (cfg.get("human_scale_table_vertical") or {}).keys()
    )

    r_root, h_root = infer_roots(mapping, robot_names, human_names)
    cfg["robot_root_name"] = robot_root_name or cfg.get("robot_root_name") or r_root
    cfg["human_root_name"] = human_root_name or cfg.get("human_root_name") or h_root

    cfg["bvh_unit"] = int(bvh_unit)
    cfg["bvh_scale"] = float(BVH_UNIT_OPTIONS[bvh_unit]["scale"])

    cfg["ik_match_table1"] = apply_mapping_to_table(
        cfg.get("ik_match_table1"), mapping
    )

    # Ensure empty table2 exists
    if "ik_match_table2" not in cfg or cfg["ik_match_table2"] is None:
        cfg["ik_match_table2"] = {}

    # Fill scale tables with 1.0 for any mapped human bones still missing
    mapped_humans = sorted(set(mapping.values()))
    for table_key in ("human_scale_table_vertical", "human_scale_table_horizontal"):
        table = cfg.get(table_key) or {}
        for name in mapped_humans:
            table.setdefault(name, 1.0)
        # Also include all known human names at 1.0 if table was empty
        if not table and human_names:
            table = {n: 1.0 for n in human_names}
        cfg[table_key] = table

    # Legacy unified table: keep if present, else omit (prefer split tables)
    if "human_scale_table" in cfg and not cfg["human_scale_table"]:
        del cfg["human_scale_table"]

    cfg.setdefault("ground_height", 0.0)
    cfg.setdefault("human_height_assumption", 1.8)
    cfg.setdefault("use_ik_match_table1", True)
    cfg.setdefault("use_ik_match_table2", False)
    cfg.setdefault("use_frame_joint_damper", False)
    cfg.setdefault("offset_to_ground", False)

    return cfg


def apply_solver_flags(cfg: dict, payload: dict | None = None) -> dict:
    """Write damper / offset_to_ground flags into an ik_config dict."""
    cfg = copy.deepcopy(cfg)
    payload = payload or {}
    for key, default in (
        ("use_frame_joint_damper", False),
        ("offset_to_ground", False),
    ):
        if key in payload and payload[key] is not None:
            cfg[key] = bool(payload[key])
        else:
            cfg.setdefault(key, default)
    return cfg


def load_config_file(path: str) -> dict:
    p = resolve_under_project(path)
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_config_file(config: dict, filename: str, dest_dir: str | None = None) -> str:
    """Save config JSON. Returns project-relative path."""
    if dest_dir:
        out_dir = resolve_under_project(dest_dir)
    else:
        out_dir = PROJECT_ROOT / "tools" / "retargeting_app" / "saved_configs"

    out_dir.mkdir(parents=True, exist_ok=True)
    name = pathlib.Path(filename).name
    if not name.endswith(".json"):
        name += ".json"
    out_path = out_dir / name

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=4, ensure_ascii=False)
        fh.write("\n")

    try:
        return out_path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(out_path)
