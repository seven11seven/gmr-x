"""Load robot MuJoCo bodies and BVH skeleton frames for the web app."""

from __future__ import annotations

import pathlib
import sys
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

_HERE = pathlib.Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Reuse battle-tested loaders from the existing links-mapping tool.
from tools.links_mapping.visualize_links_mapping import (  # noqa: E402
    detect_bvh_human_height,
    get_robot_body_edges,
    get_robot_body_info,
    load_bvh_first_frame,
    load_robot_model,
)

PROJECT_ROOT = _PROJECT_ROOT

# UI unit divisor → multiplier applied to BVH positions (to meters).
BVH_UNIT_OPTIONS = {
    100: {"label": "cm → m (÷100)", "scale": 0.01},
    1000: {"label": "mm → m (÷1000)", "scale": 0.001},
    1: {"label": "m → m (÷1)", "scale": 1.0},
}


def resolve_under_project(path: str | pathlib.Path) -> pathlib.Path:
    p = pathlib.Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


def list_robot_xmls() -> list[dict[str, str]]:
    roots = [
        PROJECT_ROOT / "assets",
    ]
    results: list[dict[str, str]] = []
    for root in roots:
        if not root.exists():
            continue
        for p in sorted(root.rglob("*.xml")):
            # Skip mesh/include fragments when possible: keep mjcf-looking files
            rel = p.relative_to(PROJECT_ROOT).as_posix()
            if any(skip in rel for skip in ("/meshes/", "/texture", ".obj")):
                continue
            results.append({"path": rel, "name": p.name})
    return results


def list_bvh_files() -> list[dict[str, str]]:
    roots = [
        PROJECT_ROOT / "datas",
    ]
    results: list[dict[str, str]] = []
    for root in roots:
        if not root.exists():
            continue
        for p in sorted(root.rglob("*.bvh")):
            rel = p.relative_to(PROJECT_ROOT).as_posix()
            results.append({"path": rel, "name": p.name})
    return results


def list_ik_configs() -> list[dict[str, str]]:
    roots = [
        PROJECT_ROOT / "general_motion_retargeting" / "ik_configs",
        _HERE / "saved_configs",
    ]
    results: list[dict[str, str]] = []
    for root in roots:
        if not root.exists():
            continue
        for p in sorted(root.glob("*.json")):
            rel = p.relative_to(PROJECT_ROOT).as_posix()
            results.append({"path": rel, "name": p.name, "dir": root.name})
    return results


def _mat_to_quat_wxyz(mat: np.ndarray) -> list[float]:
    q = R.from_matrix(mat).as_quat(scalar_first=True)
    return [float(x) for x in q]


def load_scene(
    xml_path: str,
    bvh_path: str,
    bvh_unit: int = 1,
) -> dict[str, Any]:
    """Load robot bodies + BVH first frame for Stage-1 visualization.

    Args:
        xml_path: robot MuJoCo XML (project-relative or absolute)
        bvh_path: BVH motion file
        bvh_unit: divisor converting BVH units to meters (100/1000/1)
    """
    if bvh_unit not in BVH_UNIT_OPTIONS:
        raise ValueError(
            f"Unsupported bvh_unit={bvh_unit}. "
            f"Choose one of {sorted(BVH_UNIT_OPTIONS)}"
        )
    scale = BVH_UNIT_OPTIONS[bvh_unit]["scale"]

    xml = resolve_under_project(xml_path)
    bvh = resolve_under_project(bvh_path)
    if not xml.exists():
        raise FileNotFoundError(f"Robot XML not found: {xml}")
    if not bvh.exists():
        raise FileNotFoundError(f"BVH file not found: {bvh}")

    model, data = load_robot_model(xml)
    bodies = get_robot_body_info(model, data)
    robot_edges = get_robot_body_edges(bodies)

    human_data, human_edges = load_bvh_first_frame(bvh, scale=scale)
    detected_height = detect_bvh_human_height(human_data)

    robot_nodes = []
    for b in bodies:
        pos = b["pos"]
        robot_nodes.append({
            "name": b["name"],
            "pos": [float(pos[0]), float(pos[1]), float(pos[2])],
            "quat_wxyz": _mat_to_quat_wxyz(b["xmat"]),
            "parent": b["parent_name"],
        })

    human_nodes = []
    for name, (pos, quat) in human_data.items():
        human_nodes.append({
            "name": name,
            "pos": [float(pos[0]), float(pos[1]), float(pos[2])],
            "quat_wxyz": [float(x) for x in quat],
        })

    return {
        "xml_path": xml.relative_to(PROJECT_ROOT).as_posix()
        if xml.is_relative_to(PROJECT_ROOT) else str(xml),
        "bvh_path": bvh.relative_to(PROJECT_ROOT).as_posix()
        if bvh.is_relative_to(PROJECT_ROOT) else str(bvh),
        "bvh_unit": bvh_unit,
        "bvh_scale": scale,
        "detected_human_height": detected_height,
        "robot": {
            "nodes": robot_nodes,
            "edges": [[a, b] for a, b in robot_edges if a and b],
        },
        "human": {
            "nodes": human_nodes,
            "edges": [[a, b] for a, b in human_edges],
        },
    }
