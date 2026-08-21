"""Stage-3: fine-tune aux_* body local positions in robot MJCF XML."""

from __future__ import annotations

import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

from tools.links_mapping.visualize_links_mapping import (
    get_robot_body_edges,
    get_robot_body_info,
    load_robot_model,
)
from tools.retargeting_app.loaders import PROJECT_ROOT, resolve_under_project


AUX_PREFIX = "aux_"


def _parse_pos(attr: str | None) -> list[float]:
    if not attr or not str(attr).strip():
        return [0.0, 0.0, 0.0]
    parts = str(attr).replace(",", " ").split()
    vals = [float(p) for p in parts[:3]]
    while len(vals) < 3:
        vals.append(0.0)
    return vals


def _format_pos(xyz: list[float] | tuple[float, ...]) -> str:
    return " ".join(f"{float(v):g}" for v in xyz[:3])


def list_aux_bodies(xml_path: str | Path) -> list[dict[str, Any]]:
    """Return aux_* bodies with local pos and parent name (XML DFS order)."""
    path = resolve_under_project(xml_path)
    tree = ET.parse(path)
    root = tree.getroot()
    rows: list[dict[str, Any]] = []

    def walk(elem: ET.Element, parent_name: str | None):
        if elem.tag == "body":
            name = elem.get("name") or ""
            if name.startswith(AUX_PREFIX):
                rows.append({
                    "name": name,
                    "parent": parent_name,
                    "pos": _parse_pos(elem.get("pos")),
                })
            for child in elem:
                walk(child, name if name else parent_name)
        else:
            for child in elem:
                walk(child, parent_name)

    walk(root, None)
    return rows


def apply_aux_positions_to_tree(
    tree: ET.ElementTree,
    aux_positions: dict[str, list[float]],
) -> int:
    """Set body@pos for matching aux_* names. Returns number of bodies updated."""
    updated = 0
    for elem in tree.iter("body"):
        name = elem.get("name") or ""
        if name in aux_positions:
            elem.set("pos", _format_pos(aux_positions[name]))
            updated += 1
    return updated


def write_xml_with_aux_positions(
    *,
    xml_path: str | Path,
    aux_positions: dict[str, list[float]],
    dest_path: str | Path | None = None,
) -> str:
    """Apply aux positions and write XML, preserving comments/formatting when possible."""
    import re

    src = resolve_under_project(xml_path)
    text = src.read_text(encoding="utf-8")
    positions = {
        k: [float(x) for x in v[:3]] for k, v in (aux_positions or {}).items()
    }
    updated = 0
    for name, xyz in positions.items():
        pos_str = _format_pos(xyz)
        # Match opening <body ... name="NAME" ...> (single-line tags as in D20.xml)
        pattern = re.compile(
            rf'(<body\b(?=[^>]*\bname=["\']{re.escape(name)}["\'])[^>]*>)',
            re.IGNORECASE,
        )

        def _repl(m: re.Match) -> str:
            tag = m.group(1)
            if re.search(r'\bpos\s*=', tag, re.IGNORECASE):
                new_tag = re.sub(
                    r'\bpos\s*=\s*"[^"]*"',
                    f'pos="{pos_str}"',
                    tag,
                    count=1,
                    flags=re.IGNORECASE,
                )
                if new_tag == tag:
                    new_tag = re.sub(
                        r"\bpos\s*=\s*'[^']*'",
                        f"pos='{pos_str}'",
                        tag,
                        count=1,
                        flags=re.IGNORECASE,
                    )
            else:
                new_tag = tag[:-1] + f' pos="{pos_str}">'
            return new_tag

        new_text, n = pattern.subn(_repl, text, count=1)
        if n == 0:
            raise ValueError(f"Body not found in XML: {name}")
        updated += n
        text = new_text

    if updated == 0 and positions:
        raise ValueError(
            "No aux bodies found for: " + ", ".join(sorted(positions))
        )
    out = resolve_under_project(dest_path) if dest_path else src
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    try:
        return out.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(out)


def _mat_to_quat_wxyz(mat: np.ndarray) -> list[float]:
    q = R.from_matrix(mat).as_quat(scalar_first=True)
    return [float(x) for x in q]


def _shift_robot_to_root(robot_nodes: list[dict], root_candidates: list[str]) -> None:
    by = {n["name"]: n for n in robot_nodes}
    origin = None
    for cand in root_candidates:
        if cand in by:
            origin = np.asarray(by[cand]["pos"], dtype=float)
            break
    if origin is None and robot_nodes:
        origin = np.asarray(robot_nodes[0]["pos"], dtype=float)
    if origin is None:
        return
    for n in robot_nodes:
        p = np.asarray(n["pos"], dtype=float) - origin
        n["pos"] = [float(p[0]), float(p[1]), float(p[2])]


def build_aux_scene(
    *,
    xml_path: str,
    bvh_path: str | None = None,
    config: dict | None = None,
    bvh_unit: int | None = None,
    aux_positions: dict[str, list[float]] | None = None,
) -> dict[str, Any]:
    """Load robot FK with optional aux overrides + Stage-2 human overlay."""
    src_xml = resolve_under_project(xml_path)
    if not src_xml.exists():
        raise FileNotFoundError(f"Robot XML not found: {src_xml}")

    overrides = {
        k: [float(x) for x in v[:3]]
        for k, v in (aux_positions or {}).items()
    }

    aux_rows = list_aux_bodies(src_xml)
    for row in aux_rows:
        if row["name"] in overrides:
            row["pos"] = list(overrides[row["name"]])

    tmp_path = None
    load_path = src_xml
    if overrides:
        tree = ET.parse(src_xml)
        apply_aux_positions_to_tree(tree, overrides)
        tmp = tempfile.NamedTemporaryFile(
            mode="wb", suffix=".xml", delete=False, dir=str(src_xml.parent),
        )
        tmp_path = Path(tmp.name)
        tmp.close()
        tree.write(tmp_path, encoding="utf-8", xml_declaration=True)
        load_path = tmp_path

    try:
        model, data = load_robot_model(load_path)
        bodies = get_robot_body_info(model, data)
        edges = get_robot_body_edges(bodies)
        robot_nodes = []
        for b in bodies:
            if b["name"] == "world":
                continue
            robot_nodes.append({
                "name": b["name"],
                "pos": [float(x) for x in b["pos"]],
                "quat_wxyz": _mat_to_quat_wxyz(b["xmat"]),
                "parent": b.get("parent_name"),
                "is_aux": str(b["name"]).startswith(AUX_PREFIX),
            })
        robot_edges = [[a, b] for a, b in edges if a and b]
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

    human = None
    robot_root = (config or {}).get("robot_root_name") or "aux_base"
    if bvh_path and config is not None:
        from tools.retargeting_app.stage2_logic import build_stage2_scene
        xml_arg = (
            src_xml.relative_to(PROJECT_ROOT).as_posix()
            if src_xml.is_relative_to(PROJECT_ROOT) else str(src_xml)
        )
        scene = build_stage2_scene(
            xml_path=xml_arg,
            bvh_path=bvh_path,
            config=config,
            bvh_unit=bvh_unit,
        )
        human = scene["human"]
        robot_root = scene.get("robot_root_name") or robot_root

    _shift_robot_to_root(
        robot_nodes,
        [robot_root, "aux_base", "base_link", "pelvis"],
    )

    xml_rel = (
        src_xml.relative_to(PROJECT_ROOT).as_posix()
        if src_xml.is_relative_to(PROJECT_ROOT) else str(src_xml)
    )
    return {
        "xml_path": xml_rel,
        "aux_rows": aux_rows,
        "robot_root_name": robot_root,
        "robot": {"nodes": robot_nodes, "edges": robot_edges},
        "human": human,
    }
