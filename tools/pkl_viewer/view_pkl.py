#!/usr/bin/env python3
"""
View a cached retargeting PKL file – no re-computation needed.

Quickstart::

    python tools/pkl_viewer/view_pkl.py cache.pkl

    # With robot XML for MuJoCo Panel B and body-name-aware coloring:
    python tools/pkl_viewer/view_pkl.py cache.pkl \\
        --xml assets/D20_29dof/mjcf/D20.xml

    # With IK config for mapped/unmapped colour coding:
    python tools/pkl_viewer/view_pkl.py cache.pkl \\
        --xml assets/D20_29dof/mjcf/D20.xml \\
        --ik_config general_motion_retargeting/ik_configs/xsens_bvh_to_d20_29dof_getup.json

    # Limit frames:
    python tools/pkl_viewer/view_pkl.py cache.pkl --max-frames 100

    # Panels:
    python tools/pkl_viewer/view_pkl.py cache.pkl --no-panel-a  # MuJoCo only
    python tools/pkl_viewer/view_pkl.py cache.pkl --no-panel-b  # web only
"""

import argparse
import json
import os
import pathlib
import pickle
import sys
import tempfile
import webbrowser
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

_HERE = pathlib.Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import mujoco as mj
import mujoco.viewer as mjv

# Re-use the heavy-lifting functions from the retargeting visualizer.
from tools.retargeting_visualizer.visualize_retargeting import (
    build_plotly_html,
    launch_mujoco_viewer,
    _start_sync_server,
    _SyncHandler,
    _build_edge_segments,
    C_ROBOT_MAPPED, C_ROBOT_UNMAPPED,
    C_HUMAN_MAPPED, C_HUMAN_UNMAPPED,
    C_ROBOT_BONE, C_HUMAN_BONE,
)

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_pkl(pkl_path: pathlib.Path):
    """Return (all_frame_data, motion_fps, human_edges)."""
    with open(pkl_path, "rb") as fh:
        cache = pickle.load(fh)
    return (
        cache["all_frame_data"],
        cache.get("motion_fps", 30.0),
        cache.get("human_edges", []),
    )


# ---------------------------------------------------------------------------
# Robot edges (minimal – doesn't need IK config)
# ---------------------------------------------------------------------------

def get_robot_edges_from_xml(xml_path: pathlib.Path):
    """Extract parent→child edges from a MuJoCo XML."""
    model = mj.MjModel.from_xml_path(str(xml_path))
    edges = []
    for i in range(model.nbody):
        pid = model.body_parentid[i]
        if pid < 0:
            continue
        cname = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i)
        pname = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, pid)
        if cname and pname and pname != "world":
            edges.append((pname, cname))
    return edges


def get_body_names_from_xml(xml_path: pathlib.Path):
    """Return sorted list of non-world body names."""
    model = mj.MjModel.from_xml_path(str(xml_path))
    names = []
    for i in range(model.nbody):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i)
        if name and name != "world":
            names.append(name)
    return sorted(names)


# ---------------------------------------------------------------------------
# Mapping helpers
# ---------------------------------------------------------------------------

def build_mapping_sets(ik_config_path: Optional[pathlib.Path]):
    """Return (robot_mapped, human_mapped) sets from an IK config JSON."""
    if ik_config_path is None or not ik_config_path.exists():
        return set(), set()
    with open(ik_config_path) as fh:
        cfg = json.load(fh)
    match_table = cfg.get("ik_match_table1", {})
    robot_mapped: set = set()
    human_mapped: set = set()
    for frame_name, entry in match_table.items():
        entries = entry if isinstance(entry[0], list) else [entry]
        robot_mapped.add(frame_name)
        for body_name, *_ in entries:
            human_mapped.add(body_name)
    return robot_mapped, human_mapped


# ---------------------------------------------------------------------------
# Robots without XML
# ---------------------------------------------------------------------------

def infer_robot_edges_from_data(all_frame_data):
    """If we don't have an XML, guess edges from the frame-0 body positions.

    Heuristic: connect each body to its nearest neighbour that is closer
    to the root.  This is a fallback – pass ``--xml`` for accurate edges.
    """
    f0 = all_frame_data[0]
    pos = f0["robot_positions"]
    names = sorted(pos.keys())
    edges = []
    for name in names:
        p = pos[name]
        # find nearest body that is closer to origin → likely parent
        best, best_dist = None, np.linalg.norm(p)
        for other in names:
            if other == name:
                continue
            op = pos[other]
            if np.linalg.norm(op) >= best_dist:
                continue
            d = np.linalg.norm(p - op)
            if best is None or d < best_dist:
                best = other
        if best:
            edges.append((best, name))
    return edges


# ============================================================================
#  Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="View a cached retargeting PKL file – no re-computation.")

    ap.add_argument("pkl", type=str, help="Path to .pkl file")
    ap.add_argument("--xml", type=str, default=None,
                    help="Robot XML (for MuJoCo Panel B + correct bone edges)")
    ap.add_argument("--ik_config", type=str, default=None,
                    help="IK config JSON (for mapped/unmapped colour coding)")

    ap.add_argument("-o", "--output", type=str, default=None,
                    help="Save Panel-A HTML to this path")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="Limit frames shown")
    ap.add_argument("--no-panel-a", action="store_true",
                    help="Skip plotly HTML view")
    ap.add_argument("--no-panel-b", action="store_true",
                    help="Skip MuJoCo view")
    ap.add_argument("--no-open", action="store_true",
                    help="Do not open browser")
    ap.add_argument("--title", type=str, default=None,
                    help="Custom title for the HTML page")

    args = ap.parse_args()

    pkl_path = pathlib.Path(args.pkl).resolve()
    if not pkl_path.exists():
        print(f"ERROR: file not found: {pkl_path}")
        sys.exit(1)

    # Resolve optional paths
    xml_path = (_PROJECT_ROOT / args.xml).resolve() if args.xml else None
    ik_path = (_PROJECT_ROOT / args.ik_config).resolve() if args.ik_config else None

    if xml_path and not xml_path.exists():
        print(f"WARNING: XML not found ({xml_path}), proceeding without.")
        xml_path = None
    if ik_path and not ik_path.exists():
        print(f"WARNING: IK config not found ({ik_path}), proceeding without.")
        ik_path = None

    # ---- Load PKL -------------------------------------------------------
    print(f"[1/3] Loading {pkl_path.name} …")
    all_frame_data, motion_fps, human_edges = load_pkl(pkl_path)
    N = len(all_frame_data)
    if args.max_frames and args.max_frames < N:
        all_frame_data = all_frame_data[:args.max_frames]
        N = len(all_frame_data)
    print(f"  {N} frames, fps≈{motion_fps}, human_edges={len(human_edges)}")

    # ---- Robot edges -----------------------------------------------------
    if xml_path:
        robot_edges = get_robot_edges_from_xml(xml_path)
        print(f"[2/3] Robot edges from XML: {len(robot_edges)} edges")
    else:
        robot_edges = infer_robot_edges_from_data(all_frame_data)
        print(f"[2/3] Robot edges inferred (pass --xml for accurate): "
              f"{len(robot_edges)} edges")

    # ---- Mapping colours -------------------------------------------------
    robot_mapped, human_mapped = build_mapping_sets(ik_path)
    if ik_path:
        print(f"[3/3] IK config: {len(robot_mapped)} robot + "
              f"{len(human_mapped)} human mapped")
    else:
        print(f"[3/3] No IK config – all bodies shown as unmapped.")

    title = args.title or f"PKL: {pkl_path.name}  [{N} frames]"

    # ---- Panel A ----------------------------------------------------------
    sync_port = 0
    if not args.no_panel_a:
        print("\n--- Panel A (HTML) ---")
        html = build_plotly_html(
            all_frame_data, robot_edges, human_edges,
            robot_mapped, human_mapped,
            title=title, motion_fps=float(motion_fps),
            sync_port=0,
        )

        if not args.no_panel_b and xml_path:
            sync_port = _start_sync_server(html)
            html = html.replace('"syncPort": 0,', f'"syncPort": {sync_port},')
            _SyncHandler.html_content = html.encode("utf-8")
            print(f"  Sync server on http://localhost:{sync_port}")

        os.chdir(_PROJECT_ROOT)
        if args.output:
            html_path = pathlib.Path(args.output).resolve()
        else:
            html_path = pathlib.Path(tempfile.mktemp(suffix="_pkl_view.html"))
        with open(str(html_path), "w") as fh:
            fh.write(html)
        print(f"  Saved: {html_path}")

        if not args.no_open:
            if sync_port:
                webbrowser.open(f"http://localhost:{sync_port}/")
            else:
                webbrowser.open(f"file://{html_path}")

    # ---- Panel B ----------------------------------------------------------
    if not args.no_panel_b:
        if xml_path is None:
            print("\n--- Panel B SKIPPED (needs --xml) ---")
        else:
            print("\n--- Panel B (MuJoCo) ---")
            launch_mujoco_viewer(
                xml_path=xml_path,
                all_frame_data=all_frame_data,
                human_edges=human_edges,
                total_frames=N,
                motion_fps=float(motion_fps),
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
