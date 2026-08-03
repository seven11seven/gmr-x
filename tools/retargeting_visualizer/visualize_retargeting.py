#!/usr/bin/env python3
"""
Stage 2: Retargeting Visualization Tool
=========================================

Two-panel visualization for inspecting motion retargeting quality frame-by-frame.

  **Panel A (Links)** – Interactive plotly 3D HTML page:
      Robot skeleton + human skeleton overlaid in the **same coordinate system**
      with a frame slider, play/pause controls, and keyboard shortcuts.

  **Panel B (MuJoCo)** – Native MuJoCo viewer:
      Full mesh rendering with ground plane and the robot's real geometry,
      plus human skeleton overlay as custom line/sphere geometry.
      Keyboard frame navigation.

Quickstart::

    python tools/retargeting_visualizer/visualize_retargeting.py \
        --xml assets/D20_29dof/mjcf/D20.xml \
        --bvh datas/resource/fallAndGetUp2/Xsens2/5.bvh \
        --ik_config general_motion_retargeting/ik_configs/xsens_bvh_to_d20_29dof_getup.json

Defaults are the D20 + 5.bvh + getup config if arguments are omitted.
"""

import argparse
import json
import os
import pathlib
import sys
import time
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

from general_motion_retargeting.utils.xsens_vendor.BVHParser import (
    BVHParser,
    euler_to_quat,
    quat_fk,
    remove_quat_discontinuities,
)

# ---------------------------------------------------------------------------
# Default paths (relative to project root)
# ---------------------------------------------------------------------------
DEFAULT_XML = "assets/D20_29dof/mjcf/D20.xml"
DEFAULT_BVH = "datas/resource/fallAndGetUp2/Xsens2/5.bvh"
DEFAULT_IK_CONFIG = "general_motion_retargeting/ik_configs/xsens_bvh_to_d20_29dof_getup.json"

# ---------------------------------------------------------------------------
# Visual style (consistent with links_mapping tool)
# ---------------------------------------------------------------------------
C_ROBOT_MAPPED = "#2ca02c"       # green – robot body with IK mapping
C_ROBOT_UNMAPPED = "#7faac9"     # steel blue – robot body without mapping
C_HUMAN_MAPPED = "#ff7f0e"       # orange – human joint with IK mapping
C_HUMAN_UNMAPPED = "#ff9896"     # salmon – human joint without mapping
C_ROBOT_BONE = "#5b9bd5"         # light blue – robot bone lines
C_HUMAN_BONE = "#888888"         # grey – human bone lines
C_GROUND = "rgba(180,180,180,0.25)"

SPHERE_SIZE_ROBOT = 6
SPHERE_SIZE_HUMAN = 7
LINE_WIDTH = 2
AXIS_LENGTH = 0.05


# ============================================================================
#  Data loading
# ============================================================================

def load_robot_model(xml_path: pathlib.Path):
    """Load a MuJoCo model and return (model, data) with keyframe applied."""
    model = mj.MjModel.from_xml_path(str(xml_path))
    data = mj.MjData(model)
    if model.nkey > 0:
        data.qpos = model.key_qpos[0].copy()
    mj.mj_forward(model, data)
    return model, data


def get_robot_body_info(model, data):
    """Return list of dicts with body id, name, pos, xmat, parent_id."""
    bodies = []
    for i in range(model.nbody):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i)
        if name is None or name == "world":
            continue
        pid = model.body_parentid[i]
        pname = None
        if pid >= 0:
            pname = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, pid)
            if pname == "world":
                pname = None
        bodies.append({
            "id": i,
            "name": name,
            "pos": data.xpos[i].copy(),
            "xmat": data.xmat[i].copy().reshape(3, 3),
            "parent_id": pid,
            "parent_name": pname,
        })
    return bodies


def get_robot_edges(bodies):
    """Return (parent_name, child_name) edge list for robot skeleton."""
    return [(b["parent_name"], b["name"])
            for b in bodies if b["parent_name"] is not None]


def load_bvh_hierarchy(bvh_path: pathlib.Path):
    """Parse BVH skeleton hierarchy only (no motion data)."""
    p = BVHParser("zxy", 1.0)
    with open(bvh_path, "r") as fh:
        content = fh.read()
    # Parse just enough to get hierarchy – HIERARCHY_paser populates names/parents
    p.HIERARCHY_paser(content)
    return p


def load_bvh_all_frames(
    bvh_path: pathlib.Path,
    start: Optional[int] = None,
    end: Optional[int] = None,
    reset_to_zero: bool = False,
    scale: float = 1.0,
):
    """Load all BVH frames with global positions/quaternions.

    Returns:
        frames: list of dict {joint_name: [pos(3,), quat(4,)]}
        actual_human_height: float
        frame_time: float
        parser: BVHParser (has .names, .parents attributes)
    """
    parser = BVHParser("zxy", scale)

    with open(bvh_path, "r") as f:
        rotations, positions = parser.parse(
            f.read(),
            start=start,
            end=end,
            reset_to_zero=reset_to_zero,
        )

    quats = np.array([[euler_to_quat(rot) for rot in frame]
                      for frame in rotations])
    quats = remove_quat_discontinuities(quats)
    global_quats, global_positions = quat_fk(
        quats, positions, np.array(parser.parents, dtype=int),
    )

    frames = []
    for frame_idx in range(global_positions.shape[0]):
        result = {}
        for joint_idx, name in enumerate(parser.names):
            if name.endswith("_end_site"):
                continue
            result[name] = [
                global_positions[frame_idx, joint_idx].copy(),
                global_quats[frame_idx, joint_idx].copy(),
            ]
        # Synthetic FootMod joints (ankle pos + toe orientation)
        for side in ("Left", "Right"):
            ankle, toe, mod = f"{side}Ankle", f"{side}Toe", f"{side}FootMod"
            if ankle in result and toe in result:
                result[mod] = [result[ankle][0].copy(),
                               result[toe][1].copy()]
        frames.append(result)

    # Compute max height across ALL frames (first frame may show person on ground
    # for "fall and get up" motions, which would badly underestimate height).
    max_height = 0.0
    for frame in frames:
        foot_z = min(
            frame.get("LeftFootMod", frame.get("LeftAnkle", [0, 0, 0]))[0][2],
            frame.get("RightFootMod", frame.get("RightAnkle", [0, 0, 0]))[0][2],
        )
        head_z = frame.get("Head", [[0, 0, 0]])[0][2]
        h = head_z - foot_z
        if h > max_height:
            max_height = h
    # Fallback: if all frames are prone/crouching, use config assumption
    if max_height < 0.5:
        max_height = 1.8
    actual_human_height = max_height

    return frames, actual_human_height, parser.frame_time, parser


def get_human_edges(parser):
    """Build human skeleton edge list from BVH parser hierarchy."""
    name_set = set(parser.names)
    edges = []
    for i, name in enumerate(parser.names):
        if name.endswith("_end_site"):
            continue
        pi = parser.parents[i]
        if pi < 0:
            continue
        pname = parser.names[pi]
        if (not pname.endswith("_end_site")
                and pname in name_set and name in name_set):
            edges.append((pname, name))
    # Add FootMod edges
    for side in ("Left", "Right"):
        mod, ankle = f"{side}FootMod", f"{side}Ankle"
        if mod in name_set or ankle in name_set:
            if ankle in name_set:
                edges.append((ankle, mod))
    return edges


# ============================================================================
#  Frame processing
# ============================================================================

def process_all_frames(retargeter, human_frames, model):
    """Run retargeting on every frame, collect robot FK + human target positions.

    Args:
        retargeter: GeneralMotionRetargeting instance (already initialized).
        human_frames: list of dict {joint_name: [pos, quat]}.
        model: MuJoCo model (for FK).

    Returns:
        list of per-frame dicts:
            {'qpos': ndarray, 'robot_positions': {body_name: pos},
             'robot_xmat': {body_name: 3x3 mat},
             'human_positions': {joint_name: pos},
             'human_data': {joint_name: [pos, quat]}}
    """
    all_data = []
    data_fk = mj.MjData(model)

    print(f"\n  Processing {len(human_frames)} frames ...")
    for fi, human_data in enumerate(human_frames):
        # Retarget this frame
        qpos = retargeter.retarget(human_data, offset_to_ground=True)

        # Robot forward kinematics
        data_fk.qpos = qpos
        mj.mj_forward(model, data_fk)
        robot_positions = {}
        robot_xmat = {}
        for i in range(model.nbody):
            name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i)
            if name and name != "world":
                robot_positions[name] = data_fk.xpos[i].copy()
                robot_xmat[name] = data_fk.xmat[i].copy().reshape(3, 3)

        # Human target positions (scaled + offset, from retargeter state)
        human_scaled = retargeter.scaled_human_data
        human_positions = {name: val[0].copy()
                           for name, val in human_scaled.items()}

        all_data.append({
            "qpos": qpos.copy(),
            "robot_positions": robot_positions,
            "robot_xmat": robot_xmat,
            "human_positions": human_positions,
            "human_data": {name: [val[0].copy(), val[1].copy()]
                           for name, val in human_scaled.items()},
        })

        if (fi + 1) % 100 == 0:
            print(f"    ... {fi + 1}/{len(human_frames)} frames")

    print(f"  Done. {len(all_data)} frames processed.")
    return all_data


# ============================================================================
#  Plotly HTML – Panel A: Links comparison with frame slider
# ============================================================================

def _fmt_pos(p):
    """Format a 3D position for hover display."""
    return f"({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f})"


def _build_edge_segments(positions, edges):
    """Return (xs, ys, zs) with None separators for plotly line traces."""
    xs, ys, zs = [], [], []
    for pa, ch in edges:
        if pa not in positions or ch not in positions:
            continue
        p1, p2 = positions[pa], positions[ch]
        xs.extend([p1[0], p2[0], None])
        ys.extend([p1[1], p2[1], None])
        zs.extend([p1[2], p2[2], None])
    return xs, ys, zs


def _build_axis_segments_from_mats(positions, rot_mats, names, length=AXIS_LENGTH):
    """Build (X,Y,Z) axis line segments.  Each axis is returned as a dict
    ``{'x': [...], 'y': [...], 'z': [...]}`` with ``None`` separators so
    they can be used directly as a single ``scatter3d`` line trace."""
    axes = {"x": {"x": [], "y": [], "z": []},
            "y": {"x": [], "y": [], "z": []},
            "z": {"x": [], "y": [], "z": []}}
    for name in names:
        if name not in positions or name not in rot_mats:
            continue
        p = positions[name]
        mat = rot_mats[name]
        for ai, ak in enumerate(["x", "y", "z"]):
            d = mat[:, ai]
            ep = p + d * length
            seg = axes[ak]
            seg["x"].extend([float(p[0]), float(ep[0]), None])
            seg["y"].extend([float(p[1]), float(ep[1]), None])
            seg["z"].extend([float(p[2]), float(ep[2]), None])
    return axes


def _build_axis_segments_from_quats(positions, quats, names, length=AXIS_LENGTH):
    """Same as above but input orientations are scalar-first quaternions."""
    mats = {}
    for name in names:
        if name in positions and name in quats:
            mats[name] = R.from_quat(quats[name][1], scalar_first=True).as_matrix()
    return _build_axis_segments_from_mats(positions, mats, names, length)


def build_plotly_html(
    all_frame_data, robot_edges, human_edges,
    robot_mapped, human_mapped,
    title="Retargeting Viewer",
    motion_fps=30.0,
    sync_port=0,
):
    """Build an interactive plotly HTML page with a frame slider.

    **Strategy**: Pre-compute the COMPLETE list of Plotly trace dicts for
    *every* frame in Python and embed them as JSON.  JavaScript only calls
    ``Plotly.react()`` with the pre-built traces.
    """
    N = len(all_frame_data)
    if N == 0:
        raise ValueError("No frames to visualize.")

    frame0 = all_frame_data[0]
    robot_names = sorted(frame0["robot_positions"].keys())
    human_names = sorted(frame0["human_positions"].keys())

    # Static per-body colours / labels
    robot_colors = [
        C_ROBOT_MAPPED if n in robot_mapped else C_ROBOT_UNMAPPED
        for n in robot_names
    ]
    human_colors = [
        C_HUMAN_MAPPED if n in human_mapped else C_HUMAN_UNMAPPED
        for n in human_names
    ]

    # ---- pre-compute COMPLETE trace list for every frame -------------------
    import json as _json
    all_frames_traces = []

    for fi, fd in enumerate(all_frame_data):
        rp = fd["robot_positions"]
        hp = fd["human_positions"]
        rxmat = fd.get("robot_xmat", {})
        hdata = fd.get("human_data", {})

        # --- robot spheres ---
        rx, ry, rz = [], [], []
        rhover = []
        for name in robot_names:
            p = rp.get(name, np.zeros(3))
            rx.append(float(p[0])); ry.append(float(p[1])); rz.append(float(p[2]))
            tag = " [mapped]" if name in robot_mapped else ""
            rhover.append(f"<b>{name}</b>{tag}<br>robot body<br>{_fmt_pos(p)}")

        # --- robot bones ---
        re_x, re_y, re_z = _build_edge_segments(rp, robot_edges)

        # --- robot orientation axes (3 traces: X=red, Y=green, Z=blue) ---
        r_axes = _build_axis_segments_from_mats(rp, rxmat, robot_names)

        # --- human spheres ---
        hx, hy, hz = [], [], []
        hhover = []
        for name in human_names:
            p = hp.get(name, np.zeros(3))
            hx.append(float(p[0])); hy.append(float(p[1])); hz.append(float(p[2]))
            tag = " [mapped]" if name in human_mapped else ""
            hhover.append(f"<b>{name}</b>{tag}<br>human joint<br>{_fmt_pos(p)}")

        # --- human bones ---
        he_x, he_y, he_z = _build_edge_segments(hp, human_edges)

        # --- human orientation axes (3 traces) ---
        h_axes = _build_axis_segments_from_quats(hp, hdata, human_names)

        traces = [
            # 0 – robot spheres
            {"type": "scatter3d", "uid": "r-spheres", "mode": "markers",
             "x": rx, "y": ry, "z": rz,
             "marker": {"size": SPHERE_SIZE_ROBOT, "color": robot_colors,
                        "line": {"width": 1, "color": "black"}},
             "text": robot_names, "hovertext": rhover, "hoverinfo": "text",
             "name": "Robot links", "legendgroup": "robot", "showlegend": True},
            # 1 – robot bones
            {"type": "scatter3d", "uid": "r-bones", "mode": "lines",
             "x": re_x, "y": re_y, "z": re_z,
             "line": {"color": C_ROBOT_BONE, "width": LINE_WIDTH},
             "name": "Robot bones", "legendgroup": "robot",
             "showlegend": True, "hoverinfo": "skip"},
            # 2 – robot X axis (red)
            {"type": "scatter3d", "uid": "r-axis-x", "mode": "lines",
             "x": r_axes["x"]["x"], "y": r_axes["x"]["y"], "z": r_axes["x"]["z"],
             "line": {"color": "#e74c3c", "width": 2},
             "name": "Robot X axis", "legendgroup": "r-axes",
             "showlegend": True, "hoverinfo": "skip"},
            # 3 – robot Y axis (green)
            {"type": "scatter3d", "uid": "r-axis-y", "mode": "lines",
             "x": r_axes["y"]["x"], "y": r_axes["y"]["y"], "z": r_axes["y"]["z"],
             "line": {"color": "#27ae60", "width": 2},
             "name": "Robot Y axis", "legendgroup": "r-axes",
             "showlegend": True, "hoverinfo": "skip"},
            # 4 – robot Z axis (blue)
            {"type": "scatter3d", "uid": "r-axis-z", "mode": "lines",
             "x": r_axes["z"]["x"], "y": r_axes["z"]["y"], "z": r_axes["z"]["z"],
             "line": {"color": "#3498db", "width": 2},
             "name": "Robot Z axis", "legendgroup": "r-axes",
             "showlegend": True, "hoverinfo": "skip"},
            # 5 – human spheres
            {"type": "scatter3d", "uid": "h-spheres", "mode": "markers",
             "x": hx, "y": hy, "z": hz,
             "marker": {"size": SPHERE_SIZE_HUMAN, "color": human_colors,
                        "line": {"width": 1, "color": "black"}},
             "text": human_names, "hovertext": hhover, "hoverinfo": "text",
             "name": "Human joints", "legendgroup": "human", "showlegend": True},
            # 6 – human bones
            {"type": "scatter3d", "uid": "h-bones", "mode": "lines",
             "x": he_x, "y": he_y, "z": he_z,
             "line": {"color": C_HUMAN_BONE, "width": LINE_WIDTH},
             "name": "Human bones", "legendgroup": "human",
             "showlegend": True, "hoverinfo": "skip"},
            # 7 – human X axis (red)
            {"type": "scatter3d", "uid": "h-axis-x", "mode": "lines",
             "x": h_axes["x"]["x"], "y": h_axes["x"]["y"], "z": h_axes["x"]["z"],
             "line": {"color": "#e74c3c", "width": 2},
             "name": "Human X axis", "legendgroup": "h-axes",
             "showlegend": True, "hoverinfo": "skip"},
            # 8 – human Y axis (green)
            {"type": "scatter3d", "uid": "h-axis-y", "mode": "lines",
             "x": h_axes["y"]["x"], "y": h_axes["y"]["y"], "z": h_axes["y"]["z"],
             "line": {"color": "#27ae60", "width": 2},
             "name": "Human Y axis", "legendgroup": "h-axes",
             "showlegend": True, "hoverinfo": "skip"},
            # 9 – human Z axis (blue)
            {"type": "scatter3d", "uid": "h-axis-z", "mode": "lines",
             "x": h_axes["z"]["x"], "y": h_axes["z"]["y"], "z": h_axes["z"]["z"],
             "line": {"color": "#3498db", "width": 2},
             "name": "Human Z axis", "legendgroup": "h-axes",
             "showlegend": True, "hoverinfo": "skip"},
        ]
        all_frames_traces.append(traces)

    # Compute scene bounds from ALL frames so the camera encompasses the
    # entire motion (not just frame 0 – which was causing the skeleton to
    # be clipped / invisible on later frames).
    all_x, all_y, all_z = [], [], []
    for fd in all_frame_data:
        for p in fd["robot_positions"].values():
            all_x.append(float(p[0])); all_y.append(float(p[1])); all_z.append(float(p[2]))
        for p in fd["human_positions"].values():
            all_x.append(float(p[0])); all_y.append(float(p[1])); all_z.append(float(p[2]))
    all_pts = np.array([all_x, all_y, all_z]).T
    center = (all_pts.max(axis=0) + all_pts.min(axis=0)) / 2
    extent = float((all_pts.max(axis=0) - all_pts.min(axis=0)).max()) * 0.55 + 0.5

    # Ground plane – sized to cover all motion
    g_cx, g_cy = center[0], center[1]
    g_ext = extent * 1.2
    gx = np.linspace(g_cx - g_ext, g_cx + g_ext, 2).tolist()
    gy = np.linspace(g_cy - g_ext, g_cy + g_ext, 2).tolist()
    gz = [[0, 0], [0, 0]]

    ground_trace = {
        "type": "surface", "uid": "ground",
        "x": gx, "y": gy, "z": gz,
        "colorscale": [[0, "rgba(180,180,180,0.2)"], [1, "rgba(180,180,180,0.2)"]],
        "showscale": False,
        "name": "ground (z=0)",
        "hoverinfo": "skip",
        "legendgroup": "ref",
        "showlegend": True,
    }

    # Append ground to every frame's traces
    for traces in all_frames_traces:
        traces.append(ground_trace)

    # Layout – use broad fixed ranges computed from ALL frames.
    # (Plotly.react does NOT auto-update scene ranges when data changes,
    # so we must set ranges wide enough to show the entire motion.)
    layout = {
        "title": {"text": title, "font": {"size": 14}},
        "scene": {
            "xaxis": {"title": "X", "range": [center[0] - extent, center[0] + extent]},
            "yaxis": {"title": "Y", "range": [center[1] - extent, center[1] + extent]},
            "zaxis": {"title": "Z", "range": [center[2] - extent, center[2] + extent]},
            "aspectmode": "cube",
            "camera": {"eye": {"x": 1.0, "y": 1.0, "z": 0.7}},
        },
        "legend": {
            "yanchor": "top", "y": 0.99,
            "xanchor": "left", "x": 0.01,
            "bgcolor": "rgba(255,255,255,0.8)",
        },
        "margin": {"l": 0, "r": 0, "t": 50, "b": 50},
    }

    # ---- Serialize everything for JS --------------------------------------
    frame_data_json = _json.dumps({
        "totalFrames": N,
        "fps": motion_fps,
        "syncPort": sync_port if sync_port else 0,
        "frames": all_frames_traces,
        "layout": layout,
    })

    # Build the HTML manually (NOT via fig.to_html) so we control everything.
    html_parts = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        f"<title>{title}</title>",
        '<script src="https://cdn.plot.ly/plotly-3.7.0.min.js"></script>',
        "<style>",
        "  body { margin:0; padding:0; overflow:hidden; background:#111; }",
        "  #gmr-graph { position:absolute; top:0; left:0; right:0; bottom:60px; }",
        "</style>",
        "</head>",
        "<body>",
        '<div id="gmr-graph"></div>',
        _build_slider_panel(N),
        _build_frame_slider_js(frame_data_json),
        "</body>",
        "</html>",
    ]
    return "\n".join(html_parts)


def _build_slider_panel(total_frames: int) -> str:
    """HTML + CSS for the bottom control bar."""
    return f"""
<div id="gmr-controls" style="
    position:fixed; bottom:0; left:0; right:0; z-index:9999;
    background:rgba(28,28,32,0.94); color:#ddd; padding:10px 24px;
    font-family:monospace; font-size:13px;
    display:flex; gap:16px; align-items:center; flex-wrap:wrap;
    box-shadow:0 -2px 16px rgba(0,0,0,0.5); user-select:none;
    border-top:1px solid rgba(255,255,255,0.08);
">
  <span style="opacity:0.7;white-space:nowrap;">Frame</span>

  <button id="gmr-play" title="Play/Pause (Space)" style="
      padding:4px 10px; border-radius:6px; border:1px solid #666;
      background:#3a3a44; color:#eee; cursor:pointer; font-size:14px;
  ">&#9654;</button>

  <button id="gmr-prev" title="Prev frame (&larr;)" style="
      padding:4px 8px; border-radius:6px; border:1px solid #666;
      background:#3a3a44; color:#eee; cursor:pointer; font-size:14px;
  ">&#9664;</button>

  <input type="range" id="gmr-slider" min="0" max="{total_frames - 1}"
      value="0" step="1"
      style="flex:1; min-width:200px; max-width:600px; accent-color:#ff7f0e;">

  <button id="gmr-next" title="Next frame (&rarr;)" style="
      padding:4px 8px; border-radius:6px; border:1px solid #666;
      background:#3a3a44; color:#eee; cursor:pointer; font-size:14px;
  ">&#9658;</button>

  <span id="gmr-counter" style="min-width:130px;text-align:center;opacity:0.9;">
    1 / {total_frames}</span>

  <span style="opacity:0.4;">|</span>

  <label style="opacity:0.7;white-space:nowrap;">Speed
    <select id="gmr-speed" style="
        background:#3a3a44; color:#eee; border:1px solid #666;
        border-radius:4px; padding:2px 6px; font-size:12px;">
      <option value="0.25">0.25x</option>
      <option value="0.5">0.5x</option>
      <option value="1" selected>1x</option>
      <option value="2">2x</option>
      <option value="4">4x</option>
    </select>
  </label>

  <span style="opacity:0.4;font-size:11px;margin-left:auto;">
    &larr;&rarr; frame | Space play | 1-4 speed | R reset cam</span>
</div>
"""


def _build_frame_slider_js(frame_data_json: str) -> str:
    """JavaScript – trivially simple: just call Plotly.react() with
    pre-built trace arrays for the requested frame."""
    return f"""
<script>
(function() {{
  var FD = {frame_data_json};
  var N = FD.totalFrames;
  var fps = FD.fps || 30;
  var current = 0;
  var playing = false;
  var speed = 1.0;
  var animId = null;
  var lastTime = 0;

  var $ = function(id) {{ return document.getElementById(id); }};

  // Notify MuJoCo viewer of frame change (best-effort, fire-and-forget)
  function _notifyMujoco(frame) {{
    if (FD.syncPort) {{
      fetch('/frame/' + frame).catch(function(){{}});
    }}
  }}

  function updateFrame(idx) {{
    if (idx < 0) idx = 0;
    if (idx >= N) idx = N - 1;
    current = idx;
    Plotly.react('gmr-graph', FD.frames[idx], FD.layout);
    $('gmr-slider').value = current;
    $('gmr-counter').textContent = (current + 1) + ' / ' + N;
    _notifyMujoco(current);
  }}

  function playLoop(ts) {{
    if (!playing) return;
    if (lastTime === 0) {{ lastTime = ts; }}
    var dt = (ts - lastTime) / 1000;
    if (dt > 0 && dt < 1.0) {{
      var advance = Math.round(dt * fps * speed);
      if (advance < 1) advance = 1;
      if (advance > 10) advance = 10;
      var nxt = current + advance;
      if (nxt >= N) nxt = 0;
      updateFrame(nxt);
      lastTime = ts;
    }}
    animId = requestAnimationFrame(playLoop);
  }}

  function togglePlay() {{
    playing = !playing;
    $('gmr-play').innerHTML = playing ? '&#9632;' : '&#9654;';
    if (playing) {{
      lastTime = 0;
      animId = requestAnimationFrame(playLoop);
    }} else {{
      if (animId) cancelAnimationFrame(animId);
    }}
  }}

  $('gmr-slider').addEventListener('input', function() {{
    updateFrame(+this.value);
  }});
  $('gmr-play').addEventListener('click', togglePlay);
  $('gmr-prev').addEventListener('click', function() {{ updateFrame(current - 1); }});
  $('gmr-next').addEventListener('click', function() {{ updateFrame(current + 1); }});
  $('gmr-speed').addEventListener('change', function() {{ speed = +this.value; }});

  window.addEventListener('keydown', function(e) {{
    var handled = true;
    switch (e.key) {{
      case 'ArrowLeft':  updateFrame(current - 1); break;
      case 'ArrowRight': updateFrame(current + 1); break;
      case 'ArrowUp':    updateFrame(current + 10); break;
      case 'ArrowDown':  updateFrame(current - 10); break;
      case ' ': case 'Spacebar':
        togglePlay(); e.preventDefault(); break;
      case '1': $('gmr-speed').value = '0.25'; speed = 0.25; break;
      case '2': $('gmr-speed').value = '0.5';  speed = 0.5;  break;
      case '3': $('gmr-speed').value = '1';    speed = 1.0;  break;
      case '4': $('gmr-speed').value = '2';    speed = 2.0;  break;
      case '5': $('gmr-speed').value = '4';    speed = 4.0;  break;
      case 'r': case 'R':
        if (!e.ctrlKey && !e.metaKey) {{
          Plotly.relayout('gmr-graph', {{
            'scene.camera.eye': {{x: 1.0, y: 1.0, z: 0.7}}
          }});
        }} else {{ handled = false; }}
        break;
      default: handled = false;
    }}
    if (handled) {{ e.preventDefault(); e.stopPropagation(); }}
  }});

  // Initial render
  Plotly.react('gmr-graph', FD.frames[0], FD.layout);
}})();
</script>
"""


# ============================================================================
#  MuJoCo viewer – Panel B: Mesh rendering + human skeleton overlay
# ============================================================================

def _draw_human_skeleton_mj(viewer, human_positions, human_edges, human_data=None):
    """Draw human skeleton as custom geometry in the MuJoCo viewer.

    - Joints → **light-green** spheres
    - Bones  → **orange** thin connectors between parent–child
    - Each joint also gets RGB orientation axes (X=red, Y=green, Z=blue)
    """
    C_JOINT = [0.4, 1.0, 0.4, 0.90]        # light green
    C_BONE  = [1.0, 0.55, 0.0, 0.50]        # orange, semi-transparent
    AXIS_LEN = 0.08                          # +1/3 from 0.06
    AXIS_RGB = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]

    # Pre-compute rotation matrices from quaternions (scalar-first [w,x,y,z])
    orient_mats = {}
    if human_data:
        for name, val in human_data.items():
            if name in human_positions and len(val) >= 2:
                quat = np.asarray(val[1], dtype=float)
                if quat.shape == (4,):
                    orient_mats[name] = R.from_quat(quat, scalar_first=True).as_matrix()

    # ---- joint spheres (light green, 2/3 of original size) ----
    for _name, pos in human_positions.items():
        g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
        mj.mjv_initGeom(
            g, type=mj.mjtGeom.mjGEOM_SPHERE,
            size=[0.015, 0, 0],
            pos=np.array(pos, dtype=float),
            mat=np.eye(3).flatten(),
            rgba=C_JOINT,
        )
        viewer.user_scn.ngeom += 1

    # ---- bone connectors (orange capsules, no arrowhead) ----
    for parent, child in human_edges:
        if parent not in human_positions or child not in human_positions:
            continue
        p1 = np.array(human_positions[parent], dtype=float)
        p2 = np.array(human_positions[child], dtype=float)
        mid = (p1 + p2) / 2
        direction = p2 - p1
        dist = float(np.linalg.norm(direction))
        if dist < 1e-6:
            continue
        half = dist / 2
        z_axis = direction / dist
        # pick an x-axis perpendicular to z
        ref = np.array([1.0, 0.0, 0.0]) if abs(z_axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        x_axis = np.cross(z_axis, ref)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        bone_mat = np.column_stack([x_axis, y_axis, z_axis])

        g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
        mj.mjv_initGeom(
            g, type=mj.mjtGeom.mjGEOM_CAPSULE,
            size=[0.005, half, 0.0],       # radius, half-length
            pos=mid,
            mat=bone_mat.flatten(),
            rgba=C_BONE,
        )
        viewer.user_scn.ngeom += 1

    # ---- orientation axes (R/G/B arrows at each joint) ----
    for name, pos in human_positions.items():
        if name not in orient_mats:
            continue
        mat = orient_mats[name]
        origin = np.array(pos, dtype=float)
        for ai in range(3):
            g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
            direction = mat[:, ai]
            mj.mjv_initGeom(
                g, type=mj.mjtGeom.mjGEOM_ARROW,
                size=[0.003, 0.0, 0.0],        # no arrowhead on axes
                pos=origin,
                mat=mat.flatten(),
                rgba=AXIS_RGB[ai],
            )
            mj.mjv_connector(
                g, type=mj.mjtGeom.mjGEOM_ARROW, width=0.003,
                from_=origin,
                to=origin + direction * AXIS_LEN,
            )
            viewer.user_scn.ngeom += 1


def launch_mujoco_viewer(
    xml_path: pathlib.Path,
    all_frame_data: list,
    human_edges: list,
    total_frames: int,
    motion_fps: float = 30.0,
):
    """Panel B: MuJoCo viewer with full mesh rendering & human skeleton overlay.

    Keyboard controls:
      ← →    : previous / next frame
      ↑ ↓    : jump ±10 frames
      Space  : play / pause
      1-5    : playback speed (0.25× … 4×)
      Home   : jump to first frame
      End    : jump to last frame
      Esc    : close viewer
    """
    model = mj.MjModel.from_xml_path(str(xml_path))
    data = mj.MjData(model)

    # State – start playing immediately so the user sees motion without
    # needing to press any key first.
    state = {
        "current": 0,
        "playing": True,
        "speed": 1.0,
        "last_switch": time.time(),
    }

    # Build ordered human name list to iterate consistently
    human_names = sorted(all_frame_data[0]["human_positions"].keys())

    def key_callback(keycode: int):
        """GLFW key callback."""
        c = state
        if keycode == 263:  # LEFT
            c["current"] = max(c["current"] - 1, 0)
        elif keycode == 262:  # RIGHT
            c["current"] = min(c["current"] + 1, total_frames - 1)
        elif keycode == 265:  # UP
            c["current"] = min(c["current"] + 10, total_frames - 1)
        elif keycode == 264:  # DOWN
            c["current"] = max(c["current"] - 10, 0)
        elif keycode == 268:  # HOME
            c["current"] = 0
        elif keycode == 269:  # END
            c["current"] = total_frames - 1
        elif keycode == 32:  # SPACE
            c["playing"] = not c["playing"]
            c["last_switch"] = time.time()
        elif keycode == 49:  # 1
            c["speed"] = 0.25
        elif keycode == 50:  # 2
            c["speed"] = 0.5
        elif keycode == 51:  # 3
            c["speed"] = 1.0
        elif keycode == 52:  # 4
            c["speed"] = 2.0
        elif keycode == 53:  # 5
            c["speed"] = 4.0
        else:
            return  # unrecognised key – do not print
        _print_viewer_status(state, total_frames)

    viewer = mjv.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=False,
        key_callback=key_callback,
    )

    print("\n  [MuJoCo Panel] Controls:")
    print("    ← →  prev/next frame    ↑ ↓  jump ±10")
    print("    Space  play/pause       1-5  speed (0.25× … 4×)")
    print("    Home/End  first/last    Esc  close\n")

    _print_viewer_status(state, total_frames)

    # Main render loop
    prev_frame = -1
    last_time = time.time()
    loop_count = 0
    _axes_diag_done = False  # one-shot diagnostic

    while viewer.is_running():
        now = time.time()
        loop_count += 1

        # ---- sync from web panel (if HTTP server is active) ----
        if _sync_state["updated"]:
            state["current"] = _sync_state["web_frame"]
            _sync_state["updated"] = False
            state["playing"] = False  # pause when user scrubs in web

        if state["playing"]:
            dt = now - last_time
            advance = int(dt * motion_fps * state["speed"])
            if advance > 0:
                advance = min(advance, max(1, int(motion_fps * 0.5)))
                state["current"] = (state["current"] + advance) % total_frames
                last_time = now

        # Render only when frame changes
        if state["current"] != prev_frame:
            prev_frame = state["current"]
            fd = all_frame_data[state["current"]]

            # Set robot state (slice assignment – updates C-level data)
            data.qpos[:] = fd["qpos"]
            mj.mj_forward(model, data)

            # Draw human skeleton overlay
            viewer.user_scn.ngeom = 0
            try:
                _draw_human_skeleton_mj(viewer, fd["human_positions"], human_edges,
                                        fd.get("human_data", {}))
                if not _axes_diag_done:
                    hd = fd.get("human_data", {})
                    n_quat = sum(1 for v in hd.values()
                                 if len(v) >= 2 and np.asarray(v[1]).shape == (4,))
                    print(f"  [mj] human_data joints: {len(hd)}, with quat: {n_quat}")
                    _axes_diag_done = True
            except Exception as exc:
                print(f"  [WARN] _draw_human_skeleton_mj error: {exc}")

            if loop_count <= 3:
                print(f"  [mj] rendered frame {state['current'] + 1}/{total_frames}")

        viewer.sync()

        if not state["playing"]:
            time.sleep(0.02)  # reduce CPU when paused

    viewer.close()
    print("  [MuJoCo Panel] Viewer closed.")


def _print_viewer_status(state, total_frames):
    """Print current viewer status to console."""
    play_icon = "▶" if not state["playing"] else "⏸"
    print(f"  {play_icon} Frame {state['current'] + 1}/{total_frames}  "
          f"speed={state['speed']:.2f}×")


# ============================================================================
#  HTTP sync server – bridges web Panel A ↔ MuJoCo Panel B
# ============================================================================

import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

# Shared mutable state between the HTTP handler thread and the MuJoCo loop.
# _sync_state["web_frame"] is the last frame the web slider sent.
# _sync_state["updated"] is set True when the web slider moves.
_sync_state = {"web_frame": 0, "updated": False}


class _SyncHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler: serves the plotly HTML and receives frame sync."""

    # Class-level storage, set before the server starts
    html_content: bytes = b""
    html_path: str = ""

    def do_GET(self):
        if self.path.startswith("/frame/"):
            try:
                frame = int(self.path.rsplit("/", 1)[-1])
                if 0 <= frame:
                    _sync_state["web_frame"] = frame
                    _sync_state["updated"] = True
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"ok")
            except (ValueError, IndexError):
                self.send_response(400)
                self.end_headers()
        elif self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(self.html_content)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # silence HTTP request logs


def _start_sync_server(html_str: str) -> int:
    """Launch the sync HTTP server in a daemon thread.  Returns the port."""
    _SyncHandler.html_content = html_str.encode("utf-8")
    server = HTTPServer(("127.0.0.1", 0), _SyncHandler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return port


# ============================================================================
#  Mapping summary
# ============================================================================

def _write_csv(path: str, all_frame_data: list):
    """Export robot motion to CSV.
    Columns (36): base_x, base_y, base_z, base_qx, base_qy, base_qz, base_qw,
    then 29 joint values in MuJoCo default order.
    """
    rows = []
    for fd in all_frame_data:
        qpos = fd["qpos"]
        base_pos = qpos[0:3]
        base_quat_xyzw = np.roll(qpos[3:7], -1)  # wxyz → xyzw
        rows.append(np.concatenate([base_pos, base_quat_xyzw, qpos[7:]]))
    csv_data = np.stack(rows, axis=0)
    np.savetxt(path, csv_data, delimiter=",", fmt="%.8f",
               header=(
                   "base_x,base_y,base_z,"
                   "base_qx,base_qy,base_qz,base_qw,"
                   "waist_yaw,waist_roll,waist_pitch,"
                   "L_shoulder_pitch,L_shoulder_roll,L_shoulder_yaw,"
                   "L_elbow_pitch,L_wrist_roll,L_wrist_pitch,L_wrist_yaw,"
                   "R_shoulder_pitch,R_shoulder_roll,R_shoulder_yaw,"
                   "R_elbow_pitch,R_wrist_roll,R_wrist_pitch,R_wrist_yaw,"
                   "L_hip_pitch,L_hip_roll,L_hip_yaw,"
                   "L_knee_pitch,L_ankle_pitch,L_ankle_roll,"
                   "R_hip_pitch,R_hip_roll,R_hip_yaw,"
                   "R_knee_pitch,R_ankle_pitch,R_ankle_roll"),
               comments="")
    print(f"  CSV saved: {path}  ({csv_data.shape[0]} rows x "
          f"{csv_data.shape[1]} cols)")


def print_mapping_summary(robot_mapped, human_mapped, human_to_robot):
    """Print a compact mapping table to the terminal."""
    print("\n" + "=" * 70)
    print("IK Mapping Summary")
    print("=" * 70)
    print(f"  Robot bodies with mapping:  {len(robot_mapped)}")
    print(f"  Human joints with mapping:  {len(human_mapped)}\n")

    r2h = {}
    for h_name, r_names in human_to_robot.items():
        for rn in r_names:
            r2h.setdefault(rn, []).append(h_name)

    print(f"  {'Robot Body':<32s} →  Human Joint(s)")
    print(f"  {'-'*32}     {'-'*32}")
    for rb in sorted(r2h):
        print(f"  {rb:<32s} →  {', '.join(r2h[rb])}")
    print("=" * 70)


# ============================================================================
#  Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Stage 2: Retargeting Visualization – compare robot vs "
                    "human skeletons frame-by-frame with dual-panel views.")

    ap.add_argument("--xml", type=str, default=DEFAULT_XML,
                    help="Robot MuJoCo XML path (relative to project root)")
    ap.add_argument("--bvh", type=str, default=DEFAULT_BVH,
                    help="BVH motion file path (relative to project root)")
    ap.add_argument("--ik_config", type=str, default=DEFAULT_IK_CONFIG,
                    help="IK config JSON path (relative to project root)")

    ap.add_argument("-o", "--output", type=str, default=None,
                    help="Save Panel-A HTML to this path (default: temp file)")

    ap.add_argument("--start", type=int, default=None,
                    help="First BVH frame to process (0-indexed)")
    ap.add_argument("--end", type=int, default=None,
                    help="Last BVH frame to process (exclusive)")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="Limit total frames processed (for quick preview)")

    ap.add_argument("--no-panel-a", action="store_true",
                    help="Skip the plotly HTML links view")
    ap.add_argument("--no-panel-b", action="store_true",
                    help="Skip the MuJoCo mesh viewer")
    ap.add_argument("--no-open", action="store_true",
                    help="Do not open HTML in browser")

    ap.add_argument("--human-height", type=float, default=None,
                    help="Actual human height in meters (auto-detected if omitted)")

    ap.add_argument("--save", type=str, default=None, metavar="PATH",
                    help="Save retargeting results to .pkl file (skip this step next time)")
    ap.add_argument("--save-csv", type=str, default=None, metavar="PATH",
                    help="Save robot motion to CSV (base_pos_xyz + base_quat_xyzw + dof_pos)")
    ap.add_argument("--load", type=str, default=None, metavar="PATH",
                    help="Load pre-computed retargeting results from .pkl file")

    args = ap.parse_args()

    # Resolve paths
    xml_path = (_PROJECT_ROOT / args.xml).resolve()
    bvh_path = (_PROJECT_ROOT / args.bvh).resolve()
    ik_path = (_PROJECT_ROOT / args.ik_config).resolve()

    for p, label in [(xml_path, "XML"), (bvh_path, "BVH"), (ik_path, "IK config")]:
        if not p.exists():
            print(f"ERROR: {label} not found: {p}")
            sys.exit(1)

    print(f"Robot XML : {xml_path}")
    print(f"BVH file  : {bvh_path}")
    print(f"IK config : {ik_path}")

    # ---- 1. Load BVH --------------------------------------------------------
    print("\n[1/5] Loading BVH motion data …")
    human_frames, actual_human_height, frame_time, parser = load_bvh_all_frames(
        bvh_path,
        start=args.start,
        end=args.end,
    )
    motion_fps = int(1.0 / frame_time) if frame_time > 0 else 30
    print(f"  {len(human_frames)} frames, "
          f"human_height={actual_human_height:.2f}m, "
          f"fps≈{motion_fps}")

    # Apply frame limit
    if args.max_frames and args.max_frames < len(human_frames):
        human_frames = human_frames[:args.max_frames]
        print(f"  Limited to {args.max_frames} frames (--max-frames)")

    human_edges = get_human_edges(parser)
    print(f"  {len(human_edges)} human skeleton edges")

    # ---- 2. Robot model -----------------------------------------------------
    print("\n[2/5] Loading robot model …")
    robot_model, robot_data = load_robot_model(xml_path)
    robot_bodies = get_robot_body_info(robot_model, robot_data)
    robot_edges = get_robot_edges(robot_bodies)
    print(f"  {len(robot_bodies)} bodies, {len(robot_edges)} edges")

    # ---- 3. IK config -------------------------------------------------------
    print("[3/5] Loading IK config …")
    with open(ik_path) as fh:
        ik_config = json.load(fh)

    # Build mapping sets for coloring
    robot_mapped: set = set()
    human_mapped: set = set()
    human_to_robot: dict = {}

    match_table = ik_config.get("ik_match_table1", {})
    for frame_name, entry in match_table.items():
        entries = entry if isinstance(entry[0], list) else [entry]
        for body_name, _pw, _rw, _po, _ro in entries:
            robot_mapped.add(frame_name)
            human_mapped.add(body_name)
            human_to_robot.setdefault(body_name, []).append(frame_name)

    print(f"  {len(robot_mapped)} mapped robot bodies, "
          f"{len(human_mapped)} mapped human joints")
    print_mapping_summary(robot_mapped, human_mapped, human_to_robot)

    # ---- 4. Retarget all frames (or load from cache) ------------------------
    title = f"Retargeting: {bvh_path.name} → {xml_path.parent.name}  [{ik_path.name}]"

    if args.load:
        print(f"\n[4/5] Loading cached retargeting results from {args.load} …")
        import pickle
        with open(args.load, "rb") as fh:
            cache = pickle.load(fh)
        all_frame_data = cache["all_frame_data"]
        motion_fps = cache.get("motion_fps", motion_fps)
        print(f"  Loaded {len(all_frame_data)} frames (skipping retargeting).")

        # CSV export from loaded cache
        if args.save_csv:
            _write_csv(args.save_csv, all_frame_data)
    else:
        print("\n[4/5] Retargeting all frames (this may take a while) …")

        # Dynamically register the user-specified XML so GMR can load it.
        from general_motion_retargeting.params import ROBOT_XML_DICT, ROBOT_BASE_DICT

        _custom_robot_key = "_custom_robot_xml"
        ROBOT_XML_DICT[_custom_robot_key] = str(xml_path)
        ROBOT_BASE_DICT[_custom_robot_key] = "base_link"

        from general_motion_retargeting import GeneralMotionRetargeting as GMR

        # Height / scaling ratio
        actual_human_height = float(actual_human_height)
        use_height = float(args.human_height) if args.human_height is not None else actual_human_height
        with open(ik_path) as fh:
            _ik_tmp = json.load(fh)
        assumed_h = float(_ik_tmp.get("human_height_assumption", 1.8))
        ratio = use_height / assumed_h if use_height else 1.0

        used_label = "auto" if args.human_height is None else f"{args.human_height:.2f}"
        print("\n  Human height: detected=%.2fm  assumed(ik_config)=%.2fm  used=%s"
              % (actual_human_height, assumed_h, used_label))
        print("  Scale ratio = %.4f  (1.0 = exact match with links_mapping proportions)"
              % ratio)
        if abs(ratio - 1.0) > 0.01:
            print("  ! proportions differ from links_mapping tool by %.1f%%"
                  % (abs(1.0 - ratio) * 100))
            print("    -> use --human-height %.1f to match links_mapping" % assumed_h)

        retargeter = GMR(
            src_human="bvh_xsens",
            tgt_robot=_custom_robot_key,
            actual_human_height=use_height,
            ik_config_path=str(ik_path),
            verbose=False,
        )

        all_frame_data = process_all_frames(retargeter, human_frames, robot_model)

        if args.save:
            print(f"  Saving results to {args.save} …")
            import pickle
            cache = {
                "all_frame_data": all_frame_data,
                "motion_fps": motion_fps,
                "human_edges": human_edges,
            }
            with open(args.save, "wb") as fh:
                pickle.dump(cache, fh)
            print(f"  Saved {len(all_frame_data)} frames to {args.save}")

        # CSV export
        if args.save_csv:
            _write_csv(args.save_csv, all_frame_data)

    # ---- 5a. Panel A – Plotly HTML ------------------------------------------
    sync_port = 0
    if not args.no_panel_a:
        print("\n[5a/5] Building Panel A – Links comparison HTML …")

        # Build HTML with placeholder port – we'll fill it in after the
        # sync server starts (so we only build the heavy traces ONCE).
        html = build_plotly_html(
            all_frame_data, robot_edges, human_edges,
            robot_mapped, human_mapped,
            title=title,
            motion_fps=float(motion_fps),
            sync_port=0,
        )

        # Start the sync HTTP server if Panel B is also active
        if not args.no_panel_b:
            sync_port = _start_sync_server(html)
            html = html.replace('"syncPort": 0,', f'"syncPort": {sync_port},')
            _SyncHandler.html_content = html.encode("utf-8")
            print(f"  Sync server on http://localhost:{sync_port}")

        os.chdir(_PROJECT_ROOT)
        if args.output:
            html_path = pathlib.Path(args.output).resolve()
        else:
            import tempfile
            html_path = pathlib.Path(tempfile.mktemp(suffix="_retargeting.html"))

        with open(str(html_path), "w") as fh:
            fh.write(html)
        print(f"  Panel A HTML saved: {html_path}")

        if not args.no_open:
            if sync_port:
                url = f"http://localhost:{sync_port}/"
                webbrowser.open(url)
                print(f"  Opened {url} (synced with MuJoCo viewer)")
            else:
                webbrowser.open(f"file://{html_path}")
                print("  Opened in browser.")

        print(f"\n  Legend:")
        print(f"    ● {C_ROBOT_MAPPED}        = mapped robot body")
        print(f"    ● {C_ROBOT_UNMAPPED}        = unmapped robot body")
        print(f"    ● {C_HUMAN_MAPPED}        = mapped human joint")
        print(f"    ● {C_HUMAN_UNMAPPED}        = unmapped human joint")
        print(f"    ─ blue/grey = robot/human bone lines")
        print(f"    ─ R/G/B = orientation axes")
        print(f"    🖱 hover any point → name + coordinates")
        print(f"    🎬 bottom slider → scrub frames, Space to play")
    else:
        print("\n[5a/5] Panel A skipped (--no-panel-a).")

    # ---- 5b. Panel B – MuJoCo viewer ----------------------------------------
    if not args.no_panel_b:
        print("\n[5b/5] Launching Panel B – MuJoCo viewer with mesh rendering …")
        launch_mujoco_viewer(
            xml_path=xml_path,
            all_frame_data=all_frame_data,
            human_edges=human_edges,
            total_frames=len(all_frame_data),
            motion_fps=float(motion_fps),
        )
    else:
        print("\n[5b/5] Panel B skipped (--no-panel-b).")

    print("\nDone.")


if __name__ == "__main__":
    main()
