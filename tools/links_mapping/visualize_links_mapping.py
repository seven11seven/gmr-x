#!/usr/bin/env python3
"""
Visualize links mapping between robot XML bodies and BVH human skeleton joints.

Generates an **interactive 3D HTML page** (plotly) that opens in your browser.
Each link shows:
  - a sphere at its position
  - RGB orientation axes (X=red, Y=green, Z=blue) on human joints
  - a text label
  - bone-lines connecting parent→child joints

Optionally loads an IK config JSON to apply scaling/offset transformations
and highlight mapped bodies.

Usage:
    # With IK config:
    python tools/links_mapping/visualize_links_mapping.py \\
        --xml assets/D20_29dof/mjcf/D20.xml \\
        --bvh datas/resource/fallAndGetUp2/Xsens2/5.bvh \\
        --ik_config general_motion_retargeting/ik_configs/xsens_bvh_to_d20_29dof_getup.json

    # Defaults (D20_29dof + 5.bvh, no IK config):
    python tools/links_mapping/visualize_links_mapping.py

    # Save HTML without opening browser:
    python tools/links_mapping/visualize_links_mapping.py -o mapping.html --no-open
"""

import argparse
import json
import sys
import pathlib
import os
import webbrowser
import numpy as np
from scipy.spatial.transform import Rotation as R

_HERE = pathlib.Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import plotly.graph_objects as go

DEFAULT_XML = "assets/D20_29dof/mjcf/D20.xml"
DEFAULT_BVH = "datas/resource/fallAndGetUp2/Xsens2/5.bvh"

# ---------------------------------------------------------------------------
# Visual style
# ---------------------------------------------------------------------------
C_ROBOT_MAPPED     = "#2ca02c"   # green
C_ROBOT_UNMAPPED   = "#7faac9"   # steel blue
C_HUMAN_MAPPED     = "#ff7f0e"   # orange
C_HUMAN_UNMAPPED   = "#ff9896"   # salmon
C_BONE             = "#888888"   # grey

SPHERE_SIZE  = 8
AXIS_LENGTH  = 0.06
AXIS_WIDTH   = 3
LINE_WIDTH   = 2


# ===================================================================
# Robot helpers
# ===================================================================

def load_robot_model(xml_path):
    import mujoco as mj
    model = mj.MjModel.from_xml_path(str(xml_path))
    data = mj.MjData(model)
    # Apply keyframe if defined (qpos0 stays at XML defaults)
    if model.nkey > 0:
        data.qpos = model.key_qpos[0].copy()
    mj.mj_forward(model, data)
    return model, data


def get_robot_body_info(model, data):
    import mujoco as mj
    bodies = []
    for i in range(model.nbody):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i)
        if name is None:
            name = f"body_{i}"
        pid = model.body_parentid[i]
        pname = None
        if pid >= 0:
            pname = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, pid)
        # xmat is row-major 9-vector → reshape to 3×3 where columns are body axes
        rot_mat = data.xmat[i].copy().reshape(3, 3)
        bodies.append({
            "id": i, "name": name, "pos": data.xpos[i].copy(),
            "xmat": rot_mat,
            "parent_id": pid, "parent_name": pname,
        })
    return bodies


def get_robot_body_positions(bodies):
    return {b["name"]: b["pos"] for b in bodies}


def get_robot_body_orientations(bodies):
    return {b["name"]: b["xmat"] for b in bodies}


def get_robot_body_edges(bodies):
    return [(b["parent_name"], b["name"])
            for b in bodies if b["parent_name"] is not None]


# ===================================================================
# BVH helpers
# ===================================================================

def load_bvh_first_frame(bvh_path, bvh_format="3DSM", scale=1.0):
    from general_motion_retargeting.utils.xsens_vendor.BVHParser import (
        BVHParser, euler_to_quat, quat_fk, remove_quat_discontinuities,
    )
    p = BVHParser("zxy", scale)
    with open(bvh_path, "r") as fh:
        content = fh.read()
    rotations, positions = p.parse(content, start=0, end=1,
                                   reset_to_zero=False)
    quats = np.array([[euler_to_quat(rot) for rot in frame]
                      for frame in rotations])
    quats = remove_quat_discontinuities(quats)
    global_quats, global_positions = quat_fk(
        quats, positions, np.array(p.parents, dtype=int))

    frame_data: dict = {}
    for ji, name in enumerate(p.names):
        if name.endswith("_end_site"):
            continue
        frame_data[name] = [
            global_positions[0, ji].copy(),
            global_quats[0, ji].copy(),
        ]
    for side in ("Left", "Right"):
        ankle, toe, mod = f"{side}Ankle", f"{side}Toe", f"{side}FootMod"
        if ankle in frame_data and toe in frame_data:
            frame_data[mod] = [frame_data[ankle][0].copy(),
                               frame_data[toe][1].copy()]

    edges = []
    name_set = set(frame_data.keys())
    for i, name in enumerate(p.names):
        if name.endswith("_end_site"):
            continue
        pi = p.parents[i]
        if pi < 0:
            continue
        pname = p.names[pi]
        if (not pname.endswith("_end_site")
                and pname in name_set and name in name_set):
            edges.append((pname, name))
    for side in ("Left", "Right"):
        mod, ankle = f"{side}FootMod", f"{side}Ankle"
        if mod in frame_data:
            edges.append((ankle, mod))
    return frame_data, edges


# ===================================================================
# IK config processing
# ===================================================================

def process_human_data(human_data, ik_config, ratio=1.0):
    sv_table = {k: v * ratio for k, v in
                ik_config.get("human_scale_table_vertical", {}).items()}
    sh_table = {k: v * ratio for k, v in
                ik_config.get("human_scale_table_horizontal", {}).items()}
    human_root = ik_config["human_root_name"]
    ground = ik_config.get("ground_height", 0.0) * np.array([0, 0, 1])

    root_pos, root_quat = human_data[human_root]
    sv_r, sh_r = sv_table.get(human_root, 1.0), sh_table.get(human_root, 1.0)
    scaled_root = root_pos * np.array([sh_r, sh_r, sv_r])

    local: dict = {}
    for name in human_data:
        if name == human_root or name not in sv_table:
            continue
        lp = human_data[name][0] - root_pos
        local[name] = lp * np.array([sh_table[name], sh_table[name],
                                      sv_table[name]])
    scaled: dict = {human_root: (scaled_root, root_quat)}
    for name, lp in local.items():
        scaled[name] = (lp + scaled_root, human_data[name][1])

    # Show scale factors for verification
    print(f"  [scale] ratio={ratio:.3f}  human_root={human_root} "
          f"h={sh_r:.2f} v={sv_r:.2f}")
    sample = ["LeftShoulder", "RightShoulder", "LeftHip", "LeftElbow", "Neck"]
    for key in sample:
        if key in sv_table:
            print(f"           {key}: h_scale={sh_table[key]:.2f} v_scale={sv_table[key]:.2f}")

    match_table = ik_config.get("ik_match_table1", {})
    pos_off, rot_off = {}, {}
    human_to_robot: dict = {}
    for frame_name, entry in match_table.items():
        entries = entry if isinstance(entry[0], list) else [entry]
        for body_name, _pw, _rw, po, ro in entries:
            po_np = np.array(po) - ground
            ro_R = R.from_quat(ro, scalar_first=True)
            pos_off[body_name] = po_np
            rot_off[body_name] = ro_R
            human_to_robot.setdefault(body_name, []).append(frame_name)

    processed: dict = {}
    for name, (pos, quat) in scaled.items():
        q = quat
        if name in rot_off:
            q = (R.from_quat(quat, scalar_first=True)
                 * rot_off[name]).as_quat(scalar_first=True)
        if name in pos_off:
            go = R.from_quat(q, scalar_first=True).apply(pos_off[name])
            processed[name] = [pos + go, q]
        else:
            processed[name] = [pos, q]
    return processed, human_to_robot


# ===================================================================
# plotly figure builder
# ===================================================================

def _build_edge_segments(positions, edges, offset=np.zeros(3)):
    """Return (xs, ys, zs) lists with None separators for disconnected lines."""
    xs, ys, zs = [], [], []
    for pa, ch in edges:
        if pa not in positions or ch not in positions:
            continue
        p1 = positions[pa] + offset
        p2 = positions[ch] + offset
        xs.extend([p1[0], p2[0], None])
        ys.extend([p1[1], p2[1], None])
        zs.extend([p1[2], p2[2], None])
    return xs, ys, zs


def _build_orientation_segments(positions, quats, offset=np.zeros(3)):
    """Build (R,G,B) axis line segments from {name: (pos, quat)} dict."""
    mats = {}
    for name, (pos, quat) in quats.items():
        mats[name] = R.from_quat(quat, scalar_first=True).as_matrix()
    return _build_orientation_segments_from_mats(positions, mats, offset)


def _build_orientation_segments_from_mats(positions, rot_mats, offset=np.zeros(3)):
    """Build (R,G,B) axis line segments from {name: 3×3 rotation matrix} dict."""
    colours = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    all_segs = {c: ([], [], []) for c in colours}

    for name, rot_mat in rot_mats.items():
        if name not in positions:
            continue
        p = positions[name] + offset
        for ci, (r, g, b) in enumerate(colours):
            direction = rot_mat[:, ci]
            end = p + direction * AXIS_LENGTH
            all_segs[(r, g, b)][0].extend([p[0], end[0], None])
            all_segs[(r, g, b)][1].extend([p[1], end[1], None])
            all_segs[(r, g, b)][2].extend([p[2], end[2], None])
    return all_segs


def build_figure(robot_bodies, robot_positions, robot_orientations,
                 robot_edges, human_data, human_edges,
                 robot_mapped, human_mapped, show_frames, title=""):
    """Build and return a plotly Figure + robot trace metadata for JS sliders."""

    # Compute offset so human sits beside robot
    robot_pts = np.array([b["pos"] for b in robot_bodies])
    human_pts = np.array([d[0] for d in human_data.values()])
    rw = float(robot_pts[:, 1].max() - robot_pts[:, 1].min())
    hw = float(human_pts[:, 1].max() - human_pts[:, 1].min())
    sep = max(rw, hw) * 0.7 + 0.5
    human_offset = np.array([0.0, sep, 0.0])

    human_positions = {n: d[0] for n, d in human_data.items()}

    fig = go.Figure()
    _fmt = lambda v: f"({v[0]:.4f}, {v[1]:.4f}, {v[2]:.4f})"

    robot_trace_start = 0  # first robot trace index

    # ---- Robot spheres ----
    for b in robot_bodies:
        name, p = b["name"], b["pos"]
        c = C_ROBOT_MAPPED if name in robot_mapped else C_ROBOT_UNMAPPED
        fig.add_trace(go.Scatter3d(
            x=[p[0]], y=[p[1]], z=[p[2]],
            mode="markers+text",
            marker=dict(size=SPHERE_SIZE, color=c,
                        line=dict(width=1, color="black")),
            text=name,
            textposition="top center",
            textfont=dict(size=9, color="black"),
            name="robot" if b["id"] == 0 else None,
            legendgroup="robot",
            showlegend=(b["id"] == 0),
            hoverinfo="text",
            hovertext=f"<b>{name}</b><br>robot body<br>{_fmt(p)}",
        ))

    # ---- Robot bone lines ----
    r_xs, r_ys, r_zs = _build_edge_segments(robot_positions, robot_edges)
    fig.add_trace(go.Scatter3d(
        x=r_xs, y=r_ys, z=r_zs,
        mode="lines",
        line=dict(color=C_BONE, width=LINE_WIDTH),
        name="robot bones",
        legendgroup="robot",
        showlegend=True,
        hoverinfo="skip",
    ))

    # ---- Robot orientation axes (RGB) ----
    if show_frames:
        r_segs = _build_orientation_segments_from_mats(
            robot_positions, robot_orientations)
        axis_names = [("X", "#e74c3c"), ("Y", "#27ae60"), ("Z", "#3498db")]
        for (r_c, g_c, b_c), (label, hex_c) in zip(
            [(1, 0, 0), (0, 1, 0), (0, 0, 1)], axis_names):
            xs, ys, zs = r_segs[(r_c, g_c, b_c)]
            fig.add_trace(go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode="lines",
                line=dict(color=hex_c, width=AXIS_WIDTH),
                name=f"robot {label} axis",
                legendgroup="robot_axes",
                showlegend=(r_c == 1 and g_c == 0),
                hoverinfo="skip",
            ))

    robot_trace_end = len(fig.data)  # end of robot traces

    # ---- Human spheres ----
    for name, (pos, quat) in human_data.items():
        p = pos + human_offset
        c = C_HUMAN_MAPPED if name in human_mapped else C_HUMAN_UNMAPPED
        fig.add_trace(go.Scatter3d(
            x=[p[0]], y=[p[1]], z=[p[2]],
            mode="markers+text",
            marker=dict(size=SPHERE_SIZE, color=c,
                        line=dict(width=1, color="black")),
            text=name,
            textposition="top center",
            textfont=dict(size=9, color="black"),
            name="human" if list(human_data.keys())[0] == name else None,
            legendgroup="human",
            showlegend=(list(human_data.keys())[0] == name),
            hoverinfo="text",
            hovertext=f"<b>{name}</b><br>human joint<br>{_fmt(pos)}",
        ))

    # ---- Human bone lines ----
    h_xs, h_ys, h_zs = _build_edge_segments(human_positions, human_edges,
                                             human_offset)
    fig.add_trace(go.Scatter3d(
        x=h_xs, y=h_ys, z=h_zs,
        mode="lines",
        line=dict(color=C_BONE, width=LINE_WIDTH),
        name="human bones",
        legendgroup="human",
        showlegend=True,
        hoverinfo="skip",
    ))

    # ---- Human orientation axes (RGB) ----
    if show_frames:
        segs = _build_orientation_segments(human_positions, human_data,
                                           human_offset)
        axis_names = [("X", "#e74c3c"), ("Y", "#27ae60"), ("Z", "#3498db")]
        for (r, g, b), (label, hex_c) in zip(
            [(1, 0, 0), (0, 1, 0), (0, 0, 1)], axis_names):
            xs, ys, zs = segs[(r, g, b)]
            fig.add_trace(go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode="lines",
                line=dict(color=hex_c, width=AXIS_WIDTH),
                name=f"human {label} axis",
                legendgroup="axes",
                showlegend=(r == 1 and g == 0),
                hoverinfo="skip",
            ))

    # ---- Layout ----
    all_pts = np.concatenate([robot_pts, human_pts + human_offset], axis=0)
    center = (all_pts.max(axis=0) + all_pts.min(axis=0)) / 2
    extent = float((all_pts.max(axis=0) - all_pts.min(axis=0)).max()) * 0.7

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        scene=dict(
            xaxis=dict(title="X",
                       range=[center[0] - extent, center[0] + extent]),
            yaxis=dict(title="Y",
                       range=[center[1] - extent, center[1] + extent]),
            zaxis=dict(title="Z",
                       range=[center[2] - extent, center[2] + extent]),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.2, y=1.2, z=0.8)),
        ),
        legend=dict(
            yanchor="top", y=0.99, xanchor="left", x=0.01,
            bgcolor="rgba(255,255,255,0.8)",
        ),
        margin=dict(l=0, r=0, t=50, b=0),
    )

    # Collect original robot trace data for JS slider interaction
    robot_original = []
    for i in range(robot_trace_start, robot_trace_end):
        t = fig.data[i]
        robot_original.append({
            "x": list(t.x) if hasattr(t.x, '__iter__') else [t.x],
            "y": list(t.y) if hasattr(t.y, '__iter__') else [t.y],
            "z": list(t.z) if hasattr(t.z, '__iter__') else [t.z],
        })

    return fig, robot_original


# ===================================================================
# Interactive robot-translation sliders (embedded JS)
# ===================================================================

ROBOT_SLIDER_JS = r"""
<div id="gmr-controls" style="
    position:fixed; bottom:20px; left:50%; transform:translateX(-50%);
    background:rgba(30,30,30,0.92); color:#eee; padding:12px 24px;
    border-radius:12px; z-index:9999; font-family:monospace; font-size:13px;
    display:flex; gap:24px; align-items:center; box-shadow:0 2px 16px rgba(0,0,0,0.5);
    user-select:none;
">
  <span style="opacity:0.7;">🤖 robot offset (m)</span>
  <label>X <span id="gmx-v" style="color:#e74c3c;">0.00</span>
    <input type="range" id="gmx" min="-4" max="4" step="0.01" value="0"
           style="width:100px;accent-color:#e74c3c;">
  </label>
  <label>Y <span id="gmy-v" style="color:#27ae60;">0.00</span>
    <input type="range" id="gmy" min="-4" max="4" step="0.01" value="0"
           style="width:100px;accent-color:#27ae60;">
  </label>
  <label>Z <span id="gmz-v" style="color:#3498db;">0.00</span>
    <input type="range" id="gmz" min="-4" max="4" step="0.01" value="0"
           style="width:100px;accent-color:#3498db;">
  </label>
  <button id="gmr-reset" style="
      padding:4px 12px; border-radius:6px; border:1px solid #888;
      background:#444; color:#eee; cursor:pointer;
  ">↺ reset</button>
  <span id="gmr-pos" style="opacity:0.6;min-width:180px;">(0.00, 0.00, 0.00)</span>
  <span style="opacity:0.4;font-size:11px;">↑↓←→ X/Y | W/S Z | Shift=fine | R reset</span>
</div>
<script>
(function() {
  var ORIG = __ROBOT_ORIG_DATA__;
  var N = ORIG.length;
  var curr = [0, 0, 0];
  var indices = Array.from({length:N}, function(_,i){return i;});
  var STEP = 0.005, FINE = 0.001;

  function isNum(v) { return typeof v === 'number' && isFinite(v); }

  function apply() {
    var newX = [], newY = [], newZ = [];
    for (var i = 0; i < N; i++) {
      var ox = ORIG[i].x, oy = ORIG[i].y, oz = ORIG[i].z;
      newX.push(ox.map(function(v){return isNum(v)?v+curr[0]:v;}));
      newY.push(oy.map(function(v){return isNum(v)?v+curr[1]:v;}));
      newZ.push(oz.map(function(v){return isNum(v)?v+curr[2]:v;}));
    }
    Plotly.restyle('gmr-graph', {x: newX, y: newY, z: newZ}, indices);
    document.getElementById('gmx').value = curr[0];
    document.getElementById('gmy').value = curr[1];
    document.getElementById('gmz').value = curr[2];
    document.getElementById('gmx-v').textContent = curr[0].toFixed(2);
    document.getElementById('gmy-v').textContent = curr[1].toFixed(2);
    document.getElementById('gmz-v').textContent = curr[2].toFixed(2);
    document.getElementById('gmr-pos').textContent =
      '(' + curr[0].toFixed(2) + ', ' + curr[1].toFixed(2) + ', ' + curr[2].toFixed(2) + ')';
  }

  function shift(dx, dy, dz) {
    curr[0] = Math.round((curr[0] + dx)*1000)/1000;
    curr[1] = Math.round((curr[1] + dy)*1000)/1000;
    curr[2] = Math.round((curr[2] + dz)*1000)/1000;
    apply();
  }

  // Sliders
  document.getElementById('gmx').addEventListener('input', function(){
    curr[0] = +this.value; apply();});
  document.getElementById('gmy').addEventListener('input', function(){
    curr[1] = +this.value; apply();});
  document.getElementById('gmz').addEventListener('input', function(){
    curr[2] = +this.value; apply();});

  // Reset
  document.getElementById('gmr-reset').addEventListener('click', function(){
    curr = [0, 0, 0]; apply();});

  // Keyboard: arrows X/Y, W/S Z, Shift=fine, R=reset
  window.addEventListener('keydown', function(e) {
    var s = e.shiftKey ? FINE : STEP;
    var handled = true;
    switch (e.key) {
      case 'ArrowUp':    shift( s, 0, 0); break;
      case 'ArrowDown':  shift(-s, 0, 0); break;
      case 'ArrowRight': shift(0, -s, 0); break;
      case 'ArrowLeft':  shift(0,  s, 0); break;
      case 'w': case 'W':
        if (!e.ctrlKey && !e.metaKey) { shift(0, 0,  s); } else { handled = false; } break;
      case 's': case 'S':
        if (!e.ctrlKey && !e.metaKey) { shift(0, 0, -s); } else { handled = false; } break;
      case 'r': case 'R':
        if (!e.ctrlKey && !e.metaKey) { curr = [0, 0, 0]; apply(); } else { handled = false; } break;
      default: handled = false;
    }
    if (handled) { e.preventDefault(); e.stopPropagation(); }
  });
})();
</script>
"""


# ===================================================================
# print mapping
# ===================================================================

def print_mapping_table(robot_mapped, human_to_robot, human_mapped):
    print("\n" + "=" * 70)
    print("IK Config Mapping Summary (ik_match_table1)")
    print("=" * 70)
    print(f"  Robot bodies with mapping:  {len(robot_mapped)}")
    print(f"  Human joints with mapping:  {len(human_mapped)}\n")
    r2h: dict = {}
    for h_name, r_names in human_to_robot.items():
        for rn in r_names:
            r2h.setdefault(rn, []).append(h_name)
    print(f"  {'Robot Body':<30s} →  Human Joint(s)")
    print(f"  {'-'*30}     {'-'*30}")
    for rb in sorted(r2h):
        print(f"  {rb:<30s} →  {', '.join(r2h[rb])}")
    aux_mapped = [b for b in r2h if b.startswith("aux_")]
    if aux_mapped:
        print(f"\n  Aux bodies used as mapping targets: {len(aux_mapped)}")
        for b in aux_mapped:
            print(f"    {b}  →  {', '.join(r2h[b])}")
    print("=" * 70)


# ===================================================================
# Main
# ===================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Visualize robot↔human links mapping (plotly 3D)")
    ap.add_argument("--xml", type=str, default=DEFAULT_XML,
                    help="Robot XML path (relative to project root)")
    ap.add_argument("--bvh", type=str, default=DEFAULT_BVH,
                    help="BVH file path (relative to project root)")
    ap.add_argument("--ik_config", type=str, default=None,
                    help="IK config JSON path (relative to project root)")
    ap.add_argument("-o", "--output", type=str, default=None,
                    help="Save HTML to this path instead of temp file")
    ap.add_argument("--no-frames", action="store_true",
                    help="Hide RGB orientation axes on human joints")
    ap.add_argument("--no-open", action="store_true",
                    help="Do not open browser, just save HTML")
    ap.add_argument("--no-show", action="store_true",
                    help="Print mapping table only, do not generate HTML")
    ap.add_argument("--human-height", type=float, default=None,
                    help="Actual human height (m). Overrides ratio. "
                         "Default: use scale table as-is (ratio=1.0)")
    args = ap.parse_args()

    xml_path = (_PROJECT_ROOT / args.xml).resolve()
    bvh_path = (_PROJECT_ROOT / args.bvh).resolve()
    ik_path = (_PROJECT_ROOT / args.ik_config).resolve() if args.ik_config else None

    for p, label in [(xml_path, "XML"), (bvh_path, "BVH")]:
        if not p.exists():
            print(f"ERROR: {label} not found: {p}")
            sys.exit(1)
    if ik_path and not ik_path.exists():
        print(f"WARNING: IK config not found ({ik_path}), proceeding without it.")

    print(f"Robot XML : {xml_path}")
    print(f"BVH file  : {bvh_path}")
    print(f"IK config : {ik_path if ik_path else '(none)'}")

    # ---- 1. Robot ----
    print("\n[1/4] Loading robot model …")
    robot_model, robot_data = load_robot_model(xml_path)
    robot_bodies = get_robot_body_info(robot_model, robot_data)
    robot_positions = get_robot_body_positions(robot_bodies)
    robot_orientations = get_robot_body_orientations(robot_bodies)
    robot_edges = get_robot_body_edges(robot_bodies)
    print(f"  {len(robot_bodies)} bodies, {len(robot_edges)} edges")

    # ---- 2. BVH ----
    print("[2/4] Loading BVH first frame …")
    human_data, human_edges = load_bvh_first_frame(bvh_path)
    print(f"  {len(human_data)} joints, {len(human_edges)} edges")

    # ---- 3. IK config ----
    robot_mapped: set = set()
    human_mapped: set = set()
    human_to_robot: dict = {}

    if ik_path and ik_path.exists():
        print("[3/4] Applying IK config transformations …")
        with open(ik_path) as fh:
            ik_config = json.load(fh)
        ratio = 1.0
        if args.human_height is not None:
            ratio = args.human_height / ik_config["human_height_assumption"]
            print(f"  Human height: {args.human_height:.2f} m  →  "
                  f"ratio = {args.human_height:.2f} / "
                  f"{ik_config['human_height_assumption']:.2f} = {ratio:.4f}")
        human_data, human_to_robot = process_human_data(human_data, ik_config, ratio)
        for h_name, r_names in human_to_robot.items():
            human_mapped.add(h_name)
            robot_mapped.update(r_names)
        print(f"  Mapped {len(human_mapped)} human joints → "
              f"{len(robot_mapped)} robot bodies")
        print_mapping_table(robot_mapped, human_to_robot, human_mapped)
    else:
        print("[3/4] No IK config — showing raw BVH vs robot positions.")

    if args.no_show:
        print("[4/4] Skipping visualization (--no-show).")
        return

    # ---- 4. Build figure ----
    print("[4/4] Building 3D visualization …")
    title_parts = [f"{xml_path.name}  ↔  {bvh_path.name}"]
    if ik_path:
        title_parts.append(f"[{ik_path.name}]")
    title = "  |  ".join(title_parts)

    fig, robot_original = build_figure(
        robot_bodies, robot_positions, robot_orientations, robot_edges,
        human_data, human_edges,
        robot_mapped, human_mapped,
        show_frames=not args.no_frames,
        title=title,
    )

    # ---- Save & open ----
    os.chdir(_PROJECT_ROOT)
    if args.output:
        html_path = pathlib.Path(args.output).resolve()
    else:
        import tempfile
        html_path = pathlib.Path(tempfile.mktemp(suffix=".html"))

    # Generate HTML, then inject slider controls
    html = fig.to_html(include_plotlyjs="cdn", full_html=True, div_id="gmr-graph")
    import json as _json
    slider_code = ROBOT_SLIDER_JS.replace(
        "__ROBOT_ORIG_DATA__", _json.dumps(robot_original))
    html = html.replace("</body>", slider_code + "\n</body>")

    with open(str(html_path), "w") as fh:
        fh.write(html)
    print(f"  HTML saved: {html_path}")

    if not args.no_open:
        webbrowser.open(f"file://{html_path}")
        print("  Opened in browser.")

    print("\n  Legend:")
    print(f"    ● {C_ROBOT_MAPPED}        = mapped robot body")
    print(f"    ● {C_ROBOT_UNMAPPED}        = unmapped robot body")
    print(f"    ● {C_HUMAN_MAPPED}        = mapped human joint")
    print(f"    ● {C_HUMAN_UNMAPPED}        = unmapped human joint")
    print(f"    ┃ red/green/blue axes = orientation (X/Y/Z) on both sides")
    print(f"    🖱 hover any point → shows name + (x, y, z) coordinates")


if __name__ == "__main__":
    main()
