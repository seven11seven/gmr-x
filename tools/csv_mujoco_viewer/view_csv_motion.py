#!/usr/bin/env python3
"""
Visualise a retargeted-motion CSV file in MuJoCo.

The CSV must follow the 36-column format produced by
``visualize_retargeting.py --save-csv``::

    base_x, base_y, base_z,
    base_qx, base_qy, base_qz, base_qw,     ← xyzw order
    waist_yaw, waist_roll, waist_pitch,
    L_shoulder_pitch, L_shoulder_roll, L_shoulder_yaw,
    L_elbow_pitch, L_wrist_roll, L_wrist_pitch, L_wrist_yaw,
    R_shoulder_pitch, R_shoulder_roll, R_shoulder_yaw,
    R_elbow_pitch, R_wrist_roll, R_wrist_pitch, R_wrist_yaw,
    L_hip_pitch, L_hip_roll, L_hip_yaw,
    L_knee_pitch, L_ankle_pitch, L_ankle_roll,
    R_hip_pitch, R_hip_roll, R_hip_yaw,
    R_knee_pitch, R_ankle_pitch, R_ankle_roll

Quickstart::

    python tools/csv_mujoco_viewer/view_csv_motion.py motion.csv

    python tools/csv_mujoco_viewer/view_csv_motion.py motion.csv \\
        --xml assets/D20_29dof/mjcf/D20.xml
"""

import argparse
import pathlib
import sys
import time

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import mujoco as mj
import mujoco.viewer as mjv

DEFAULT_XML = "assets/D20_29dof/mjcf/D20.xml"

# Ground-projection marker parameters (drawn on the z=0 ground plane).
_GROUND_OFFSET = 0.01          # small lift off the ground to avoid z-fighting
_COM_MARKER_SIZE = 0.04
_ANKLE_MARKER_SIZE = 0.03
_ANKLE_LINE_RADIUS = 0.012
_COM_RGBA = (1.0, 0.0, 0.0, 0.9)        # red    – robot centre of mass
_LEFT_ANKLE_RGBA = (0.0, 1.0, 0.0, 0.9)  # green  – left ankle
_RIGHT_ANKLE_RGBA = (0.0, 0.0, 1.0, 0.9) # blue   – right ankle
_ANKLE_LINE_RGBA = (1.0, 1.0, 0.0, 0.9)  # yellow – ankle-to-ankle segment


# ============================================================================
#  Load CSV
# ============================================================================

def _is_pickle(path: pathlib.Path) -> bool:
    """Heuristic: first two bytes of pickle protocol ≥2 are 0x80 0x02–0x05."""
    with open(path, "rb") as fh:
        head = fh.read(2)
    return len(head) == 2 and head[0] == 0x80 and 0x02 <= head[1] <= 0x05


def load_csv(csv_path: pathlib.Path):
    """Return (all_qpos, fps, num_joints) from a retargeting CSV (or PKL).

    all_qpos   – (N, C) float array  [base_xyz, base_quat_xyzw, joints]
    fps        – float
    num_joints – number of actuated joints (= C - 7)
    """
    # Auto-detect pickle files (even if named .csv)
    if _is_pickle(csv_path):
        import pickle
        print("  (detected pickle format – loading via pickle)")
        with open(csv_path, "rb") as fh:
            cache = pickle.load(fh)
        all_frame_data = cache["all_frame_data"]
        fps = cache.get("motion_fps", 120.0)
        rows = []
        for fd in all_frame_data:
            qpos = fd["qpos"]
            base_quat_xyzw = np.roll(qpos[3:7], -1)  # wxyz → xyzw
            rows.append(np.concatenate([qpos[0:3], base_quat_xyzw, qpos[7:]]))
        data = np.stack(rows, axis=0)
        return data, fps, data.shape[1] - 7

    # Plain CSV
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data, 120.0, data.shape[1] - 7


def csv_to_mj_qpos(csv_row: np.ndarray, num_joints: int):
    """Convert one CSV row → MuJoCo qpos (wxyz quaternion).

    Works for any DOF count: qpos = [base_xyz, base_quat_wxyz, joints...]
    """
    nq = 7 + num_joints
    qpos = np.zeros(nq, dtype=np.float64)
    qpos[0:3] = csv_row[0:3]                      # base x, y, z
    qpos[3]   = csv_row[6]                         # qw
    qpos[4]   = csv_row[3]                         # qx
    qpos[5]   = csv_row[4]                         # qy
    qpos[6]   = csv_row[5]                         # qz
    qpos[7:]  = csv_row[7 : 7 + num_joints]        # joints
    return qpos


# ============================================================================
#  MuJoCo viewer
# ============================================================================

def _load_model_with_markers(xml_path: pathlib.Path):
    """Load the robot model and attach ground-projection marker geoms.

    Adds four mocap bodies (whose pose is driven per-frame via ``mocap_pos`` /
    ``mocap_quat``) so the playback loop can overlay on the ground plane::

        __com_proj     – red sphere:  projection of the robot centre of mass
        __l_ankle_proj – green sphere: projection of the left ankle
        __r_ankle_proj – blue sphere:  projection of the right ankle
        __ankle_line   – yellow capsule: segment joining the two ankle projections
    """
    spec = mj.MjSpec.from_file(str(xml_path))
    wb = spec.worldbody

    com = wb.add_body(name="__com_proj", mocap=True)
    com.add_geom(name="__com_proj_geom", type=mj.mjtGeom.mjGEOM_SPHERE,
                 size=[_COM_MARKER_SIZE], rgba=_COM_RGBA,
                 contype=0, conaffinity=0)

    l = wb.add_body(name="__l_ankle_proj", mocap=True)
    l.add_geom(name="__l_ankle_proj_geom", type=mj.mjtGeom.mjGEOM_SPHERE,
               size=[_ANKLE_MARKER_SIZE], rgba=_LEFT_ANKLE_RGBA,
               contype=0, conaffinity=0)

    r = wb.add_body(name="__r_ankle_proj", mocap=True)
    r.add_geom(name="__r_ankle_proj_geom", type=mj.mjtGeom.mjGEOM_SPHERE,
               size=[_ANKLE_MARKER_SIZE], rgba=_RIGHT_ANKLE_RGBA,
               contype=0, conaffinity=0)

    line = wb.add_body(name="__ankle_line", mocap=True)
    line.add_geom(name="__ankle_line_geom", type=mj.mjtGeom.mjGEOM_CAPSULE,
                  size=[_ANKLE_LINE_RADIUS, 0.1, 0.0], rgba=_ANKLE_LINE_RGBA,
                  contype=0, conaffinity=0)

    return spec.compile()


def _find_ankle_body_id(model, side: str):
    """Auto-detect the ``side`` ('left'/'right') ankle/foot body id.

    Picks a body whose name contains both ``side`` and 'ankle', preferring the
    foot link ('roll' / 'foot' in the name) over auxiliary frames.
    """
    candidates = []
    for i in range(model.nbody):
        name = model.body(i).name or ""
        if side in name and "ankle" in name:
            candidates.append((i, name))
    if not candidates:
        return None

    def _score(item):
        _, name = item
        s = 0
        if "roll" in name or "foot" in name:
            s += 2
        if "aux" in name or "end" in name or "toe" in name:
            s -= 1
        return s

    candidates.sort(key=_score, reverse=True)
    return candidates[0][0]


def _resolve_body_id(model, preferred: str, side: str):
    """Return a body id for ``preferred``, falling back to auto-detection."""
    if preferred:
        try:
            return model.body(preferred).id
        except Exception:
            pass
    return _find_ankle_body_id(model, side)


def _update_ground_markers(model, data, com_mocapid, l_mocapid, r_mocapid,
                           line_mocapid, line_geomid, left_id, right_id):
    """Project the CoM and ankle markers onto the z=0 ground plane."""
    # Robot centre of mass → ground projection
    com = data.subtree_com[0]
    data.mocap_pos[com_mocapid] = np.array([com[0], com[1], _GROUND_OFFSET])

    if left_id is None or right_id is None:
        return

    lp = data.xpos[left_id]
    rp = data.xpos[right_id]
    l_proj = np.array([lp[0], lp[1], _GROUND_OFFSET])
    r_proj = np.array([rp[0], rp[1], _GROUND_OFFSET])

    data.mocap_pos[l_mocapid] = l_proj
    data.mocap_pos[r_mocapid] = r_proj

    # Ankle-to-ankle segment (capsule whose axis is local +Z).
    mid = 0.5 * (l_proj + r_proj)
    direc = r_proj - l_proj
    length = float(np.linalg.norm(direc))
    data.mocap_pos[line_mocapid] = mid

    q = np.zeros(4)
    if length > 1e-9:
        mj.mju_quatZ2Vec(q, direc)
    else:
        q[0] = 1.0
    data.mocap_quat[line_mocapid] = q
    model.geom_size[line_geomid] = [_ANKLE_LINE_RADIUS, 0.5 * length, 0.0]


def play_csv_in_mujoco(
    xml_path: pathlib.Path,
    all_qpos: np.ndarray,
    motion_fps: float = 120.0,
    num_joints: int = 29,
    left_ankle: str = "left_ankle_roll_link",
    right_ankle: str = "right_ankle_roll_link",
):
    """Open a MuJoCo passive viewer and play the motion.

    Overlays the robot's centre-of-mass ground projection (red sphere), the
    ankle ground projections (green/blue spheres) and the segment joining the
    two ankles (yellow capsule).
    """
    model = _load_model_with_markers(xml_path)
    data = mj.MjData(model)
    total_frames = len(all_qpos)

    # Marker handles
    com_mocapid = model.body("__com_proj").mocapid
    l_mocapid = model.body("__l_ankle_proj").mocapid
    r_mocapid = model.body("__r_ankle_proj").mocapid
    line_mocapid = model.body("__ankle_line").mocapid
    line_geomid = model.geom("__ankle_line_geom").id

    # Resolve ankle bodies (auto-detect when the named body is absent).
    left_id = _resolve_body_id(model, left_ankle, "left")
    right_id = _resolve_body_id(model, right_ankle, "right")
    if left_id is None or right_id is None:
        print("  WARNING: could not locate ankle bodies – ankle markers disabled.")
        left_id = right_id = None

    # trackball camera
    model.stat.center = np.array([0.0, 0.0, 0.8])
    model.stat.extent = 2.0

    state = {
        "current": 0,
        "playing": True,
        "speed": 1.0,
    }

    def key_callback(keycode: int):
        c = state
        if keycode == 263:   # LEFT
            c["current"] = max(c["current"] - 1, 0)
        elif keycode == 262: # RIGHT
            c["current"] = min(c["current"] + 1, total_frames - 1)
        elif keycode == 265: # UP
            c["current"] = min(c["current"] + 10, total_frames - 1)
        elif keycode == 264: # DOWN
            c["current"] = max(c["current"] - 10, 0)
        elif keycode == 268: # HOME
            c["current"] = 0
        elif keycode == 269: # END
            c["current"] = total_frames - 1
        elif keycode == 32:  # SPACE
            c["playing"] = not c["playing"]
        elif keycode == 49:  c["speed"] = 0.25
        elif keycode == 50:  c["speed"] = 0.50
        elif keycode == 51:  c["speed"] = 1.00
        elif keycode == 52:  c["speed"] = 2.00
        elif keycode == 53:  c["speed"] = 4.00
        else:
            return
        _print_status(state, total_frames)

    viewer = mjv.launch_passive(
        model=model, data=data,
        show_left_ui=False, show_right_ui=False,
        key_callback=key_callback,
    )

    print("\n  Controls:")
    print("    ← →  prev/next frame    ↑ ↓  jump ±10")
    print("    Space  play/pause       1-5  speed (0.25× … 4×)")
    print("    Home/End  first/last    Esc  close")
    print("    Markers: red=CoM  green/blue=ankles  yellow=ankle line\n")

    _print_status(state, total_frames)

    prev_frame = -1
    last_time = time.time()

    while viewer.is_running():
        now = time.time()

        if state["playing"]:
            dt = now - last_time
            advance = int(dt * motion_fps * state["speed"])
            if advance > 0:
                advance = min(advance, max(1, int(motion_fps * 0.5)))
                state["current"] = (state["current"] + advance) % total_frames
                last_time = now

        if state["current"] != prev_frame:
            prev_frame = state["current"]
            data.qpos[:] = csv_to_mj_qpos(all_qpos[state["current"]], num_joints)
            mj.mj_forward(model, data)
            _update_ground_markers(
                model, data,
                com_mocapid, l_mocapid, r_mocapid, line_mocapid,
                line_geomid, left_id, right_id,
            )

        viewer.sync()

        if not state["playing"]:
            time.sleep(0.02)

    viewer.close()
    print("  Viewer closed.")


def _print_status(state, total_frames):
    icon = "▶" if not state["playing"] else "⏸"
    print(f"  {icon} Frame {state['current'] + 1}/{total_frames}  "
          f"speed={state['speed']:.2f}×")


# ============================================================================
#  Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Visualise a retargeting CSV in MuJoCo.")
    ap.add_argument("csv", type=str, help="Path to CSV file")
    ap.add_argument("--xml", type=str, default=DEFAULT_XML,
                    help="Robot MuJoCo XML (default: D20_29dof)")
    ap.add_argument("--fps", type=float, default=120.0,
                    help="Motion frame rate (default: 120)")
    ap.add_argument("--start", type=int, default=0,
                    help="First frame index (0-based)")
    ap.add_argument("--end", type=int, default=None,
                    help="Last frame index (exclusive)")
    ap.add_argument("--left-ankle", type=str, default="left_ankle_roll_link",
                    help="Body name of the left ankle/foot "
                         "(auto-detected if not found)")
    ap.add_argument("--right-ankle", type=str, default="right_ankle_roll_link",
                    help="Body name of the right ankle/foot "
                         "(auto-detected if not found)")
    args = ap.parse_args()

    csv_path = pathlib.Path(args.csv).resolve()
    xml_path = (_PROJECT_ROOT / args.xml).resolve()

    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}")
        sys.exit(1)
    if not xml_path.exists():
        print(f"ERROR: XML not found: {xml_path}")
        sys.exit(1)

    print(f"CSV : {csv_path}")
    print(f"XML : {xml_path}")

    all_qpos, detected_fps, num_joints = load_csv(csv_path)
    all_qpos = all_qpos[args.start:args.end]
    fps = args.fps if args.fps != 120.0 else detected_fps
    print(f"Frames: {len(all_qpos)}  cols={7 + num_joints} "
          f"({num_joints} DOF)  (fps={fps})")

    play_csv_in_mujoco(xml_path, all_qpos, fps, num_joints,
                       left_ankle=args.left_ankle,
                       right_ankle=args.right_ankle)


if __name__ == "__main__":
    main()
