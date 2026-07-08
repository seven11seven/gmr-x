"""
T-Pose Viewer: Visualize the scaled Xsens human skeleton in T-pose alongside the robot.

All scale/offset parameters and the human-to-robot bone mapping are read from the
IK config JSON file (e.g. xsens_realtime_to_d20_v2.json).  The skeleton hierarchy
is derived from the robot's MuJoCo body tree paired with the IK match table — no
hardcoded bone names or scale factors outside that config.

Usage:
    python debug_tools/tpose_viewer.py --xsens_file datas/xsens_record_20260707_150940.pkl --robot d20_v2
    python debug_tools/tpose_viewer.py --xsens_file datas/xsens_record_20260707_150940.pkl --robot unitree_g1
"""

import argparse
import pickle
import numpy as np
from scipy.spatial.transform import Rotation as R
import mujoco as mj
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from rich import print


# T-pose canonical directions (world frame, relative to parent).
# These are anatomical definitions of "T-pose" — not IK parameters — and are
# needed to decide *which way* each bone should point when the skeleton is
# straightened.  Only bones that appear in the IK config's human_scale_table
# need an entry here.
TPOSE_DIRECTIONS = {
    # Spine / torso
    "T12":           np.array([0, 0, 1]),
    # Upper legs: downward
    "LeftUpperLeg":  np.array([0, 0, -1]),
    "RightUpperLeg": np.array([0, 0, -1]),
    # Lower legs: downward
    "LeftLowerLeg":  np.array([0, 0, -1]),
    "RightLowerLeg": np.array([0, 0, -1]),
    # Feet / FootMod: forward
    "LeftFootMod":   np.array([1, 0, 0]),
    "RightFootMod":  np.array([1, 0, 0]),
    # Upper arms: horizontal, out to sides
    "LeftUpperArm":  np.array([0, -1, 0]),
    "RightUpperArm": np.array([0, 1, 0]),
    # Forearms: downward
    "LeftForearm":   np.array([0, 0, -1]),
    "RightForearm":  np.array([0, 0, -1]),
    # Hands: downward
    "LeftHand":      np.array([0, 0, -1]),
    "RightHand":     np.array([0, 0, -1]),
}

IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


# ---------------------------------------------------------------------------
# Xsens data helpers
# ---------------------------------------------------------------------------

def create_human_data_from_xsens(frame: dict, segment_id_map: dict) -> dict:
    """
    Convert one Xsens frame to the human_data dict expected by GMR.

    Xsens provides global positions (meters) and scalar-first quaternions for
    each segment.  No FK is needed — all positions are world-frame.
    """
    segments = frame["segments"]
    human_data = {}
    for seg_id_str, seg_data in segments.items():
        name = segment_id_map.get(int(seg_id_str))
        if name is None:
            continue
        human_data[name] = [
            np.array(seg_data["pos"], dtype=np.float64),
            np.array(seg_data["quat"], dtype=np.float64),
        ]

    # Synthetic bones required by the IK config (FootMod = Foot position × Toe rotation)
    for side in ("Left", "Right"):
        foot = f"{side}Foot"
        toe = f"{side}Toe"
        mod = f"{side}FootMod"
        if foot in human_data and toe in human_data and mod not in human_data:
            human_data[mod] = [human_data[foot][0].copy(), human_data[toe][1].copy()]

    return human_data


# ---------------------------------------------------------------------------
# Skeleton hierarchy derived from the IK config
# ---------------------------------------------------------------------------

def derive_skeleton_hierarchy(retargeter) -> list:
    """
    Build parent→child connections for the human skeleton using the robot's
    MuJoCo body tree and the IK match table 1.

    The IK config maps robot bodies → human bone names.  Walking the robot
    body tree and finding the nearest mapped ancestor for each mapped body
    gives us the parent→child relationship between the corresponding human bones.
    """
    # robot_body_name → human_bone_name  (from ik_match_table1)
    robot_to_human: dict[str, str] = {}
    for robot_body, entry in retargeter.ik_match_table1.items():
        human_name = entry[0]
        robot_to_human[robot_body] = human_name

    # Build parent → child for human bones
    connections: list[tuple[str, str]] = []
    for robot_name, human_name in robot_to_human.items():
        body_id = mj.mj_name2id(retargeter.model, mj.mjtObj.mjOBJ_BODY, robot_name)
        if body_id < 0:
            continue
        # Walk up the robot body tree to find nearest mapped ancestor
        parent_id = retargeter.model.body_parentid[body_id]
        while parent_id > 0:
            parent_robot = mj.mj_id2name(retargeter.model, mj.mjtObj.mjOBJ_BODY, parent_id)
            if parent_robot and parent_robot in robot_to_human:
                parent_human = robot_to_human[parent_robot]
                if parent_human != human_name:
                    connections.append((parent_human, human_name))
                break
            parent_id = retargeter.model.body_parentid[parent_id]

    return connections


# ---------------------------------------------------------------------------
# T-pose reconstruction
# ---------------------------------------------------------------------------

def convert_to_tpose(human_data: dict, connections: list, root_name: str) -> dict:
    """
    Rebuild the skeleton in T-pose while preserving scaled bone geometry.

    1. Capture the full 3D parent→child offset from the scaled positions.
    2. Walk root→leaves: for each bone keep the lateral (⊥ T-pose dir)
       component of the original offset, and realign the parallel component
       along the canonical T-pose direction — preserving total bone length.
    3. Set all quaternions to identity.
    """
    children_of: dict[str, list] = {}
    for parent, child in connections:
        children_of.setdefault(parent, []).append(child)

    offsets: dict[str, np.ndarray] = {}
    bone_lengths: dict[str, float] = {}
    for parent, child in connections:
        p_pos = human_data.get(parent)
        c_pos = human_data.get(child)
        if p_pos is not None and c_pos is not None:
            offset = c_pos[0] - p_pos[0]
            offsets[child] = offset
            bone_lengths[child] = float(np.linalg.norm(offset))

    result = {}
    root_pos = human_data[root_name][0].copy()
    result[root_name] = [root_pos, IDENTITY_QUAT.copy()]

    queue = [root_name]
    while queue:
        parent = queue.pop(0)
        for child in children_of.get(parent, []):
            if child not in bone_lengths:
                continue
            tpose_dir = TPOSE_DIRECTIONS.get(child, np.array([0, 0, 1]))
            tpose_dir = tpose_dir / np.linalg.norm(tpose_dir)

            offset = offsets[child]
            bone_len = bone_lengths[child]

            parallel = np.dot(offset, tpose_dir)
            lateral = offset - tpose_dir * parallel

            lateral_sq = float(np.dot(lateral, lateral))
            tpose_parallel = np.sqrt(max(0.0, bone_len * bone_len - lateral_sq))

            child_pos = result[parent][0] + lateral + tpose_dir * tpose_parallel
            result[child] = [child_pos, IDENTITY_QUAT.copy()]
            queue.append(child)

    return result


def reposition_human_root(human_data: dict, root_name: str, target_pos: np.ndarray) -> dict:
    delta = target_pos - human_data[root_name][0]
    return {name: [pos + delta, quat] for name, (pos, quat) in human_data.items()}


# ---------------------------------------------------------------------------
# Skeleton line drawing
# ---------------------------------------------------------------------------

def draw_skeleton_line(v, p_from, p_to, rgba, width=0.005):
    """Add a single line connector to the viewer's user_scn."""
    mj.mjv_connector(
        v.user_scn.geoms[v.user_scn.ngeom],
        type=mj.mjtGeom.mjGEOM_LINE,
        width=width,
        from_=p_from,
        to=p_to,
    )
    for i in range(4):
        v.user_scn.geoms[v.user_scn.ngeom].rgba[i] = rgba[i]
    v.user_scn.ngeom += 1


def draw_skeleton_links(viewer, human_data: dict, connections: list,
                        rgba=(0.2, 0.8, 0.8, 0.7), width=0.005):
    """Draw line segments connecting parent→child bone pairs."""
    for parent, child in connections:
        if parent in human_data and child in human_data:
            draw_skeleton_line(
                viewer, human_data[parent][0], human_data[child][0],
                rgba=rgba, width=width,
            )
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize scaled Xsens human skeleton in T-pose alongside robot"
    )
    parser.add_argument("--xsens_file", required=True, help="Path to mock Xsens pickle file.")
    parser.add_argument(
        "--robot", default="d20_v2", choices=["unitree_g1", "d20_v2"],
        help="Robot name (must have an xsens_realtime IK config).",
    )
    parser.add_argument("--human_height", type=float, default=1.75)
    parser.add_argument("--root_height", type=float, default=1.8,
                        help="Root (pelvis / base_link) Z position for both skeletons.")
    parser.add_argument("--frame_idx", type=int, default=0)
    parser.add_argument("--human_offset_x", type=float, default=0.0)
    parser.add_argument("--human_offset_y", type=float, default=0.0)
    parser.add_argument("--show_body_names", action="store_true", default=True)
    parser.add_argument("--no_show_body_names", action="store_false", dest="show_body_names")
    args = parser.parse_args()

    # ---- 1. Load Xsens pickle ------------------------------------------------
    print(f"[bold]Loading Xsens data from:[/] {args.xsens_file}")
    with open(args.xsens_file, "rb") as f:
        xsens_data = pickle.load(f)
    meta = xsens_data["meta"]
    frames = xsens_data["frames"]
    print(f"  Recorded: {meta['created']},  frame count: {len(frames)}")

    if not (0 <= args.frame_idx < len(frames)):
        raise ValueError(f"frame_idx {args.frame_idx} out of range (0–{len(frames)-1})")

    # segment_id_map comes from the pickle metadata (recorded at capture time)
    seg_id_map = meta.get("segment_id_map", {})
    if isinstance(seg_id_map, list):
        seg_id_map = dict(enumerate(seg_id_map))

    frame = frames[args.frame_idx]
    human_data = create_human_data_from_xsens(frame, seg_id_map)
    print(f"  Extracted {len(human_data)} body parts from Xsens frame {args.frame_idx}")

    # ---- 2. Init GMR (loads robot model & IK config) -------------------------
    print(f"[bold]Initializing GMR for robot:[/] {args.robot}")
    retargeter = GMR(
        src_human="xsens_realtime", tgt_robot=args.robot,
        actual_human_height=args.human_height, verbose=False,
    )

    # ---- Derive human skeleton hierarchy from IK config + robot body tree ----
    skeleton_connections = derive_skeleton_hierarchy(retargeter)
    config_bones = set(retargeter.human_scale_table.keys())
    print(f"  Config bones (scale_table): {sorted(config_bones)}")
    print(f"  Derived {len(skeleton_connections)} parent→child links from IK match table + robot tree")

    # ---- Robot default-pose config -------------------------------------------
    robot_root_rpy_deg = [0.0, 0.0, -90.0]
    robot_joint_offsets_deg = {
        "left_shoulder_roll_joint": 90.0,
        "right_shoulder_roll_joint": -90.0,
        "left_elbow_pitch_joint": 90.0,
        "right_elbow_pitch_joint": 90.0,
    }

    dof_name_to_qpos_idx = {}
    for i in range(retargeter.model.nv):
        jnt_id = retargeter.model.dof_jntid[i]
        if jnt_id < 0:
            continue
        qpos_adr = retargeter.model.jnt_qposadr[jnt_id]
        if qpos_adr < 7:
            continue
        name = mj.mj_id2name(retargeter.model, mj.mjtObj.mjOBJ_JOINT, jnt_id)
        if name and name not in dof_name_to_qpos_idx:
            dof_name_to_qpos_idx[name] = qpos_adr

    root_rot_quat = R.from_euler("xyz", np.deg2rad(robot_root_rpy_deg)).as_quat(scalar_first=True)
    robot_qpos = retargeter.model.qpos0.copy()
    robot_qpos[2] = args.root_height
    robot_qpos[3:7] = root_rot_quat

    for joint_name, offset_deg in robot_joint_offsets_deg.items():
        if joint_name in dof_name_to_qpos_idx:
            robot_qpos[dof_name_to_qpos_idx[joint_name]] += np.deg2rad(offset_deg)
        else:
            print(f"  [yellow]Warning:[/] joint '{joint_name}' not found in robot model")

    print(f"  Robot root RPY: roll={robot_root_rpy_deg[0]:.1f}°  "
          f"pitch={robot_root_rpy_deg[1]:.1f}°  yaw={robot_root_rpy_deg[2]:.1f}°")
    if robot_joint_offsets_deg:
        print(f"  Joint offsets applied:")
        for name, deg in robot_joint_offsets_deg.items():
            print(f"    {name}: {deg:+.1f}°")

    # ---- 3. Scale with config's human_scale_table, then rebuild in T-pose ----
    human_data = retargeter.to_numpy(human_data)

    # Only keep bones that are in the config's scale_table
    human_data = {k: v for k, v in human_data.items() if k in config_bones}

    human_data = retargeter.scale_human_data(
        human_data, retargeter.human_root_name,
        retargeter.human_scale_table_vertical,
        retargeter.human_scale_table_horizontal,
    )
    print(f"  After config-scale: {len(human_data)} bones")

    # T-pose reconstruction using config-derived hierarchy
    human_data = convert_to_tpose(
        human_data, skeleton_connections, retargeter.human_root_name,
    )
    print(f"  After T-pose: {len(human_data)} bones")

    # Apply pos/rot offsets from IK config
    human_data = retargeter.offset_human_data(
        human_data, retargeter.pos_offsets1, retargeter.rot_offsets1,
    )

    # Position human root at the shared root position
    human_root_target = np.array([args.human_offset_x, args.human_offset_y, args.root_height])
    human_data = reposition_human_root(
        human_data, retargeter.human_root_name, human_root_target,
    )

    # ---- 4. Robot mutable state ----------------------------------------------
    robot_root = [float(robot_qpos[0]), float(robot_qpos[1]), float(robot_qpos[2])]
    step_size = [0.01]
    print(f"  Robot DOF count: {retargeter.model.nv}")

    # ---- 5. Keyboard callback ------------------------------------------------
    KEY_R, KEY_1, KEY_2, KEY_3 = 82, 49, 50, 51
    KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT = 265, 264, 263, 262
    KEY_KP_2, KEY_KP_8 = 322, 328

    def make_keyboard_callback(robot_root, step_size, root_height):
        def on_key(keycode: int) -> None:
            step = step_size[0]
            if keycode == KEY_UP:
                robot_root[0] += step
            elif keycode == KEY_DOWN:
                robot_root[0] -= step
            elif keycode == KEY_LEFT:
                robot_root[1] -= step
            elif keycode == KEY_RIGHT:
                robot_root[1] += step
            elif keycode == KEY_KP_2:
                robot_root[2] -= step
            elif keycode == KEY_KP_8:
                robot_root[2] += step
            elif keycode == KEY_R:
                robot_root[0] = 0.0
                robot_root[1] = 0.0
                robot_root[2] = root_height
            elif keycode == KEY_1:
                step_size[0] = 0.01
            elif keycode == KEY_2:
                step_size[0] = 0.05
            elif keycode == KEY_3:
                step_size[0] = 0.10
            else:
                return
            print(
                f"  Robot root: X={robot_root[0]:.3f}  Y={robot_root[1]:.3f}  Z={robot_root[2]:.3f}"
                f"  |  step={step_size[0]:.2f}m"
            )
        return on_key

    key_callback = make_keyboard_callback(robot_root, step_size, args.root_height)

    # ---- 6. Launch viewer ----------------------------------------------------
    print(f"\n[bold green]Launching T-Pose Viewer[/]")
    print(f"  Robot:          {args.robot}  (transparent)")
    print(f"  Xsens file:     {args.xsens_file}")
    print(f"  Frame idx:      {args.frame_idx}  (used for bone-length measurement)")
    print(f"  Pose:           T-pose  (identity quaternions, canonical directions)")
    print(f"  Root height:    {args.root_height} m  (shared)")
    print(f"  Human offset:   X={args.human_offset_x}, Y={args.human_offset_y}")
    print(f"")
    print(f"  [bold]Robot controls (focus the MuJoCo window):[/]")
    print(f"    [cyan]↑/↓[/]           — move robot  forward / backward  (X axis)")
    print(f"    [cyan]←/→[/]           — move robot  left / right       (Y axis)")
    print(f"    [cyan]Numpad 2/8[/]    — move robot  down / up          (Z axis)")
    print(f"    [cyan]R[/]             — reset robot to (0, 0, {args.root_height})")
    print(f"    [cyan]1/2/3[/]         — step size: 0.01 / 0.05 / 0.10 m")
    print(f"")
    print(f"  [dim]Left-drag to rotate | Right-drag to pan | Scroll to zoom[/]")
    print(f"  [dim]Close the viewer window to exit.[/]")

    viewer = RobotMotionViewer(
        robot_type=args.robot, motion_fps=30, transparent_robot=1,
        keyboard_callback=key_callback,
    )
    viewer.viewer.cam.lookat = [0, 0, args.root_height]
    viewer.viewer.cam.distance = viewer.viewer_cam_distance
    viewer.viewer.cam.elevation = -15

    try:
        while viewer.viewer.is_running():
            viewer.step(
                root_pos=np.array(robot_root),
                root_rot=robot_qpos[3:7],
                dof_pos=robot_qpos[7:],
                human_motion_data=human_data,
                show_human_body_name=args.show_body_names,
                human_point_scale=0.08,
                human_pos_offset=np.zeros(3),
                follow_camera=False,
                rate_limit=True,
            )
            # Draw skeleton links on top of the coordinate frames
            draw_skeleton_links(
                viewer.viewer, human_data, skeleton_connections,
                rgba=(0.2, 0.8, 0.8, 0.7), width=0.015,
            )
            viewer.viewer.sync()
    except KeyboardInterrupt:
        print("\n[bold yellow]Interrupted by user.[/]")
    finally:
        viewer.close()
        print("[bold]Viewer closed.[/]")
