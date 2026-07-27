import argparse
import pathlib
import time
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.utils.xsens import load_xsens_file
from rich import print
from tqdm import tqdm
import os
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
def interpolate_to_from_default(motion_qpos, default_qpos, pre_steps=60, post_steps=60):
    """
    在动作序列前后添加平滑过渡段：
      - 前段：从 default 姿态 → 动作第一帧（位置固定为 default_pos）
      - 后段：从动作最后一帧 → default 姿态（位置固定为 default_pos）

    参数:
        motion_qpos: (T, D) array, D = 3(pos) + 4(quat) + N(joints)
        default_qpos: (D,) array, 默认姿态 [x,y,z, qw,qx,qy,qz, j0,j1,...]
        pre_steps: 前插值帧数（>=0）
        post_steps: 后插值帧数（>=0）

    返回:
        full_qpos: (pre_steps + T + post_steps, D)
    """
    T, D = motion_qpos.shape
    assert default_qpos.shape == (D,), f"default_qpos shape {default_qpos.shape} != ({D},)"

    # 拆分 default
    default_pos = default_qpos[:3].copy()      # 固定位置
    default_quat = default_qpos[3:7].copy()    # [w, x, y, z]
    default_joints = default_qpos[7:].copy()

    # 拆分首尾帧
    first = motion_qpos[0]
    last = motion_qpos[-1]

    first_pos = first[:3]
    first_quat = first[3:7]
    first_joints = first[7:]

    last_pos = last[:3]
    last_quat = last[3:7]
    last_joints = last[7:]

    # === 前插值段 ===
    if pre_steps > 0:
        # 时间点：0 → 1
        times_pre = np.linspace(0, 1, pre_steps)
        # 构造两个端点的 Rotation 对象（注意 scalar_first=True）
        rots_pre = R.from_quat([default_quat, first_quat], scalar_first=True)
        slerp_pre = Slerp([0, 1], rots_pre)
        
        pre_quats = slerp_pre(times_pre).as_quat(scalar_first=True)  # (pre_steps, 4)
        pre_joints = np.linspace(default_joints, first_joints, pre_steps)  # (pre_steps, N)

        pre_qpos = np.zeros((pre_steps, D))
        pre_qpos[:, :3] = first_pos  # 位置固定
        pre_qpos[:, 3:7] = first_quat
        pre_qpos[:, 7:] = pre_joints
    else:
        pre_qpos = np.zeros((0, D))

    # === 后插值段 ===
    if post_steps > 0:
        times_post = np.linspace(0, 1, post_steps)
        rots_post = R.from_quat([last_quat, default_quat], scalar_first=True)
        slerp_post = Slerp([0, 1], rots_post)
        
        post_quats = slerp_post(times_post).as_quat(scalar_first=True)
        post_joints = np.linspace(last_joints, default_joints, post_steps)

        post_qpos = np.zeros((post_steps, D))
        post_qpos[:, :3] = last_pos  # 位置固定
        post_qpos[:, 3:7] = last_quat
        post_qpos[:, 7:] = post_joints
    else:
        post_qpos = np.zeros((0, D))

    # 拼接：前 + 原始 + 后
    
    full_qpos = np.concatenate([pre_qpos, motion_qpos, post_qpos], axis=0)
    return full_qpos

if __name__ == "__main__":

    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bvh_file",
        help="BVH motion file to load.",
        # default="./xsens_bvh/xsens_walk.bvh",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--robot",
        choices=[
            "unitree_g1",
            "D12",
            "d20_v2",
            "dz_robot",
        ],
        default="D12",
    )

    parser.add_argument(
        "--ik_config",
        default=None,
        type=str,
        help="Optional IK config JSON path. Overrides params.py mapping.",
    )

    parser.add_argument(
        "--record_video",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--video_path",
        type=str,
        default="videos/example.mp4",
    )

    parser.add_argument(
        "--rate_limit",
        action="store_true",
        default=True,
    )

    parser.add_argument(
        "--save_path",
        # default="./retargeting_data/unitree_h1_2_xsens_walk.pkl",
        default=None,
        help="Path to save the robot motion.",
    )

    # parser.add_argument(
    #     "--axis_order",
    #     default="zxy",
    #     help="",
    # )

    parser.add_argument(
        "--scale",
        default=1,
        type=float,
        help="The scaling size is determined based on the units used for displacement",
    )

    parser.add_argument(
        "--reset_to_zero",
        action="store_true",
        default=False,
        help="Set the displacement and Z-axis rotation to zero",
    )

    parser.add_argument(
        "--start",
        default=None,
        type=int,
        help="The sequence number of the first frame that you want to process",
    )

    parser.add_argument(
        "--end",
        default=None,
        type=int,
        help="The sequence number of the last frame that you want to process",
    )

    parser.add_argument(
        "--bvh_format",
        default="3DSM",
        type=str,
        choices=[
            "3DSM",
            "MB",
            "P6",
        ],
        help="The format of bvh files,3ds Max,MotionBuilder,and P6?",
    )

    args = parser.parse_args()

    if args.save_path is not None:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:  # Only create directory if it's not empty
            os.makedirs(save_dir, exist_ok=True)
        qpos_list = []

    # Load SMPLX trajectory
    # lafan1_data_frames, actual_human_height = load_lafan1_file(args.bvh_file)
    lafan1_data_frames, actual_human_height,frame_time = load_xsens_file(args)

    # Initialize the retargeting system
    retargeter = GMR(
        src_human="bvh_xsens",
        tgt_robot=args.robot,
        actual_human_height=actual_human_height,
        ik_config_path=args.ik_config,
    )

    motion_fps = int(1/frame_time)
    print(f"motion_fps: {motion_fps}")

    robot_motion_viewer = RobotMotionViewer(
        robot_type=args.robot,
        motion_fps=motion_fps,
        transparent_robot=1,
        record_video=args.record_video,
        video_path=args.video_path,
        # video_width=2080,
        # video_height=1170
    )

    # FPS measurement variables
    fps_counter = 0
    fps_start_time = time.time()
    fps_display_interval = 2.0  # Display FPS every 2 seconds

    print(f"mocap_frame_rate: {motion_fps}")

    # Create tqdm progress bar for the total number of frames
    pbar = tqdm(total=len(lafan1_data_frames), desc="Retargeting")

    # Start the viewer
    i = 0

    while i < len(lafan1_data_frames):

        # FPS measurement
        fps_counter += 1
        current_time = time.time()
        if current_time - fps_start_time >= fps_display_interval:
            actual_fps = fps_counter / (current_time - fps_start_time)
            # print(f"Actual rendering FPS: {actual_fps:.2f}")
            fps_counter = 0
            fps_start_time = current_time

        # Update progress bar
        pbar.update(1)

        # Update task targets.
        smplx_data = lafan1_data_frames[i]
        smplx_data = lafan1_data_frames[i]

        # retarget
        qpos = retargeter.retarget(smplx_data, offset_to_ground=True)
        
        # qpos[21] = 0.2
        # qpos[24] = -0.2

        # visualize
        robot_motion_viewer.step(
            root_pos=qpos[:3] + [0.0, 0.0, 0.0],
            root_rot=qpos[3:7],
            dof_pos=qpos[7:],
            human_motion_data=retargeter.scaled_human_data,
            rate_limit=args.rate_limit,
            # human_pos_offset=np.array([0.0, 0.0, 0.0])
        )

        i += 1

        # if i ==  10: input("Input enter to continue")
        if i == 510: input("Input enter to continue")
        # if i == 800: input("Input enter to continue")

        if args.save_path is not None:
            if i > 10 and i < (len(lafan1_data_frames) - 10):
                qpos_list.append(qpos.copy())

    if args.save_path is not None:
        import pickle

        qpos_list = qpos_list[10:]
        root_pos = np.array([qpos[:3] for qpos in qpos_list])
        # save from wxyz to xyzw
        root_rot = np.array([qpos[3:7] for qpos in qpos_list])
        dof_pos = np.array([qpos[7:] for qpos in qpos_list])
        local_body_pos = None
        body_names = None
        
        csv_qpos = np.array(qpos_list, dtype=np.float32)
        full_motion = interpolate_to_from_default(
            csv_qpos,
            retargeter.robot_default_pos,
            pre_steps=240,
            post_steps=240
        )
        
        # np.set_printoptions(threshold=np.inf)
        # print(full_motion[:255,8])

        csv_motion = full_motion.copy()
        csv_motion[:, 3:7] = csv_motion[:, 3:7][:, [1, 2, 3, 0]]
        np.savetxt(args.save_path, csv_motion, delimiter=",")

        motion_data = {
            "fps": motion_fps,
            "root_pos": root_pos,
            "root_rot": root_rot,
            "dof_pos": dof_pos,
            "local_body_pos": local_body_pos,
            "link_body_list": body_names,
        }
        # with open(args.save_path, "wb") as f:
        #     pickle.dump(motion_data, f)
        print(f"Saved to {args.save_path}")

    # Close progress bar
    pbar.close()

    robot_motion_viewer.close()
