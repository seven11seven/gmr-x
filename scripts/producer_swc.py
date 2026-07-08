import time
import numpy as np
from scipy.spatial.transform import Rotation as R

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
import general_motion_retargeting.xsense.xsens_shared_memory as xsens_shm
from general_motion_retargeting.xsense.robot_ref_shared_memory import (
    RobotRefShareMemoryManager,
)


# -----------------------------------------------------------------------------
# 四元数工具：角速度由两帧四元数相对旋转计算，无欧拉、无 360° 跳变
# -----------------------------------------------------------------------------

def quat_inv_wxyz(q_wxyz):
    """q: (4,) wxyz"""
    q_wxyz = np.asarray(q_wxyz, dtype=np.float64)
    w, x, y, z = q_wxyz[0], q_wxyz[1], q_wxyz[2], q_wxyz[3]
    n2 = w * w + x * x + y * y + z * z
    return np.array([w, -x, -y, -z], dtype=np.float64) / n2


def quat_mul_wxyz(q1, q2):
    """(4,) wxyz each"""
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z], dtype=np.float64)


def angular_velocity_from_quat_pair(q_prev_wxyz, q_curr_wxyz, dt):
    """
    由两帧 body->world 四元数 (wxyz) 计算 body 系角速度 (3,) rad/s。
    无欧拉、无 gimbal lock、无 yaw 360° 跳变。
    """
    if dt <= 0.0:
        return np.zeros(3, dtype=np.float64)
    dq = quat_mul_wxyz(q_curr_wxyz, quat_inv_wxyz(q_prev_wxyz))
    w = float(dq[0])
    w = np.clip(w, -1.0, 1.0)
    theta = 2.0 * np.arccos(w)
    sin_half = np.sqrt(max(1.0 - w * w, 0.0))
    if sin_half < 1e-8:
        return np.zeros(3, dtype=np.float64)
    axis = np.array([dq[1], dq[2], dq[3]], dtype=np.float64) / sin_half
    omega_b = (theta / dt) * axis
    return omega_b


def get_gravity_orientation(quaternion_wxyz):
    """Body frame 下重力方向（与 sim2sim 一致）"""
    qw, qx, qy, qz = quaternion_wxyz[0], quaternion_wxyz[1], quaternion_wxyz[2], quaternion_wxyz[3]
    g = np.zeros(3, dtype=np.float64)
    g[0] = 2 * (-qz * qx + qw * qy)
    g[1] = -2 * (qz * qy + qw * qx)
    g[2] = 1 - 2 * (qw * qw + qz * qz)
    return g


# 和 read_xsens_shm.py 一致的 segment_id_map
segment_id_map = {
    0: "pelvis",
    1: "L5",
    2: "L3",
    3: "T12",
    4: "T8",
    5: "Neck",
    6: "Head",
    7: "RightShoulder",
    8: "RightUpperArm",
    9: "RightForearm",
    10: "RightHand",
    11: "LeftShoulder",
    12: "LeftUpperArm",
    13: "LeftForearm",
    14: "LeftHand",
    15: "RightUpperLeg",
    16: "RightLowerLeg",
    17: "RightFoot",
    18: "RightToe",
    19: "LeftUpperLeg",
    20: "LeftLowerLeg",
    21: "LeftFoot",
    22: "LeftToe",
}


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--xsens_freq",
        type=int,
        default=120,
        help="从 Xsens 读取频率 Hz",
    )
    parser.add_argument(
        "--policy_freq",
        type=int,
        default=50,
        help="写入共享内存的频率 Hz",
    )
    parser.add_argument(
        "--robot",
        type=str,
        default="unitree_g1",
        help="机器人类型",
    )
    parser.add_argument(
        "--vel_smooth_alpha",
        type=float,
        default=0.7,
        help="线速度一阶低通系数 (0~1)，越大越平滑；设为 0 关闭平滑",
    )
    parser.add_argument(
        "--no_viewer",
        action="store_true",
        help="不启动 RobotMotionViewer",
    )
    args = parser.parse_args()

    xsens_freq = args.xsens_freq
    policy_freq = args.policy_freq
    xsens_dt = 1.0 / xsens_freq
    policy_dt = 1.0 / policy_freq
    vel_smooth_alpha = max(0.0, min(1.0, args.vel_smooth_alpha))

    # 1. 附着 Xsens 原始共享内存
    xsens_manager = xsens_shm.XsensShareMemoryManager()

    # 2. 初始化 GMR
    print("[Producer] Initializing GMR...")
    retarget = GMR(
        src_human="xsens_realtime",
        tgt_robot=args.robot,
        actual_human_height=1.7,
    )

    if not args.no_viewer:
        viewer = RobotMotionViewer(robot_type=args.robot)
        # 查询 torso_link 的 body id（若 MJCF 中存在该 body）
        try:
            torso_body_id = viewer.model.body("torso_link").id
        except KeyError:
            torso_body_id = None
    else:
        viewer = None
        torso_body_id = None

    # 3. 初始化 RobotRef 共享内存
    print("[Producer] Initializing RobotRef shared memory...")
    robot_ref_mgr = RobotRefShareMemoryManager()

    # human_data 容器
    human_data = {}
    for seg_id in segment_id_map.keys():
        human_data[segment_id_map[seg_id]] = [np.zeros(3), np.zeros(4)]
    human_data["LeftFootMod"] = [np.zeros(3), np.zeros(4)]
    human_data["RightFootMod"] = [np.zeros(3), np.zeros(4)]

    # 速度估算：保留两帧位置用于中心差分
    pos_prev = None
    pos_prev2 = None
    q_prev = None
    joint_prev = None
    joint_prev2 = None
    # torso_link 位置 / 姿态历史（用于计算 torso_lin_vel_w / torso_ang_vel_w）
    torso_pos_prev = None
    torso_pos_prev2 = None
    torso_q_prev = None
    last_out_t = None
    v_lin_filt = None  # 线速度低通状态

    frame_id = 0

    print(f"[Producer] Xsens read @ {xsens_freq} Hz, RobotRef output @ {policy_freq} Hz.")
    if vel_smooth_alpha > 0:
        print(f"[Producer] Linear velocity low-pass alpha = {vel_smooth_alpha}")
    print("[Producer] Running...")

    while True:
        t0 = time.time()

        # 3.1 从 Xsens 共享内存读取一帧人体数据
        xsens_manager.copy_out_data()
        xd = xsens_manager.xsens_data.get_xsens_extracted_data

        for seg_id in segment_id_map.keys():
            seg = xd.body_segment[seg_id]
            t_pos = np.array([seg.pos.x, seg.pos.y, seg.pos.z], dtype=float)
            t_quat = np.array(
                [seg.quat.qw, seg.quat.qx, seg.quat.qy, seg.quat.qz], dtype=float
            )
            human_data[segment_id_map[seg_id]] = [t_pos, t_quat]

        human_data["LeftFootMod"] = [human_data["LeftFoot"][0], human_data["LeftToe"][1]]
        human_data["RightFootMod"] = [human_data["RightFoot"][0], human_data["RightToe"][1]]

        # 节流：保持 Xsens 读取频率
        elapsed = time.time() - t0
        time.sleep(max(0.0, xsens_dt - elapsed))

        now = time.time()
        # 仅当到达 policy 周期时才输出一帧（50Hz）
        if (last_out_t is not None) and (now - last_out_t < policy_dt):
            continue

        # 实际时间步长（用于速度分母）
        if last_out_t is None:
            dt = policy_dt
        else:
            dt = now - last_out_t
        dt = max(dt, 1e-6)

        # 3.2 GMR 重映射到机器人
        qpos = retarget.retarget(human_data,offset_to_ground=True)
        root_pos = qpos[:3].astype(np.float64)
        root_quat = qpos[3:7].astype(np.float64)  # wxyz
        joints = qpos[7:].astype(np.float64)
        num_dof = joints.shape[0]
        assert num_dof == 29, f"Expected 29 DOF, got {num_dof}"

        if viewer is not None:
            viewer.step(
                root_pos=qpos[:3],
                root_rot=qpos[3:7],
                dof_pos=qpos[7:],
                human_motion_data=retarget.scaled_human_data,
                rate_limit=False,
            )
            # 从 MuJoCo 读取 torso_link 的世界系四元数（wxyz）
            if torso_body_id is not None:
                torso_quat = viewer.data.xquat[torso_body_id].astype(np.float64)
                torso_pos = viewer.data.xpos[torso_body_id].astype(np.float64)
            else:
                torso_quat = root_quat.copy()
                torso_pos = root_pos.copy()
        else:
            # 无 viewer 时，无法从 MuJoCo 获得 torso_link 姿态，退化为 root_quat
            torso_quat = root_quat.copy()
            torso_pos = root_pos.copy()

        # ----- 3.3 线速度（世界系）：中心差分 + 可选低通 -----
        if pos_prev is None:
            root_lin_vel_w = np.zeros(3, dtype=np.float64)
        elif pos_prev2 is not None:
            # 中心差分：v(t) ≈ (x(t+1) - x(t-1)) / (2*dt)
            root_lin_vel_w = (root_pos - pos_prev2) / (2.0 * dt)
        else:
            root_lin_vel_w = (root_pos - pos_prev) / dt

        if vel_smooth_alpha > 0:
            if v_lin_filt is None:
                v_lin_filt = root_lin_vel_w.copy()
            else:
                v_lin_filt = vel_smooth_alpha * v_lin_filt + (1.0 - vel_smooth_alpha) * root_lin_vel_w
            root_lin_vel_w = v_lin_filt.copy()

        # ----- 3.4 角速度（世界系）：四元数相对旋转 -> body 角速度 -> 转世界系 -----
        if q_prev is None:
            root_ang_vel_w = np.zeros(3, dtype=np.float64)
        else:
            omega_b = angular_velocity_from_quat_pair(q_prev, root_quat, dt)
            # body -> world: omega_w = R_wb @ omega_b
            R_wb = R.from_quat(root_quat[[1, 2, 3, 0]]).as_matrix()
            root_ang_vel_w = (R_wb @ omega_b).astype(np.float64)

        # ----- 3.5 关节角速度：中心差分或前向差分 -----
        if joint_prev is None:
            joint_vel = np.zeros(num_dof, dtype=np.float64)
        elif joint_prev2 is not None:
            joint_vel = (joints - joint_prev2) / (2.0 * dt)
        else:
            joint_vel = (joints - joint_prev) / dt

        g_vec = get_gravity_orientation(root_quat)

        # ----- 3.6 torso_link 速度（世界系）：位置差分 + 四元数对 -----
        if torso_pos_prev is None:
            torso_lin_vel_w = np.zeros(3, dtype=np.float64)
        elif torso_pos_prev2 is not None:
            torso_lin_vel_w = (torso_pos - torso_pos_prev2) / (2.0 * dt)
        else:
            torso_lin_vel_w = (torso_pos - torso_pos_prev) / dt

        if torso_q_prev is None:
            torso_ang_vel_w = np.zeros(3, dtype=np.float64)
        else:
            torso_omega_b = angular_velocity_from_quat_pair(torso_q_prev, torso_quat, dt)
            R_wt = R.from_quat(torso_quat[[1, 2, 3, 0]]).as_matrix()
            torso_ang_vel_w = (R_wt @ torso_omega_b).astype(np.float64)

        # 更新历史（为下一帧 50Hz 输出准备）
        pos_prev2 = pos_prev
        pos_prev = root_pos.copy()
        q_prev = root_quat.copy()
        joint_prev2 = joint_prev
        joint_prev = joints.copy()

        torso_pos_prev2 = torso_pos_prev
        torso_pos_prev = torso_pos.copy()
        torso_q_prev = torso_quat.copy()

        last_out_t = now

        root_quat_xyzw = root_quat[[1, 2, 3, 0]]
        R_wb = R.from_quat(root_quat_xyzw).as_matrix()
        R_bw = R_wb.T

        root_lin_vel_b = R_bw @ root_lin_vel_w
        root_ang_vel_b = R_bw @ root_ang_vel_w

        # torso 的 body-frame 速度（如需要在 consumer 中用 torso_vel_b）
        torso_quat_xyzw = torso_quat[[1, 2, 3, 0]]
        R_wt = R.from_quat(torso_quat_xyzw).as_matrix()
        R_tw = R_wt.T
        torso_lin_vel_b = R_tw @ torso_lin_vel_w
        torso_ang_vel_b = R_tw @ torso_ang_vel_w

        # 3.7 写入 RobotRef 共享内存（速度为 body 系，另附 torso_link 姿态）
        d = robot_ref_mgr.local_data.data
        d.frame_id = frame_id
        for i in range(3):
            d.root_pos[i] = float(torso_pos[i])
            d.root_lin_vel[i] = float(torso_lin_vel_b[i])
            d.root_ang_vel[i] = float(torso_ang_vel_b[i])
            d.gravity[i] = float(g_vec[i])
        for i in range(4):
            d.root_quat[i] = float(torso_quat[i])
        for i in range(num_dof):
            d.joint_pos[i] = float(joints[i])
            d.joint_vel[i] = float(joint_vel[i])

        robot_ref_mgr.copy_in_data()
        frame_id += 1


if __name__ == "__main__":
    main()
