import time
import numpy as np

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
import general_motion_retargeting.xsense.xsens_shared_memory as xsens_shm
from general_motion_retargeting.xsense.robot_ref_shared_memory import RobotRefShareMemoryManager


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
    parser.add_argument("--robot", type=str, default="unitree_g1", help="机器人类型")
    parser.add_argument("--no_viewer", action="store_true", help="不启动 RobotMotionViewer")
    parser.add_argument("--warmup_frames", type=int, default=5, help="启动预热帧数，达到后再开始发布参考")
    parser.add_argument("--log_every_s", type=float, default=1.0, help="实时状态打印周期（秒），<=0 关闭")
    args = parser.parse_args()

    warmup_frames = max(1, args.warmup_frames)
    log_every_s = float(args.log_every_s)

    xsens_manager = xsens_shm.XsensShareMemoryManager()

    print("[Producer] Initializing GMR...")
    retarget = GMR(
        src_human="xsens_realtime",
        tgt_robot=args.robot,
        actual_human_height=1.7,
    )

    if not args.no_viewer:
        viewer = RobotMotionViewer(robot_type=args.robot)
    else:
        viewer = None

    print("[Producer] Initializing RobotRef shared memory...")
    robot_ref_mgr = RobotRefShareMemoryManager()

    human_data = {}
    for seg_id in segment_id_map.keys():
        human_data[segment_id_map[seg_id]] = [np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])]
    human_data["LeftFootMod"] = [np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])]
    human_data["RightFootMod"] = [np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])]

    frame_id = 0
    last_xsens_time_code = None
    warmup_count = 0
    loop_count = 0
    xsens_new_count = 0
    pub_count = 0
    log_t0 = time.monotonic()

    print("[Producer] RobotRef output @ full speed (new frame driven).")
    print("[Producer] Running...")

    while True:
        loop_count += 1
        xsens_manager.copy_out_data()
        xd = xsens_manager.xsens_data.get_xsens_extracted_data
        curr_time_code = int(xd.time_code)
        has_new_xsens_frame = (last_xsens_time_code is None) or (curr_time_code != last_xsens_time_code)
        if has_new_xsens_frame:
            last_xsens_time_code = curr_time_code
            warmup_count += 1
            xsens_new_count += 1

        for seg_id in segment_id_map.keys():
            seg = xd.body_segment[seg_id]
            t_pos = np.array([seg.pos.x, seg.pos.y, seg.pos.z], dtype=float)
            t_quat = np.array(
                [seg.quat.qw, seg.quat.qx, seg.quat.qy, seg.quat.qz], dtype=float
            )
            human_data[segment_id_map[seg_id]] = [t_pos, t_quat]

        human_data["LeftFootMod"] = [human_data["LeftFoot"][0], human_data["LeftToe"][1]]
        human_data["RightFootMod"] = [human_data["RightFoot"][0], human_data["RightToe"][1]]

        now_mono = time.monotonic()
        if not has_new_xsens_frame:
            continue
        if warmup_count < warmup_frames:
            continue

        qpos = retarget.retarget(human_data)
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

        d = robot_ref_mgr.local_data.data
        d.frame_id = frame_id
        d.timestamp = now_mono
        for i in range(3):
            d.root_pos[i] = float(root_pos[i])
        for i in range(4):
            d.root_quat[i] = float(root_quat[i])
        for i in range(num_dof):
            d.joint_pos[i] = float(joints[i])

        robot_ref_mgr.copy_in_data()
        frame_id += 1
        pub_count += 1

        if log_every_s > 0 and now_mono - log_t0 >= log_every_s:
            elapsed = max(now_mono - log_t0, 1e-6)
            print(
                "[ProducerRT] "
                f"loop_hz={loop_count/elapsed:.1f}, "
                f"xsens_new_hz={xsens_new_count/elapsed:.1f}, "
                f"pub_hz={pub_count/elapsed:.1f}, "
                f"warmup={warmup_count}/{warmup_frames}, frame_id={frame_id}"
            )
            loop_count = 0
            xsens_new_count = 0
            pub_count = 0
            log_t0 = now_mono


if __name__ == "__main__":
    main()
