from __future__ import annotations

import logging
import time

import msgpack
import numpy as np
import zmq

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
import general_motion_retargeting.xsense.xsens_shared_memory as xsens_shm

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

logger = logging.getLogger(__name__)


def _serialize_robot_ref(
    frame_id: int,
    timestamp_s: float,
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    joint_pos: np.ndarray,
) -> bytes:
    """Pack robot reference (root + joints) like zmq_pico4_publisher msgpack style."""
    return msgpack.packb(
        {
            "_seq": int(frame_id),
            "_ts": float(timestamp_s),
            "root_pos": root_pos.astype(np.float64).tolist(),
            "root_quat": root_quat.astype(np.float64).tolist(),
            "joint_pos": joint_pos.astype(np.float64).tolist(),
        },
        use_bin_type=True,
    )


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default="unitree_g1", help="机器人类型")
    parser.add_argument("--no_viewer", action="store_true", help="不启动 RobotMotionViewer")
    parser.add_argument("--warmup_frames", type=int, default=5, help="启动预热帧数，达到后再开始发布参考")
    parser.add_argument("--log_every_s", type=float, default=1.0, help="实时状态打印周期（秒），<=0 关闭")
    parser.add_argument("--bind", default="0.0.0.0", help="ZMQ PUB bind 地址（默认 0.0.0.0）")
    parser.add_argument("--port", type=int, default=5555, help="ZMQ PUB 端口（默认 5555）")
    parser.add_argument("--topic", default="xsens_gmr", help="ZMQ topic 前缀（默认 xsens_g1）")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

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

    zmq_ctx = zmq.Context()
    zmq_sock = zmq_ctx.socket(zmq.PUB)
    zmq_sock.setsockopt(zmq.CONFLATE, 1)  # 只保留最新一帧待发，避免发送侧堆积阻塞
    zmq_sock.bind(f"tcp://{args.bind}:{args.port}")
    topic_bytes = args.topic.encode("utf-8")
    logger.info(
        "ZMQ PUB bound on tcp://%s:%d topic=%s",
        args.bind,
        args.port,
        args.topic,
    )
    print(
        f"[Producer] ZMQ PUB tcp://{args.bind}:{args.port} topic={args.topic!r} "
        "(msgpack: _seq, _ts, root_pos, root_quat, joint_pos)"
    )
    time.sleep(0.15)  # slow-joiner: give SUB time to connect

    human_data = {}
    for seg_id in segment_id_map.keys():
        human_data[segment_id_map[seg_id]] = [np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])]
    human_data["LeftFootMod"] = [np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])]
    human_data["RightFootMod"] = [np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])]

    frame_id = 0
    last_xsens_time_code = None
    xsens_time_wrap_count = 0
    xsens_time_prev_raw = None
    warmup_count = 0
    loop_count = 0
    xsens_new_count = 0
    pub_count = 0
    log_t0 = time.monotonic()

    print("[Producer] ZMQ publish @ new Xsens frame (after warmup).")
    print("[Producer] Running...")

    try:
        while True:
            loop_count += 1
            xsens_manager.copy_out_data()
            xd = xsens_manager.xsens_data.get_xsens_extracted_data
            curr_time_code = int(xd.time_code)
            # has_new_xsens_frame = True
            has_new_xsens_frame = (last_xsens_time_code is None) or (curr_time_code != last_xsens_time_code)
            if has_new_xsens_frame:
                last_xsens_time_code = curr_time_code
                warmup_count += 1
                xsens_new_count += 1
                if xsens_time_prev_raw is not None and curr_time_code < xsens_time_prev_raw:
                    xsens_time_wrap_count += 1
                xsens_time_prev_raw = curr_time_code
            
            #if not has_new_xsens_frame:
             #   time.sleep(0.001)
              #  continue

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
            #assert num_dof == 29, f"Expected 29 DOF, got {num_dof}"
            # print(f"qpos: {qpos}")

            if viewer is not None:
                viewer.step(
                    root_pos=qpos[:3],
                    root_rot=qpos[3:7],
                    dof_pos=qpos[7:],
                    human_motion_data=retarget.scaled_human_data,
                    rate_limit=False,
                )

            xsens_time_ms = float(curr_time_code + xsens_time_wrap_count * (2**32))
            ts_s = xsens_time_ms * 1e-3
            payload = _serialize_robot_ref(frame_id, ts_s, root_pos, root_quat, joints)
            zmq_sock.send(topic_bytes + b" " + payload)

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
    except KeyboardInterrupt:
        print(f"\n[Producer] Stopped. last_frame_id={frame_id}")
    finally:
        zmq_sock.close()
        zmq_ctx.term()


if __name__ == "__main__":
    main()
