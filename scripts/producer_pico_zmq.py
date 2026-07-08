from __future__ import annotations

import logging
import time

import msgpack
import numpy as np
import zmq

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting import XRobotStreamer

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

    teleop_data_streamer = XRobotStreamer()

    print("[Producer] Initializing GMR...")
    retarget = GMR(
        src_human="xrobot",
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

    frame_id = 0
    last_stream_timestamp = None
    warmup_count = 0
    loop_count = 0
    stream_new_count = 0
    pub_count = 0
    log_t0 = time.monotonic()

    print("[Producer] ZMQ publish @ new XRobot frame (after warmup).")
    print("[Producer] Running...")

    try:
        while True:
            loop_count += 1

            human_data, _, _, controller_data, _ = teleop_data_streamer.get_current_frame()
            # print(human_data)
            if human_data is None:
                time.sleep(0.001)
                continue

            stream_timestamp = None
            if controller_data is not None:
                stream_timestamp = controller_data.get("timestamp")

            if stream_timestamp is not None:
                has_new_frame = (last_stream_timestamp is None) or (
                    stream_timestamp != last_stream_timestamp
                )
                if has_new_frame:
                    last_stream_timestamp = stream_timestamp
            else:
                has_new_frame = True

            if has_new_frame:
                warmup_count += 1
                stream_new_count += 1

            now_mono = time.monotonic()
            if not has_new_frame:
                continue
            if warmup_count < warmup_frames:
                continue

            qpos = retarget.retarget(human_data)
            root_pos = qpos[:3].astype(np.float64)
            root_quat = qpos[3:7].astype(np.float64)  # wxyz
            joints = qpos[7:].astype(np.float64)
            num_dof = joints.shape[0]
            # assert num_dof == 29, f"Expected 29 DOF, got {num_dof}"

            if viewer is not None:
                viewer.step(
                    root_pos=qpos[:3],
                    root_rot=qpos[3:7],
                    dof_pos=qpos[7:],
                    human_motion_data=retarget.scaled_human_data,
                    rate_limit=False,
                )

            if stream_timestamp is not None:
                ts_s = float(stream_timestamp) * 1e-9
            else:
                ts_s = time.time()
            payload = _serialize_robot_ref(frame_id, ts_s, root_pos, root_quat, joints)
            zmq_sock.send(topic_bytes + b" " + payload)

            frame_id += 1
            pub_count += 1

            if log_every_s > 0 and now_mono - log_t0 >= log_every_s:
                elapsed = max(now_mono - log_t0, 1e-6)
                print(
                    "[ProducerRT] "
                    f"loop_hz={loop_count/elapsed:.1f}, "
                    f"stream_new_hz={stream_new_count/elapsed:.1f}, "
                    f"pub_hz={pub_count/elapsed:.1f}, "
                    f"warmup={warmup_count}/{warmup_frames}, frame_id={frame_id}"
                )
                loop_count = 0
                stream_new_count = 0
                pub_count = 0
                log_t0 = now_mono
    except KeyboardInterrupt:
        print(f"\n[Producer] Stopped. last_frame_id={frame_id}")
    finally:
        zmq_sock.close()
        zmq_ctx.term()


if __name__ == "__main__":
    main()
