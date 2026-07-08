"""
Replay recorded Xsens motion data to mock the live data stream.

Reads a .pkl file produced by record_xsens_data.py and replays it in two ways:
  1. Direct GMR retargeting — runs the full IK pipeline and visualizes
  2. Shared memory mock — injects frames back into shared memory so other
     consumers (e.g. producer_xsenstime_zmq.py) think Xsens is live streaming

Usage:
    # View retargeted motion on a robot
    python scripts/replay_xsens_data.py --record datas/xsens_record_20260707.pkl --robot d20_v2

    # Mock shared memory (run in background) for debugging downstream consumers
    python scripts/replay_xsens_data.py --record datas/xsens_record_20260707.pkl --mock_shm

    # Loop playback N times
    python scripts/replay_xsens_data.py --record datas/xsens_record_20260707.pkl --loop 3
"""

import argparse
import os
import pickle
import signal
import time

import numpy as np

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer

try:
    import general_motion_retargeting.xsense.xsens_shared_memory as xsens_shm
    HAS_XSENS_SHM = True
except ImportError:
    HAS_XSENS_SHM = False


def load_record(path: str) -> dict:
    """Load a recorded Xsens data file."""
    print(f"[Replay] Loading record from {path}")
    with open(path, "rb") as f:
        record = pickle.load(f)

    meta = record.get("meta", {})
    print(f"[Replay]   version={meta.get('version', '?')}, "
          f"frames={meta.get('frame_count', len(record.get('frames', [])))}, "
          f"created={meta.get('created', 'unknown')}")

    segment_id_map = meta.get("segment_id_map", {})
    if not segment_id_map:
        # Fallback to the known map
        segment_id_map = {
            0: "pelvis", 1: "L5", 2: "L3", 3: "T12", 4: "T8",
            5: "Neck", 6: "Head",
            7: "RightShoulder", 8: "RightUpperArm", 9: "RightForearm", 10: "RightHand",
            11: "LeftShoulder", 12: "LeftUpperArm", 13: "LeftForearm", 14: "LeftHand",
            15: "RightUpperLeg", 16: "RightLowerLeg", 17: "RightFoot", 18: "RightToe",
            19: "LeftUpperLeg", 20: "LeftLowerLeg", 21: "LeftFoot", 22: "LeftToe",
        }

    return record, segment_id_map


def frame_to_human_data(frame: dict, segment_id_map: dict) -> dict:
    """Convert a recorded frame into the GMR human_data dict format."""
    human_data = {}

    segments = frame.get("segments", {})
    # segment keys may be int or str — normalize
    for seg_id, name in segment_id_map.items():
        seg = segments.get(seg_id) or segments.get(str(seg_id))
        if seg is None:
            # Missing segment — use identity pose
            human_data[name] = [np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])]
        else:
            pos = np.array(seg["pos"], dtype=float)
            quat = np.array(seg["quat"], dtype=float)  # [qw, qx, qy, qz]
            human_data[name] = [pos, quat]

    # Synthesize foot modifications (ankle position + toe rotation)
    if "LeftFoot" in human_data and "LeftToe" in human_data:
        human_data["LeftFootMod"] = [human_data["LeftFoot"][0].copy(),
                                     human_data["LeftToe"][1].copy()]
    if "RightFoot" in human_data and "RightToe" in human_data:
        human_data["RightFootMod"] = [human_data["RightFoot"][0].copy(),
                                      human_data["RightToe"][1].copy()]

    return human_data


def mock_shared_memory(record: dict, segment_id_map: dict, speed: float = 1.0):
    """Push recorded frames into the Xsens shared memory to mock live streaming."""
    if not HAS_XSENS_SHM:
        raise RuntimeError("xsens_shared_memory module not available")

    xsens_manager = xsens_shm.XsensShareMemoryManager()
    frames = record["frames"]
    meta = record.get("meta", {})

    print(f"[ReplaySHM] Mocking shared memory with {len(frames)} frames at {speed}x speed")
    print(f"[ReplaySHM] Other consumers can now read from shared memory.")

    stop_requested = False

    def on_sigint(sig, frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, on_sigint)

    # Compute frame interval from first two frames
    if len(frames) >= 2:
        t0 = int(frames[0]["time_code"])
        t1 = int(frames[1]["time_code"])
        dt = (t1 - t0) / 1000.0  # time_code is in ms
        if dt <= 0 or dt > 0.1:
            dt = 0.01  # fallback 100Hz
    else:
        dt = 0.01

    loop = 0
    while not stop_requested:
        loop += 1
        print(f"[ReplaySHM] Loop {loop} / Ctrl+C to stop")

        for fi, frame in enumerate(frames):
            if stop_requested:
                break

            loop_start = time.monotonic()

            xd = xsens_manager.xsens_data.get_xsens_extracted_data

            # Set segment data first
            segments = frame.get("segments", {})

            # Set header fields
            xd.time_code = frame.get("time_code", 0)
            xd.items_num = frame.get("items_num", len(segments))
            xd.character_id = frame.get("character_id", 0)
            xd.body_segments_num = frame.get("body_segments_num", 23)
            xd.prop_segments_num = frame.get("prop_segments_num", 0)
            xd.finger_segments_num = frame.get("finger_segments_num", 0)
            xd.payload_size = frame.get("payload_size", 0)
            for seg_id, seg_data in segments.items():
                sid = int(seg_id)
                if sid >= 23:
                    continue
                seg = xd.body_segment[sid]
                seg.id = seg_data.get("id", sid)
                seg.pos.x = seg_data["pos"][0]
                seg.pos.y = seg_data["pos"][1]
                seg.pos.z = seg_data["pos"][2]
                seg.quat.qw = seg_data["quat"][0]
                seg.quat.qx = seg_data["quat"][1]
                seg.quat.qy = seg_data["quat"][2]
                seg.quat.qz = seg_data["quat"][3]

            # Copy to shared memory
            xsens_manager.copy_in_data()

            # Control replay speed
            elapsed = time.monotonic() - loop_start
            sleep_time = dt / speed - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    print(f"\n[ReplaySHM] Stopped after {loop} loop(s).")


def replay_to_gmr(record: dict, segment_id_map: dict, robot: str, speed: float = 1.0,
                  record_video: bool = False, video_path: str = None, loop: int = 1,
                  no_viewer: bool = False):
    """Replay recorded data through the GMR retargeting pipeline."""
    frames = record["frames"]
    meta = record.get("meta", {})

    if not frames:
        print("[Replay] No frames in record.")
        return

    print(f"[Replay] Initializing GMR (xsens_realtime -> {robot})...")
    retarget = GMR(
        src_human="xsens_realtime",
        tgt_robot=robot,
        actual_human_height=1.7,
    )

    if not no_viewer:
        viewer = RobotMotionViewer(
            robot_type=robot,
            record_video=record_video,
            video_path=video_path,
        )
    else:
        viewer = None

    stop_requested = False

    def on_sigint(sig, frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, on_sigint)

    # Compute frame interval
    if len(frames) >= 2:
        t0 = int(frames[0]["time_code"])
        t1 = int(frames[1]["time_code"])
        dt = max((t1 - t0) / 1000.0, 0.005)
    else:
        dt = 0.01
    fps = 1.0 / dt
    print(f"[Replay] Detected {fps:.0f} fps (dt={dt*1000:.1f}ms), replay at {speed}x")

    total_frames = 0
    for lap in range(loop):
        if stop_requested:
            break
        if loop > 1:
            print(f"[Replay] Loop {lap + 1}/{loop}")

        for fi, frame in enumerate(frames):
            if stop_requested:
                break

            t_start = time.monotonic()

            human_data = frame_to_human_data(frame, segment_id_map)
            qpos = retarget.retarget(human_data)

            if viewer is not None:
                viewer.step(
                    root_pos=qpos[:3],
                    root_rot=qpos[3:7],
                    dof_pos=qpos[7:],
                    human_motion_data=retarget.scaled_human_data,
                    rate_limit=False,
                )

            total_frames += 1

            # Print status periodically
            if fi % 100 == 0 and fi > 0:
                print(f"  [Replay] frame {fi}/{len(frames)}")

            # Frame rate control
            elapsed = time.monotonic() - t_start
            sleep_time = dt / speed - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    print(f"[Replay] Done. {total_frames} frames rendered.")

    if viewer is not None:
        viewer.close()


def main():
    parser = argparse.ArgumentParser(description="Replay recorded Xsens motion data")
    parser.add_argument("--record", required=True, type=str,
                        help="Path to .pkl file from record_xsens_data.py")

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--mock_shm", action="store_true",
                      help="Mock the shared memory (for debugging downstream consumers)")
    mode.add_argument("--robot", type=str, default=None,
                      help="Robot type for GMR retargeting (e.g. d20_v2, unitree_g1)")

    parser.add_argument("--speed", type=float, default=1.0,
                        help="Replay speed multiplier (default=1.0)")
    parser.add_argument("--loop", type=int, default=1,
                        help="Loop playback N times")
    parser.add_argument("--no_viewer", action="store_true",
                        help="Disable RobotMotionViewer")
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--video_path", type=str, default="videos/replay.mp4")

    args = parser.parse_args()

    record, segment_id_map = load_record(args.record)

    if args.mock_shm:
        mock_shared_memory(record, segment_id_map, speed=args.speed)
    elif args.robot:
        replay_to_gmr(
            record, segment_id_map,
            robot=args.robot,
            speed=args.speed,
            loop=args.loop,
            no_viewer=args.no_viewer,
            record_video=args.record_video,
            video_path=args.video_path,
        )
    else:
        print("[Replay] Please specify --robot <name> or --mock_shm")


if __name__ == "__main__":
    main()
