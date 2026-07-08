"""
Offline Xsens motion debug tool — load a recorded .pkl and drive GMR.

What this shows in the viewer:
  - Robot skeleton: retargeted qpos (IK result)
  - Human reference: the *scaled* human data after GMR's internal transforms
    (scale → offset → ground compensation), NOT the raw Xsens data.

Usage:
    python scripts/debug_xsens_replay.py --record datas/xsens_record_20260707.pkl --robot d20_v2
    python scripts/debug_xsens_replay.py --record datas/xsens_record_20260707.pkl --robot unitree_g1 --loop 2
    python scripts/debug_xsens_replay.py --record datas/xsens_record_20260707.pkl --robot d20_v2 --dump_first_frame
"""

import argparse
import os
import pickle
import signal
import sys
import time

import numpy as np

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer

# ── Xsens segment id → human body name (same as producer_xsenstime_zmq.py) ──

SEGMENT_ID_MAP = {
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

BODY_SEGMENTS_NUM = 23


def load_record(path: str) -> tuple:
    """Return (frames_list, segment_id_map) from a recording .pkl."""
    path = os.path.abspath(path)
    print(f"[Load] {path}")
    with open(path, "rb") as f:
        record = pickle.load(f)

    meta = record.get("meta", {})
    frames = record["frames"]
    seg_map = meta.get("segment_id_map", SEGMENT_ID_MAP)

    if len(frames) >= 2:
        dt_ms = int(frames[1]["time_code"]) - int(frames[0]["time_code"])
        fps = 1000.0 / dt_ms if dt_ms > 0 else 60.0
    else:
        fps = 60.0
    print(f"[Load] {len(frames)} frames, ~{fps:.0f} fps, "
          f"created={meta.get('created', 'unknown')}")
    return frames, seg_map, fps


def frame_to_human_data(frame: dict, seg_map: dict) -> dict:
    """Build the raw human_data dict from a recorded frame.

    human_data = {"Pelvis": (pos_xyz, quat_wxyz), ...}

    Also synthesises LeftFootMod / RightFootMod (ankle pos + toe rotation)
    matching the convention in producer_xsenstime_zmq.py.
    """
    human_data = {}
    segments = frame["segments"]

    for seg_id, name in seg_map.items():
        seg = segments.get(seg_id) or segments.get(str(seg_id))
        if seg is None:
            human_data[name] = [np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])]
        else:
            human_data[name] = [
                np.array(seg["pos"], dtype=float),
                np.array(seg["quat"], dtype=float),  # [w, x, y, z]
            ]

    # Foot mods — same pattern as producer_xsenstime_zmq.py line 160-161
    if "LeftFoot" in human_data and "LeftToe" in human_data:
        human_data["LeftFootMod"] = [
            human_data["LeftFoot"][0].copy(),
            human_data["LeftToe"][1].copy(),
        ]
    if "RightFoot" in human_data and "RightToe" in human_data:
        human_data["RightFootMod"] = [
            human_data["RightFoot"][0].copy(),
            human_data["RightToe"][1].copy(),
        ]

    return human_data


def dump_first_frame(frames, seg_map):
    """Print the raw & scaled data for the first frame — useful to verify offsets."""
    raw = frame_to_human_data(frames[0], seg_map)

    print("\n" + "=" * 70)
    print("RAW HUMAN DATA (frame 0) — before GMR transforms")
    print("=" * 70)
    for name in sorted(raw):
        p, q = raw[name]
        print(f"  {name:20s}  pos=({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f})  "
              f"quat=({q[0]:+.3f}, {q[1]:+.3f}, {q[2]:+.3f}, {q[3]:+.3f})")


def main():
    parser = argparse.ArgumentParser(
        description="Offline Xsens replay → GMR debug tool"
    )
    parser.add_argument("--record", required=True, help="Path to .pkl recording")
    parser.add_argument("--robot", default="d20_v2",
                        choices=["d20_v2", "unitree_g1"],
                        help="Target robot (default: d20_v2)")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Replay speed multiplier")
    parser.add_argument("--loop", type=int, default=1,
                        help="Loop playback N times")
    parser.add_argument("--no_viewer", action="store_true",
                        help="Run IK only, skip MuJoCo viewer")
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--video_path", default="videos/debug_replay.mp4")
    parser.add_argument("--dump_first_frame", action="store_true",
                        help="Print raw human data of the first frame and exit")
    parser.add_argument("--start", type=int, default=0,
                        help="Start frame index")
    parser.add_argument("--end", type=int, default=None,
                        help="End frame index (exclusive)")
    parser.add_argument("--pause_at", type=int, default=10,
                        help="Pause at this frame index and wait for Enter (0 to disable)")
    args = parser.parse_args()

    # ── 1. Load recorded data ──
    frames, seg_map, fps = load_record(args.record)

    if args.end is not None:
        frames = frames[args.start : args.end]
    elif args.start > 0:
        frames = frames[args.start:]

    if args.dump_first_frame:
        dump_first_frame(frames, seg_map)
        return

    # ── 2. Init GMR (uses xsens_realtime → <robot> IK config) ──
    print(f"\n[GMR] src_human=xsens_realtime  →  tgt_robot={args.robot}")
    retarget = GMR(
        src_human="xsens_realtime",
        tgt_robot=args.robot,
        actual_human_height=1.7,
        verbose=False,  # quiet — avoid printing all DoF names
    )

    # ── 3. Init viewer ──
    if not args.no_viewer:
        viewer = RobotMotionViewer(
            robot_type=args.robot,
            transparent_robot=1,
            motion_fps=int(fps),
            record_video=args.record_video,
            video_path=args.video_path,
        )
    else:
        viewer = None

    # ── 4. Main replay loop ──
    dt = 1.0 / fps
    print(f"[Run] {len(frames)} frames, dt={dt*1000:.1f}ms, speed={args.speed}x, "
          f"loop={args.loop}")
    print(f"[Run] Viewer shows: scaled human reference + retargeted robot.\n")

    stop_req = False

    def _sigint(sig, frame):
        nonlocal stop_req
        stop_req = True
    signal.signal(signal.SIGINT, _sigint)

    frame_id = 0
    for lap in range(args.loop):
        if stop_req:
            break
        if args.loop > 1:
            print(f"[Run] loop {lap + 1}/{args.loop}")

        for fi, frame in enumerate(frames):
            if stop_req:
                break

            t0 = time.monotonic()

            # 2a. Convert recording → human_data dict
            human_data = frame_to_human_data(frame, seg_map)

            # 2b. Run retargeting (this calls update_targets internally:
            #     scale_human_data → offset_human_data → offset_to_ground)
            qpos = retarget.retarget(human_data)
            t_ik = time.monotonic() - t0

            # 2c. retarget.scaled_human_data now holds the fully-transformed
            #     human reference (scaled + IK-offset + grounded).
            #     This is what the viewer renders as the translucent human.
            if viewer is not None:
                viewer.step(
                    root_pos=qpos[:3] + [0.0, 0.0, 0.0],  # offset to avoid clipping with robot
                    root_rot=qpos[3:7],
                    dof_pos=qpos[7:],
                    human_motion_data=retarget.scaled_human_data,
                    rate_limit=False,
                )
            t_render = time.monotonic() - t0

            frame_id += 1

            # Pause at specified frame for manual inspection
            if args.pause_at > 0 and fi == args.pause_at:
                print(f"\n  ⏸  Paused at frame {fi}. Press Enter to continue...")
                input()

            if fi % 100 == 0 and fi > 0:
                print(f"  frame {fi}/{len(frames)}")

            # FPS control
            elapsed = time.monotonic() - t0
            sleep_t = dt / args.speed - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

            if fi % 100 == 0:
                print(f"  [timing] IK={t_ik*1000:.2f}ms  render={(t_render-t_ik)*1000:.2f}ms  total={t_render*1000:.2f}ms")

    print(f"[Run] Done. {frame_id} total frames rendered.")

    if viewer is not None:
        viewer.close()


if __name__ == "__main__":
    main()
