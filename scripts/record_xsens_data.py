"""
Record Xsens motion data from shared memory to local file.

Reads from the Xsens shared memory (populated by extract_xsens_data.py)
and saves all segment data frame-by-frame for offline debugging/replay.

Usage:
    python scripts/record_xsens_data.py --duration 10
    python scripts/record_xsens_data.py --duration 0  # record until Ctrl+C
"""

import argparse
import os
import pickle
import signal
import time
from datetime import datetime

import numpy as np

import general_motion_retargeting.xsense.xsens_shared_memory as xsens_shm

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
MAX_PROP_SEGMENT_NUM = 4
MAX_FINGER_SEGMENTS_NUM = 40
JOINT_NUM = 28


def extract_segment(seg):
    """Extract pos and quat from a SEGMENT ctypes struct into plain lists."""
    pos = [float(seg.pos.x), float(seg.pos.y), float(seg.pos.z)]
    quat = [float(seg.quat.qw), float(seg.quat.qx), float(seg.quat.qy), float(seg.quat.qz)]
    return {"id": int(seg.id), "pos": pos, "quat": quat}


def extract_joint(j):
    """Extract joint data from a JOINT ctypes struct."""
    return {
        "parent_segment_id": int(j.parent_segment_id),
        "child_segment_id": int(j.child_segment_id),
        "rx": float(j.pos.rx),
        "ry": float(j.pos.ry),
        "rz": float(j.pos.rz),
    }


def _safe_get_str(xd, field):
    """Read a c_char_p field from shared memory safely.
    c_char_p in shared memory is a raw pointer value from the writer's address space —
    dereferencing it in this process will segfault. We skip it entirely."""
    return ""


def snapshot(xd) -> dict:
    """Capture the full Xsens shared memory state as a plain dict."""
    # NOTE: xd.id_str is a c_char_p pointer — its value comes from the
    # extract_xsens_data.py process, so dereferencing it here would segfault.
    # We hardcode it since it's always "MXTP02".
    frame = {
        "id_str": "MXTP02",
        "items_num": int(xd.items_num),
        "time_code": int(xd.time_code),
        "character_id": int(xd.character_id),
        "body_segments_num": int(xd.body_segments_num),
        "prop_segments_num": int(xd.prop_segments_num),
        "finger_segments_num": int(xd.finger_segments_num),
        "payload_size": int(xd.payload_size),
        "segments": {},
        "props": [],
        "fingers": [],
        "joints": [],
    }

    for seg_id in SEGMENT_ID_MAP:
        seg = xd.body_segment[seg_id]
        frame["segments"][seg_id] = extract_segment(seg)

    for i in range(int(xd.prop_segments_num)):
        frame["props"].append(extract_segment(xd.prop_segment[i]))

    for i in range(int(xd.finger_segments_num)):
        frame["fingers"].append(extract_segment(xd.finger_segment[i]))

    for i in range(JOINT_NUM):
        frame["joints"].append(extract_joint(xd.joint[i]))

    return frame


def main():
    parser = argparse.ArgumentParser(description="Record Xsens data from shared memory")
    parser.add_argument("--duration", type=float, default=0,
                        help="Recording duration in seconds (0 = record until Ctrl+C)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path (default: datas/xsens_record_<timestamp>.pkl)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    # Determine output path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(script_dir, "..", "datas", f"xsens_record_{timestamp}.pkl")
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Connect to shared memory
    print("[Record] Connecting to Xsens shared memory...")
    xsens_manager = xsens_shm.XsensShareMemoryManager()

    frames = []
    start_time = time.monotonic()
    last_time_code = None
    frame_count = 0
    last_print = start_time

    stop_requested = False

    def on_sigint(sig, frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, on_sigint)

    print(f"[Record] Recording to {output_path}")
    print(f"[Record] Press Ctrl+C to stop.\n")

    try:
        while not stop_requested:
            xsens_manager.copy_out_data()
            xd = xsens_manager.xsens_data.get_xsens_extracted_data

            # Skip if shared memory is uninitialized (no data from extract_xsens_data.py yet)
            if int(xd.items_num) == 0:
                time.sleep(0.01)
                continue

            curr_time_code = int(xd.time_code)

            # Only save when a new frame arrives (time_code changes)
            if curr_time_code != last_time_code:
                last_time_code = curr_time_code
                frame = snapshot(xd)
                frames.append(frame)
                frame_count += 1

            # Check duration limit
            elapsed = time.monotonic() - start_time
            if args.duration > 0 and elapsed >= args.duration:
                break

            # Status print
            if not args.quiet and time.monotonic() - last_print >= 2.0:
                print(f"  [Record] frames={frame_count}, elapsed={elapsed:.1f}s, "
                      f"fps={frame_count / max(elapsed, 1e-6):.1f}")
                last_print = time.monotonic()

            time.sleep(0.001)

    except KeyboardInterrupt:
        pass

    elapsed = time.monotonic() - start_time
    print(f"\n[Record] Stopped. {frame_count} frames in {elapsed:.1f}s "
          f"({frame_count / max(elapsed, 1e-6):.1f} fps)")

    if not frames:
        print("[Record] No frames captured, exiting.")
        return

    # Pack into a self-describing record
    record = {
        "meta": {
            "version": 1,
            "created": datetime.now().isoformat(),
            "frame_count": len(frames),
            "segment_id_map": SEGMENT_ID_MAP,
            "body_segments_num": BODY_SEGMENTS_NUM,
            "prop_segments_max": MAX_PROP_SEGMENT_NUM,
            "finger_segments_max": MAX_FINGER_SEGMENTS_NUM,
            "joint_num": JOINT_NUM,
        },
        "frames": frames,
    }

    with open(output_path, "wb") as f:
        pickle.dump(record, f)

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[Record] Saved to {output_path} ({file_size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
