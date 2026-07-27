import numpy as np

from .xsens_vendor.BVHParser import (
    BVHParser,
    euler_to_quat,
    quat_fk,
    remove_quat_discontinuities,
)


def load_xsens_file(args):
    parser = BVHParser("zxy", args.scale)

    with open(args.bvh_file, "r") as f:
        rotations, positions = parser.parse(
            f.read(),
            start=args.start,
            end=args.end,
            reset_to_zero=args.reset_to_zero,
        )

    quats = np.array([[euler_to_quat(rot) for rot in frame] for frame in rotations])
    quats = remove_quat_discontinuities(quats)
    global_quats, global_positions = quat_fk(
        quats,
        positions,
        np.array(parser.parents, dtype=int),
    )

    frames = []
    for frame_idx in range(global_positions.shape[0]):
        result = {}
        for joint_idx, name in enumerate(parser.names):
            if name.endswith("_end_site"):
                continue
            result[name] = [
                global_positions[frame_idx, joint_idx].copy(),
                global_quats[frame_idx, joint_idx].copy(),
            ]

        result["LeftFootMod"] = [
            result["LeftAnkle"][0].copy(),
            result["LeftToe"][1].copy(),
        ]
        result["RightFootMod"] = [
            result["RightAnkle"][0].copy(),
            result["RightToe"][1].copy(),
        ]
        frames.append(result)

    first_frame = frames[0]
    foot_height = min(first_frame["LeftFootMod"][0][2], first_frame["RightFootMod"][0][2])
    actual_human_height = first_frame["Head"][0][2] - foot_height

    return frames, actual_human_height, parser.frame_time
