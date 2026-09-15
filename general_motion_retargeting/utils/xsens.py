import numpy as np

from .xsens_vendor.BVHParser import (
    BVHParser,
    quat_fk,
    remove_quat_discontinuities,
)

# FootMod = ankle position + toe orientation (flat-foot IK target).
# Xsens BVH uses Ankle/Toe; Mixamo / FZ (e.g. dance_13.bvh) uses Foot/Toe.
_FOOTMOD_ANKLE_SUFFIXES = ("Ankle", "Foot")
_FOOTMOD_TOE_SUFFIXES = ("Toe", "ToeBase")


def synthesize_footmod(frame_data: dict) -> dict:
    """Add Left/RightFootMod when the source skeleton has an ankle and a toe.

    Mutates and returns ``frame_data``. No-op for a side that already has
    FootMod, or that is missing both an ankle and a toe joint.
    """
    for side in ("Left", "Right"):
        mod = f"{side}FootMod"
        if mod in frame_data:
            continue
        ankle_name = next(
            (f"{side}{sfx}" for sfx in _FOOTMOD_ANKLE_SUFFIXES
             if f"{side}{sfx}" in frame_data),
            None,
        )
        toe_name = next(
            (f"{side}{sfx}" for sfx in _FOOTMOD_TOE_SUFFIXES
             if f"{side}{sfx}" in frame_data),
            None,
        )
        if ankle_name and toe_name:
            frame_data[mod] = [
                frame_data[ankle_name][0].copy(),
                frame_data[toe_name][1].copy(),
            ]
    return frame_data


def footmod_parent_edges(name_set) -> list[tuple[str, str]]:
    """Parent→FootMod edges for visualization (Ankle, else Foot)."""
    edges = []
    for side in ("Left", "Right"):
        ankle_name = next(
            (f"{side}{sfx}" for sfx in _FOOTMOD_ANKLE_SUFFIXES
             if f"{side}{sfx}" in name_set),
            None,
        )
        if ankle_name:
            edges.append((ankle_name, f"{side}FootMod"))
    return edges


def load_xsens_file(args):
    parser = BVHParser("zxy", args.scale)

    with open(args.bvh_file, "r") as f:
        rotations, positions = parser.parse(
            f.read(),
            start=args.start,
            end=args.end,
            reset_to_zero=args.reset_to_zero,
        )

    quats = parser.rotations_to_quats(rotations)
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

        synthesize_footmod(result)
        frames.append(result)

    first_frame = frames[0]
    foot_height = min(first_frame["LeftFootMod"][0][2], first_frame["RightFootMod"][0][2])
    actual_human_height = first_frame["Head"][0][2] - foot_height

    return frames, actual_human_height, parser.frame_time
