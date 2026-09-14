"""Export 1-1 SMPL-X ↔ robot slot pairs for contact inference.

The correspondence AE uses a shared slot index: slot ``i`` on the SMPL-X
template is the same semantic surface location as slot ``i`` on the robot
reconstruction. This script binds those paired points back onto each mesh
(face + barycentric, and robot geom/body when the dataset stored them) so a
later contact model can skin / FK the same candidates.

Typical usage after training::

    python scripts/export_correspondence_contact.py \\
        --slots output/correspondence_template_residual_ae/correspondence_slots_final.npz \\
        --data data/correspondence_ae_tpose_4096.npz \\
        --source smplx_neutral --robot d20
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from train_correspondence_template_residual_ae import is_robot_sample_name
except ImportError:
    def is_robot_sample_name(name):
        name = str(name)
        body_prefixes = ("smpl", "smplx", "smplh")
        return not any(name == prefix or name.startswith(f"{prefix}_") for prefix in body_prefixes)


def bind_points_to_mesh(points, vertices, faces):
    """Closest-triangle bind. Returns face ids, barycentric weights, normals."""
    import trimesh

    points = np.asarray(points, dtype=np.float64)
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    closest, distances, face_ids = trimesh.proximity.closest_point(mesh, points)
    face_ids = np.asarray(face_ids, dtype=np.int32)
    tris = vertices[faces[face_ids]]
    bary = trimesh.triangles.points_to_barycentric(tris, closest)
    bary = np.clip(np.asarray(bary, dtype=np.float64), 0.0, 1.0)
    bary = bary / np.maximum(bary.sum(axis=1, keepdims=True), 1e-8)
    e1 = tris[:, 1] - tris[:, 0]
    e2 = tris[:, 2] - tris[:, 0]
    normals = np.cross(e1, e2)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    return {
        "face_ids": face_ids,
        "bary": bary.astype(np.float32),
        "closest_points": np.asarray(closest, dtype=np.float32),
        "errors": np.asarray(distances, dtype=np.float32),
        "closest_normals": normals.astype(np.float32),
        "vertex_ids": faces[face_ids].astype(np.int32),
    }


def _load_mesh_bundle(data, idx):
    vertices = data[f"mesh_vertices_{idx}"].astype(np.float32) if f"mesh_vertices_{idx}" in data else None
    faces = data[f"mesh_faces_{idx}"].astype(np.int32) if f"mesh_faces_{idx}" in data else None
    sample_face_ids = (
        data[f"sample_face_ids_{idx}"].astype(np.int32) if f"sample_face_ids_{idx}" in data else None
    )
    face_geom_ids = (
        data[f"face_geom_ids_{idx}"].astype(np.int32) if f"face_geom_ids_{idx}" in data else None
    )
    face_body_ids = (
        data[f"face_body_ids_{idx}"].astype(np.int32) if f"face_body_ids_{idx}" in data else None
    )
    geom_names = data[f"geom_names_{idx}"] if f"geom_names_{idx}" in data else None
    body_names = data[f"body_names_{idx}"] if f"body_names_{idx}" in data else None
    return {
        "vertices": vertices,
        "faces": faces,
        "sample_face_ids": sample_face_ids,
        "face_geom_ids": face_geom_ids,
        "face_body_ids": face_body_ids,
        "geom_names": None if geom_names is None else np.asarray(geom_names).astype(str),
        "body_names": None if body_names is None else np.asarray(body_names).astype(str),
    }


def _lookup_names(ids, table):
    ids = np.asarray(ids, dtype=np.int32)
    names = np.full(ids.shape, "", dtype=object)
    if table is None:
        return names.astype(str)
    table = np.asarray(table).astype(str)
    valid = (ids >= 0) & (ids < len(table))
    names[valid] = table[ids[valid]]
    return names.astype(str)


def resolve_pair_names(names, source_name, robot_name, template_name=None):
    names = [str(n) for n in names]
    if source_name:
        source_name = str(source_name)
        if source_name not in names:
            raise ValueError(f"source {source_name!r} not in {names}")
    else:
        source_name = str(template_name) if template_name in names else None
        if source_name is None:
            smplx = [n for n in names if n.startswith("smplx")]
            source_name = smplx[0] if smplx else names[0]
    if robot_name:
        robot_name = str(robot_name)
        if robot_name not in names:
            raise ValueError(f"robot {robot_name!r} not in {names}")
    else:
        robots = [n for n in names if is_robot_sample_name(n)]
        if not robots:
            raise ValueError(f"no robot sample in {names}; pass --robot")
        robot_name = robots[0]
        if "d20" in robots:
            robot_name = "d20"
    return source_name, robot_name


def build_contact_pairs(
    slots,
    dataset,
    source_name,
    robot_name,
):
    names = [str(n) for n in np.asarray(slots["names"]).tolist()]
    source_idx = names.index(source_name)
    robot_idx = names.index(robot_name)
    source_points = np.asarray(slots["reconstructed_slots"][source_idx], dtype=np.float32)
    robot_points = np.asarray(slots["reconstructed_slots"][robot_idx], dtype=np.float32)
    residual = (
        np.asarray(slots["residual_slots"][robot_idx], dtype=np.float32)
        if "residual_slots" in slots
        else np.zeros_like(robot_points)
    )
    slot_index = np.arange(len(source_points), dtype=np.int32)

    source_mesh = _load_mesh_bundle(dataset, source_idx)
    robot_mesh = _load_mesh_bundle(dataset, robot_idx)
    if source_mesh["vertices"] is None or source_mesh["faces"] is None:
        raise ValueError(f"dataset is missing mesh metadata for {source_name}")
    if robot_mesh["vertices"] is None or robot_mesh["faces"] is None:
        raise ValueError(f"dataset is missing mesh metadata for {robot_name}")

    source_bind = bind_points_to_mesh(source_points, source_mesh["vertices"], source_mesh["faces"])
    robot_bind = bind_points_to_mesh(robot_points, robot_mesh["vertices"], robot_mesh["faces"])

    n_faces_src = len(source_mesh["faces"])
    n_faces_robot = len(robot_mesh["faces"])
    source_face_ids = source_bind["face_ids"]
    robot_face_ids = robot_bind["face_ids"]

    robot_geom_ids = np.full(len(slot_index), -1, dtype=np.int32)
    robot_body_ids = np.full(len(slot_index), -1, dtype=np.int32)
    if robot_mesh["face_geom_ids"] is not None:
        valid = (robot_face_ids >= 0) & (robot_face_ids < len(robot_mesh["face_geom_ids"]))
        robot_geom_ids[valid] = robot_mesh["face_geom_ids"][robot_face_ids[valid]]
        robot_body_ids[valid] = robot_mesh["face_body_ids"][robot_face_ids[valid]]

    payload = {
        "slot_index": slot_index,
        "source_name": np.asarray(source_name),
        "robot_name": np.asarray(robot_name),
        "source_points": source_points,
        "robot_points": robot_points,
        "source_face_ids": source_face_ids,
        "source_bary": source_bind["bary"],
        "source_vertex_ids": source_bind["vertex_ids"],
        "source_normals": source_bind["closest_normals"],
        "source_bind_error": source_bind["errors"],
        "robot_face_ids": robot_face_ids,
        "robot_bary": robot_bind["bary"],
        "robot_vertex_ids": robot_bind["vertex_ids"],
        "robot_normals": robot_bind["closest_normals"],
        "robot_bind_error": robot_bind["errors"],
        "robot_geom_ids": robot_geom_ids,
        "robot_body_ids": robot_body_ids,
        "robot_geom_names": _lookup_names(robot_geom_ids, robot_mesh["geom_names"]),
        "robot_body_names": _lookup_names(robot_body_ids, robot_mesh["body_names"]),
        "robot_residual": residual,
        "correspondence_error": np.linalg.norm(residual, axis=-1).astype(np.float32),
        "n_source_faces": np.int32(n_faces_src),
        "n_robot_faces": np.int32(n_faces_robot),
    }
    if "normalization_scales" in slots:
        payload["source_normalization_scale"] = np.float32(
            np.asarray(slots["normalization_scales"]).reshape(-1)[source_idx]
        )
        payload["robot_normalization_scale"] = np.float32(
            np.asarray(slots["normalization_scales"]).reshape(-1)[robot_idx]
        )
    return payload


def save_contact_pairs(path: Path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    sidecar = path.with_suffix(".json")
    sidecar.write_text(
        json.dumps(
            {
                "source_name": str(np.asarray(payload["source_name"])),
                "robot_name": str(np.asarray(payload["robot_name"])),
                "num_slots": int(payload["slot_index"].shape[0]),
                "mean_source_bind_error": float(np.mean(payload["source_bind_error"])),
                "mean_robot_bind_error": float(np.mean(payload["robot_bind_error"])),
                "mean_correspondence_error": float(np.mean(payload["correspondence_error"])),
                "fields": {
                    "slot_index": "Shared AE slot. The same i is one SMPL-X/robot pair.",
                    "source_points / robot_points": "Paired surface points in each template root frame.",
                    "source_face_ids, source_bary, source_vertex_ids": "Bind onto the SMPL-X rest mesh; skin with LBS.",
                    "robot_face_ids, robot_bary, robot_geom_ids, robot_body_ids": "Bind onto the robot rest mesh; FK the geom.",
                    "correspondence_error": "L2 of the robot residual slot (match quality).",
                },
            },
            indent=2,
        )
        + "\n"
    )
    return path


def export_contact_pairs_from_files(
    slots_path: Path,
    data_path: Path,
    out_path: Path,
    source_name: str | None = None,
    robot_name: str | None = None,
) -> Path:
    slots = np.load(slots_path, allow_pickle=True)
    dataset = np.load(data_path, allow_pickle=True)
    names = np.asarray(slots["names"]).astype(str).tolist()
    template_name = (
        str(np.asarray(slots["template_name"]).reshape(-1)[0])
        if "template_name" in slots
        else None
    )
    source_name, robot_name = resolve_pair_names(names, source_name, robot_name, template_name)
    payload = build_contact_pairs(slots, dataset, source_name, robot_name)
    saved = save_contact_pairs(out_path, payload)
    print(
        f"[ContactPairs] {source_name} ↔ {robot_name}  slots={payload['slot_index'].shape[0]}  "
        f"src_bind={float(np.mean(payload['source_bind_error'])):.5f}  "
        f"robot_bind={float(np.mean(payload['robot_bind_error'])):.5f}  "
        f"corr_err={float(np.mean(payload['correspondence_error'])):.5f}"
    )
    print(f"[ContactPairs] saved {saved} and {saved.with_suffix('.json')}")
    return saved


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export 1-1 SMPL-X/robot correspondence slots for contact inference.",
    )
    parser.add_argument(
        "--slots",
        type=Path,
        default=Path("output/correspondence_template_residual_ae/correspondence_slots_final.npz"),
        help="Trained correspondence_slots_final.npz (or _latest).",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/correspondence_ae_tpose_4096.npz"),
        help="Dataset npz from build_correspondence_ae_dataset.py (needs mesh metadata).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output npz. Default: <slots-dir>/correspondence_contact_<source>_<robot>.npz",
    )
    parser.add_argument("--source", type=str, default=None, help="SMPL-X sample name (default: template / first smplx_*).")
    parser.add_argument("--robot", type=str, default=None, help="Robot sample name (default: d20 if present).")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.slots.is_file():
        raise FileNotFoundError(f"slots npz not found: {args.slots}")
    if not args.data.is_file():
        raise FileNotFoundError(f"dataset npz not found: {args.data}")
    slots = np.load(args.slots, allow_pickle=True)
    names = np.asarray(slots["names"]).astype(str).tolist()
    template_name = (
        str(np.asarray(slots["template_name"]).reshape(-1)[0])
        if "template_name" in slots
        else None
    )
    source_name, robot_name = resolve_pair_names(names, args.source, args.robot, template_name)
    out = args.out
    if out is None:
        out = args.slots.parent / f"correspondence_contact_{source_name}_{robot_name}.npz"
    export_contact_pairs_from_files(args.slots, args.data, out, source_name, robot_name)


if __name__ == "__main__":
    main()
