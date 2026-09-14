#!/usr/bin/env python3
"""Build and train SMPL-X ↔ robot surface slots (UMR correspondence recipe).

This is the GMR counterpart of UMR ``humanoid_retarget_pipeline.py`` for the
build + train (+ optional view) stages. It does not run motion retargeting.

The previous direct train command used argparse defaults that are *not* the
UMR recipe:

* ``repulsion_weight=0.01`` vs UMR ``0.002`` (this alone kept loss ≈ 0.0063)
* ``edge_graph=euclidean`` / ``edge_weight=0.01`` / ``edge_k=8`` vs geodesic 0.4 / 32
* ``--fixed-template`` trains only the robot; UMR ``batch_size=2`` averages SMPL-X + robot
* D20 was centered on ``base_link``; UMR uses ``body:waist_yaw_link`` to match SMPL-X ``spine1``

Example::

    python scripts/humanoid_points_matching_pipeline.py \\
      --config robot_configs/humanoid_points_matching_d20.json \\
      --defaults robot_configs/humanoid_points_matching_defaults.json \\
      --stage train --force-build --force-train
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
POINTS_CONTACT = ROOT / "general_motion_retargeting" / "points_contact"
for path in (ROOT, SCRIPTS, POINTS_CONTACT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from humanoid_points_matching_config import (  # noqa: E402
    list_of_args,
    load_config,
    resolve_path,
    robot_config,
    section,
)

PYTHON = sys.executable
TRAIN_SCRIPT = POINTS_CONTACT / "train_correspondence_template_residual_ae.py"
VIEW_SCRIPT = SCRIPTS / "vis_correspondence_slots.py"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--defaults",
        type=Path,
        default=ROOT / "robot_configs" / "humanoid_points_matching_defaults.json",
    )
    parser.add_argument(
        "--stage",
        choices=["all", "build", "train", "view"],
        default="all",
        help="all runs build, train, then optional view.",
    )
    parser.add_argument("--force-build", action="store_true")
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--skip-view", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _clean_config(value):
    if isinstance(value, dict):
        return {str(key): _clean_config(item) for key, item in value.items() if not str(key).startswith("_")}
    if isinstance(value, list):
        return [_clean_config(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _deep_merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in {"extends", "_config_path", "_config_dir"}:
            continue
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _deep_merge_config(current, value)
        else:
            merged[key] = value
    return merged


def load_pipeline_config(config_path: Path, defaults_path: Path | None = None) -> dict[str, Any]:
    if defaults_path is None:
        return load_config(config_path)
    defaults = load_config(Path(defaults_path), use_default_extends=False)
    override = load_config(Path(config_path), use_default_extends=False)
    merged = _deep_merge_config(_clean_config(defaults), _clean_config(override))
    merged["_config_path"] = str(Path(override["_config_path"]).resolve())
    merged["_config_dir"] = str(Path(override["_config_dir"]).resolve())
    merged["_defaults_config_path"] = str(Path(defaults["_config_path"]).resolve())
    return merged


def bool_value(value, default=False):
    if value is None:
        return default
    return bool(value)


def parse_number(value):
    return float(value)


def config_joint_values(config: dict[str, Any], key: str) -> dict[str, float]:
    robot = robot_config(config)
    values = robot.get(key, {})
    if values is None:
        return {}
    if not isinstance(values, dict):
        raise ValueError(f"robot.{key} must be an object mapping joint name to value.")
    return {str(name): parse_number(value) for name, value in values.items()}


def mimic_qpos_for_build(robot: dict[str, Any]) -> dict[str, tuple[str, float]]:
    mimic = robot.get("mimic_qpos", {}) or {}
    converted = {}
    for joint_name, spec in mimic.items():
        if isinstance(spec, dict):
            source = spec.get("source")
            multiplier = spec.get("multiplier", 1.0)
            if source:
                converted[str(joint_name)] = (str(source), float(multiplier))
        elif isinstance(spec, (list, tuple)) and len(spec) == 2:
            converted[str(joint_name)] = (str(spec[0]), float(spec[1]))
    return converted


def run_command(cmd: list[str], dry_run=False):
    print("[PointsMatching] " + " ".join(str(part) for part in cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=ROOT, check=True)


def safe_cache_component(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value))


def smpl_template_config(config: dict[str, Any]) -> dict[str, Any]:
    template = dict(config.get("smpl_template", {}) or {})
    model_type = str(template.get("type", "smplx"))
    gender = str(template.get("gender", "neutral")).lower()
    name = template.get("name")
    if str(name) == "auto" or not name:
        name = f"{model_type}_{gender}"
    return {
        "name": str(name),
        "type": model_type,
        "gender": gender,
        "betas": template.get("betas"),
        "source": str(template.get("source", "manual")),
    }


def expected_smpl_slot_name(config: dict[str, Any]) -> str:
    configured = str(section(config, "correspondence").get("smpl_name", "auto"))
    return str(smpl_template_config(config)["name"]) if configured == "auto" else configured


def effective_robot_sample_pose(config: dict[str, Any]) -> str:
    return str(robot_config(config).get("sample_pose", "tpose"))


def effective_robot_sample_qpos(config: dict[str, Any], sample_pose: str) -> dict[str, Any]:
    pose = str(sample_pose)
    pose_qpos = robot_config(config).get(f"{pose}_qpos")
    if pose_qpos is not None:
        return config_joint_values(config, f"{pose}_qpos")
    if pose not in {"default", "none", "raw", "off"}:
        return config_joint_values(config, "tpose_qpos")
    return {}


def correspondence_cache_suffix(config: dict[str, Any]) -> str:
    template_cfg = smpl_template_config(config)
    name = str(template_cfg["name"])
    pose = effective_robot_sample_pose(config)
    pose_suffix = "" if pose in {"", "tpose"} else "_" + safe_cache_component(pose)
    dataset = section(section(config, "correspondence"), "dataset")
    smpl_center = str(dataset.get("smpl_center", "spine1"))
    center_suffix = ""
    if smpl_center != "spine1":
        center_suffix = "_smplcenter_" + safe_cache_component(smpl_center)
    if name == "smplx_neutral" and not pose_suffix:
        return center_suffix
    return "_" + safe_cache_component(name) + pose_suffix + center_suffix


def dataset_out(config: dict[str, Any]) -> Path:
    dataset = section(section(config, "correspondence"), "dataset")
    robot = robot_config(config)
    return resolve_path(
        dataset.get("out"),
        config,
        ROOT / f"data/correspondence_{robot['name']}{correspondence_cache_suffix(config)}.npz",
    )


def train_out_dir(config: dict[str, Any]) -> Path:
    train = section(section(config, "correspondence"), "train")
    robot = robot_config(config)
    return resolve_path(
        train.get("out_dir"),
        config,
        ROOT / f"output/correspondence_{robot['name']}{correspondence_cache_suffix(config)}",
    )


def slots_out(config: dict[str, Any]) -> Path:
    corr = section(config, "correspondence")
    if corr.get("slots"):
        return resolve_path(corr.get("slots"), config)
    return train_out_dir(config) / "correspondence_slots_final.npz"


def correspondence_dataset_compatible(dataset_path: Path, config: dict[str, Any], quiet=False) -> bool:
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        return False
    try:
        with np.load(dataset_path, allow_pickle=True) as data:
            names = [str(name) for name in np.asarray(data["names"]).reshape(-1).tolist()]
    except Exception as exc:
        if not quiet:
            print(f"[PointsMatching][WARN] cannot read correspondence dataset {dataset_path}: {exc}")
        return False
    robot = robot_config(config)
    robot_names = {str(robot.get("slot_name", robot.get("name"))), str(robot.get("name"))}
    expected_smpl = expected_smpl_slot_name(config)
    missing = []
    if expected_smpl not in names:
        missing.append(f"SMPL template {expected_smpl!r}")
    if not any(name in names for name in robot_names):
        missing.append(f"robot sample one of {sorted(robot_names)!r}")
    if missing and not quiet:
        print(
            f"[PointsMatching][WARN] correspondence dataset mismatch: {dataset_path} "
            f"names={names}; missing {', '.join(missing)}"
        )
    return not missing


def correspondence_slots_compatible(slots_path: Path, config: dict[str, Any], quiet=False) -> bool:
    slots_path = Path(slots_path)
    if not slots_path.exists():
        return False
    try:
        slots = np.load(slots_path, allow_pickle=True)
        names = [str(name) for name in np.asarray(slots["names"]).reshape(-1).tolist()]
    except Exception as exc:
        if not quiet:
            print(f"[PointsMatching][WARN] cannot read correspondence slots {slots_path}: {exc}")
        return False
    robot = robot_config(config)
    robot_name = str(robot.get("slot_name", robot["name"]))
    expected_smpl = expected_smpl_slot_name(config)
    missing = []
    if expected_smpl not in names:
        missing.append(f"SMPL template {expected_smpl!r}")
    if robot_name not in names:
        missing.append(f"robot sample {robot_name!r}")
    if missing and not quiet:
        print(
            f"[PointsMatching][WARN] correspondence slots mismatch: {slots_path} "
            f"names={names}; missing {', '.join(missing)}"
        )
    return not missing


def smpl_model_specs(config: dict[str, Any]):
    specs = section(section(config, "correspondence"), "dataset").get("smpl_models")
    if specs is None:
        return [
            {
                "type": "smplx",
                "dir": config.get("smplx_model_dir", "assets/smpl-x"),
                "genders": ["neutral"],
            }
        ]
    if not isinstance(specs, list):
        raise ValueError("correspondence.dataset.smpl_models must be a list.")
    return specs


def write_runtime_config(config: dict[str, Any], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    runtime_path = directory / "humanoid_points_matching_runtime_config.json"
    runtime_path.write_text(json.dumps(_clean_config(config), indent=2, sort_keys=True))
    print(f"[PointsMatching] wrote {runtime_path}")


def build_correspondence_dataset(config: dict[str, Any], force=False, dry_run=False) -> Path:
    import build_correspondence_ae_dataset as build

    out = dataset_out(config)
    if out.exists() and not force:
        if correspondence_dataset_compatible(out, config):
            print(f"[PointsMatching] reuse correspondence dataset: {out}")
            return out
        print(f"[PointsMatching] rebuild incompatible correspondence dataset: {out}")
    if dry_run:
        print(f"[PointsMatching] would build correspondence dataset: {out}")
        return out

    corr = section(config, "correspondence")
    dataset = section(corr, "dataset")
    robot = robot_config(config)
    num_points = int(dataset.get("num_points", 4096))
    seed = int(dataset.get("seed", 0))
    oversample_ratio = int(dataset.get("surface_oversample_ratio", 8))
    curvature_weight = float(dataset.get("surface_curvature_weight", 0.0))
    curvature_power = float(dataset.get("surface_curvature_power", 1.0))
    exterior_surface = bool_value(dataset.get("exterior_surface"), True)
    exterior_occlusion_distance = float(dataset.get("exterior_occlusion_distance", 0.12))
    exterior_method = str(dataset.get("exterior_method", "first_hit"))
    exterior_ray_distance = float(dataset.get("exterior_ray_distance", 0.0))
    smpl_center = str(dataset.get("smpl_center", "spine1"))
    bbox_center_ratio = float(dataset.get("bbox_center_ratio", 0.45))
    template_cfg = smpl_template_config(config)

    samples = []
    for offset, spec in enumerate(smpl_model_specs(config)):
        model_type = str(spec.get("type", "smplx"))
        model_dir = resolve_path(spec.get("dir"), config, config.get(f"{model_type}_model_dir", "assets/smpl-x"))
        genders = tuple(str(g) for g in spec.get("genders", [template_cfg["gender"]]))
        samples.extend(
            build.build_smpl_samples(
                model_dir,
                model_type,
                num_points,
                seed + 1000 * (offset + 1),
                oversample_ratio,
                curvature_weight,
                curvature_power,
                smpl_center,
                bbox_center_ratio,
                genders=genders,
                betas=spec.get("betas", template_cfg.get("betas")),
                name=spec.get("name", template_cfg["name"] if len(genders) == 1 else None),
            )
        )

    expected_human_name = expected_smpl_slot_name(config)
    built_human_names = {str(sample["name"]) for sample in samples}
    if expected_human_name not in built_human_names:
        raise RuntimeError(
            f"Required human template {expected_human_name!r} could not be built. "
            f"Got {sorted(built_human_names)}. Check smplx_model_dir / SMPLX_NEUTRAL.pkl."
        )

    robot_xml = resolve_path(robot.get("xml"), config)
    robot_name = str(robot.get("slot_name", robot["name"]))
    point_cloud_center = str(robot["point_cloud_center"])
    sample_point_cloud_center = str(robot.get("sample_point_cloud_center", point_cloud_center))
    robot_sample_pose = effective_robot_sample_pose(config)
    robot_sample_qpos = effective_robot_sample_qpos(config, robot_sample_pose)
    reset_key = robot.get("reset_key")
    if reset_key is not None:
        reset_key = str(reset_key).strip() or None
    samples.append(
        build.build_robot_sample(
            robot_xml,
            robot_name,
            num_points,
            seed + 9000,
            oversample_ratio,
            curvature_weight,
            curvature_power,
            to_smpl_frame=bool_value(robot.get("to_smpl_frame"), True),
            pose=robot_sample_pose,
            exterior_surface=exterior_surface,
            exterior_occlusion_distance=exterior_occlusion_distance,
            exterior_method=exterior_method,
            exterior_ray_distance=exterior_ray_distance,
            point_cloud_center_name=sample_point_cloud_center,
            tpose_qpos=robot_sample_qpos,
            mimic_qpos=mimic_qpos_for_build(robot),
            reset_key=reset_key,
        )
    )

    names = np.asarray([s["name"] for s in samples])
    points = np.stack([s["points"] for s in samples], axis=0).astype(np.float32)
    root_offsets = np.stack([s["root_offset"] for s in samples], axis=0).astype(np.float32)
    center_modes = np.asarray([s["center_mode"] for s in samples])
    save_data = {
        "names": names,
        "points": points,
        "root_offsets": root_offsets,
        "center_modes": center_modes,
        "num_points": np.asarray(num_points, dtype=np.int32),
        "seed": np.asarray(seed, dtype=np.int32),
        "surface_oversample_ratio": np.asarray(oversample_ratio, dtype=np.int32),
        "surface_curvature_weight": np.asarray(curvature_weight, dtype=np.float32),
        "surface_curvature_power": np.asarray(curvature_power, dtype=np.float32),
        "bbox_center_ratio": np.asarray(bbox_center_ratio, dtype=np.float32),
        "robot_exterior_surface": np.asarray(exterior_surface),
        "robot_exterior_occlusion_distance": np.asarray(exterior_occlusion_distance, dtype=np.float32),
        "robot_exterior_method": np.asarray(exterior_method),
        "robot_exterior_ray_distance": np.asarray(exterior_ray_distance, dtype=np.float32),
        "custom_robot_name": np.asarray(robot_name),
        "custom_robot_xml": np.asarray(str(robot_xml)),
        "custom_robot_point_cloud_center": np.asarray(sample_point_cloud_center),
        "custom_robot_sample_pose": np.asarray(robot_sample_pose),
        "custom_robot_sample_qpos_names": np.asarray(list(robot_sample_qpos.keys()), dtype=object),
        "custom_robot_sample_qpos_values": np.asarray(
            [float(value) for value in robot_sample_qpos.values()], dtype=np.float32
        ),
    }
    for idx, sample in enumerate(samples):
        save_data[f"mesh_vertices_{idx}"] = sample["vertices"].astype(np.float32)
        save_data[f"mesh_faces_{idx}"] = sample["faces"].astype(np.int32)
        save_data[f"sample_face_ids_{idx}"] = sample["sample_face_ids"].astype(np.int32)
        if "face_geom_ids" in sample:
            save_data[f"face_geom_ids_{idx}"] = np.asarray(sample["face_geom_ids"], dtype=np.int32)
            save_data[f"face_body_ids_{idx}"] = np.asarray(sample["face_body_ids"], dtype=np.int32)
            save_data[f"geom_names_{idx}"] = np.asarray(sample["geom_names"])
            save_data[f"body_names_{idx}"] = np.asarray(sample["body_names"])
        if "betas" in sample:
            save_data[f"betas_{idx}"] = np.asarray(sample["betas"], dtype=np.float32)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **save_data)
    print(f"[PointsMatching] saved correspondence dataset: {out} names={names.tolist()} points={points.shape}")
    return out


def train_correspondence(config: dict[str, Any], dataset_path: Path, force=False, dry_run=False) -> Path:
    out_dir = train_out_dir(config)
    final_slots = slots_out(config)
    if final_slots.exists() and not force:
        if correspondence_slots_compatible(final_slots, config):
            print(f"[PointsMatching] reuse correspondence slots: {final_slots}")
            return final_slots
        print(f"[PointsMatching] rebuild correspondence slots for template={expected_smpl_slot_name(config)!r}")
    if not dry_run and not correspondence_dataset_compatible(dataset_path, config):
        raise ValueError(
            f"Correspondence dataset is missing or incompatible: {dataset_path}. "
            "Build it before training."
        )
    train = section(section(config, "correspondence"), "train")
    template_name = train.get("template_name", "auto")
    if str(template_name) == "auto":
        template_name = smpl_template_config(config)["name"]
    write_runtime_config(config, out_dir)
    train_args = {
        "data": dataset_path,
        "out_dir": out_dir,
        "template_name": template_name,
        "template_sort": train.get("template_sort", "y_z_x"),
        "num_points": section(section(config, "correspondence"), "dataset").get("num_points", 4096),
        "normalize": train.get("normalize", "per_sample_height"),
        "epochs": train.get("epochs", 500),
        "batch_size": train.get("batch_size", 2),
        "fixed_template": bool_value(train.get("fixed_template"), False),
        "lr": train.get("lr", 1e-3),
        "lr_scheduler": train.get("lr_scheduler", "cosine"),
        "min_lr": train.get("min_lr", 1e-5),
        "chamfer_weight": train.get("chamfer_weight", 1.0),
        "repulsion_weight": train.get("repulsion_weight", 0.002),
        "repulsion_k": train.get("repulsion_k", 8),
        "repulsion_radius": train.get("repulsion_radius", 0.035),
        "residual_weight": train.get("residual_weight", 0.0),
        "edge_weight": train.get("edge_weight", 0.4),
        "edge_graph": train.get("edge_graph", "geodesic"),
        "edge_k": train.get("edge_k", 32),
        "noise_std": train.get("noise_std", 0.002),
        "dropout_ratio": train.get("dropout_ratio", 0.0),
        "resample_every": train.get("resample_every", 0),
        "log_every": train.get("log_every", 100),
        "save_every": train.get("save_every", 1000),
        "seed": train.get("seed", 0),
        "device": train.get("device", "auto"),
    }
    extra = train.get("extra_args", {})
    if isinstance(extra, dict):
        train_args.update(extra)
    cmd = [PYTHON, str(TRAIN_SCRIPT), *list_of_args(train_args)]
    run_command(cmd, dry_run=dry_run)
    return final_slots


def visualize_slots(config: dict[str, Any], dataset_path: Path, slots_path: Path, dry_run=False):
    view = section(config, "view")
    if view.get("enabled", True) is False:
        print("[PointsMatching] view disabled by config.")
        return
    robot = robot_config(config)
    view_args = {
        "data": dataset_path,
        "slots": slots_path if slots_path.exists() else None,
        "source": expected_smpl_slot_name(config),
        "robot": str(robot.get("slot_name", robot["name"])),
        "smplx_model": resolve_path(config.get("smplx_model_dir"), config, ROOT / "assets" / "smpl-x"),
        "color_by": view.get("color_by", "source-part"),
        "backend": view.get("backend", "matplotlib"),
        "lines": view.get("lines", 0),
    }
    extra = view.get("extra_args", {}) or {}
    if isinstance(extra, dict):
        view_args.update(extra)
    cmd = [PYTHON, str(VIEW_SCRIPT), *list_of_args(view_args)]
    run_command(cmd, dry_run=dry_run)


def run_pipeline(config: dict[str, Any], args: argparse.Namespace) -> None:
    stages = ["build", "train", "view"] if args.stage == "all" else [args.stage]
    dataset_path = dataset_out(config)
    slots_path = slots_out(config)

    if "build" in stages:
        dataset_path = build_correspondence_dataset(config, force=args.force_build, dry_run=args.dry_run)
    if "train" in stages:
        dataset_path = build_correspondence_dataset(config, force=args.force_build, dry_run=args.dry_run)
        slots_path = train_correspondence(config, dataset_path, force=args.force_train, dry_run=args.dry_run)
    if "view" in stages and not args.skip_view:
        visualize_slots(config, dataset_path, slots_path, dry_run=args.dry_run)


def main() -> int:
    args = parse_args()
    config = load_pipeline_config(args.config, args.defaults)
    robot = robot_config(config)
    print(
        f"[PointsMatching] robot={robot['name']} xml={resolve_path(robot.get('xml'), config)} "
        f"center={robot['point_cloud_center']} pose={effective_robot_sample_pose(config)} "
        f"template={expected_smpl_slot_name(config)}"
    )
    run_pipeline(config, args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
