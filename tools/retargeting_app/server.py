#!/usr/bin/env python3
"""
Stage-1 retargeting config web app.

Visualize robot XML links + BVH bones, set body↔bone mapping and BVH unit,
then save an ik_config JSON (same schema as general_motion_retargeting/ik_configs/).

Usage (from GMR/):
    conda run -n gmr python tools/retargeting_app/server.py
    # then open http://127.0.0.1:8765
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import traceback

from flask import Flask, jsonify, render_template, request

_HERE = pathlib.Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools.retargeting_app.config_builder import (  # noqa: E402
    apply_solver_flags,
    build_config,
    load_config_file,
    mapping_from_ik_table,
    save_config_file,
)
from tools.retargeting_app.loaders import (  # noqa: E402
    BVH_UNIT_OPTIONS,
    list_bvh_files,
    list_ik_configs,
    list_robot_xmls,
    load_scene,
)
from tools.retargeting_app.stage2_logic import (  # noqa: E402
    build_stage2_scene,
    matrix_to_quat_wxyz,
    merge_stage2_edits,
    quat_wxyz_to_matrix,
)
from tools.retargeting_app.auto_estimate import auto_estimate  # noqa: E402
from tools.retargeting_app.stage3_logic import (  # noqa: E402
    extract_weight_rows,
    merge_stage3_weights,
    run_preview_retarget,
)
from tools.retargeting_app import stage4_logic as stage4  # noqa: E402
from tools.retargeting_app import aux_logic  # noqa: E402

app = Flask(
    __name__,
    template_folder=str(_HERE / "templates"),
    static_folder=str(_HERE / "static"),
    static_url_path="/static",
)

DEFAULT_XML = "assets/D20_29dof/mjcf/D20.xml"
DEFAULT_BVH = "datas/resource/fallAndGetUp2/Xsens2/5.bvh"
# Sample Xsens BVHs in this repo are already in meters; cm/mm selectable in UI.
DEFAULT_BVH_UNIT = 1


DEFAULT_STAGE2_CONFIG = (
    "tools/retargeting_app/saved_configs/stage1_xsens_to_d20.json"
)


@app.get("/")
def index():
    return render_template("stage1.html")


@app.get("/stage2")
def stage2_page():
    return render_template("stage2.html")


@app.get("/stage3")
def stage3_page():
    return render_template("stage3.html")


@app.get("/stage4")
def stage4_page():
    return render_template("stage4.html")


@app.get("/stage5")
def stage5_page():
    return render_template("stage5.html")


@app.get("/api/meta")
def api_meta():
    return jsonify({
        "defaults": {
            "xml": DEFAULT_XML,
            "bvh": DEFAULT_BVH,
            "bvh_unit": DEFAULT_BVH_UNIT,
            "config": DEFAULT_STAGE2_CONFIG,
        },
        "bvh_units": [
            {"unit": u, "label": info["label"], "scale": info["scale"]}
            for u, info in BVH_UNIT_OPTIONS.items()
        ],
        "xml_files": list_robot_xmls(),
        "bvh_files": list_bvh_files(),
        "ik_configs": list_ik_configs(),
    })


@app.post("/api/scene")
def api_scene():
    try:
        payload = request.get_json(force=True) or {}
        xml = payload.get("xml") or DEFAULT_XML
        bvh = payload.get("bvh") or DEFAULT_BVH
        bvh_unit = int(payload.get("bvh_unit", 100))
        scene = load_scene(xml, bvh, bvh_unit=bvh_unit)

        # Optional: attach mapping preview from an existing config
        config_path = payload.get("config_path")
        mapping = payload.get("mapping") or {}
        if config_path and not mapping:
            cfg = load_config_file(config_path)
            mapping = mapping_from_ik_table(cfg.get("ik_match_table1", {}))
            if "bvh_unit" in cfg:
                # If config has unit but user didn't override, keep user's unit
                pass
            scene["loaded_config"] = {
                "path": config_path,
                "robot_root_name": cfg.get("robot_root_name"),
                "human_root_name": cfg.get("human_root_name"),
                "mapping": mapping,
                "bvh_unit": cfg.get("bvh_unit"),
                "bvh_scale": cfg.get("bvh_scale"),
            }
        scene["mapping"] = mapping
        return jsonify({"ok": True, "scene": scene})
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.get("/api/config")
def api_get_config():
    path = request.args.get("path")
    if not path:
        return jsonify({"ok": False, "error": "missing path"}), 400
    try:
        cfg = load_config_file(path)
        mapping = mapping_from_ik_table(cfg.get("ik_match_table1", {}))
        return jsonify({
            "ok": True,
            "path": path,
            "config": cfg,
            "mapping": mapping,
            "bvh_unit": cfg.get("bvh_unit", 100),
        })
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.post("/api/config/save")
def api_save_config():
    try:
        payload = request.get_json(force=True) or {}
        mapping = payload.get("mapping") or {}
        bvh_unit = int(payload.get("bvh_unit", 100))
        filename = payload.get("filename") or "stage1_mapping.json"
        dest_dir = payload.get("dest_dir")  # optional
        base_path = payload.get("base_config_path")
        robot_names = payload.get("robot_names") or []
        human_names = payload.get("human_names") or []
        robot_root = payload.get("robot_root_name")
        human_root = payload.get("human_root_name")

        # Drop empty mappings
        mapping = {k: v for k, v in mapping.items() if k and v}

        base_cfg = load_config_file(base_path) if base_path else None
        config = build_config(
            mapping=mapping,
            bvh_unit=bvh_unit,
            robot_names=robot_names,
            human_names=human_names,
            base_config=base_cfg,
            robot_root_name=robot_root,
            human_root_name=human_root,
        )

        # Also allow writing into the official ik_configs folder when requested
        if payload.get("save_to_ik_configs"):
            dest_dir = "general_motion_retargeting/ik_configs"

        saved = save_config_file(config, filename, dest_dir=dest_dir)
        return jsonify({"ok": True, "path": saved, "config": config})
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.post("/api/config/preview")
def api_preview_config():
    """Return the JSON that would be saved, without writing to disk."""
    try:
        payload = request.get_json(force=True) or {}
        mapping = {k: v for k, v in (payload.get("mapping") or {}).items() if k and v}
        bvh_unit = int(payload.get("bvh_unit", 100))
        base_path = payload.get("base_config_path")
        base_cfg = load_config_file(base_path) if base_path else None
        config = build_config(
            mapping=mapping,
            bvh_unit=bvh_unit,
            robot_names=payload.get("robot_names") or [],
            human_names=payload.get("human_names") or [],
            base_config=base_cfg,
            robot_root_name=payload.get("robot_root_name"),
            human_root_name=payload.get("human_root_name"),
        )
        return jsonify({
            "ok": True,
            "config": config,
            "json_text": json.dumps(config, indent=4, ensure_ascii=False),
        })
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


# ---------------------------------------------------------------------------
# Stage-2 APIs: scale + offset tuning with aligned skeleton comparison
# ---------------------------------------------------------------------------

@app.post("/api/stage2/scene")
def api_stage2_scene():
    """Load XML+BVH+config, apply scale/offset, align roots to origin."""
    try:
        payload = request.get_json(force=True) or {}
        xml = payload.get("xml") or DEFAULT_XML
        bvh = payload.get("bvh") or DEFAULT_BVH
        config_path = payload.get("config_path")
        if not config_path and not payload.get("config"):
            config_path = DEFAULT_STAGE2_CONFIG

        if payload.get("config"):
            base_cfg = payload["config"]
        else:
            base_cfg = load_config_file(config_path)

        cfg = merge_stage2_edits(
            base_cfg,
            scale_vertical=payload.get("scale_vertical"),
            scale_horizontal=payload.get("scale_horizontal"),
            offset_rows=payload.get("offset_rows"),
            robot_root_name=payload.get("robot_root_name"),
            human_root_name=payload.get("human_root_name"),
            ground_height=payload.get("ground_height"),
            human_height_assumption=payload.get("human_height_assumption"),
            bvh_unit=payload.get("bvh_unit"),
        )

        unit = int(payload.get("bvh_unit", cfg.get("bvh_unit", DEFAULT_BVH_UNIT)))
        scene = build_stage2_scene(
            xml_path=xml,
            bvh_path=bvh,
            config=cfg,
            bvh_unit=unit,
            apply_height_ratio=bool(payload.get("apply_height_ratio", True)),
        )
        return jsonify({
            "ok": True,
            "scene": scene,
            "config": cfg,
            "config_path": config_path,
        })
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.post("/api/stage2/config/save")
def api_stage2_save():
    try:
        payload = request.get_json(force=True) or {}
        config_path = payload.get("config_path") or payload.get("base_config_path")
        if not config_path and not payload.get("base_config"):
            return jsonify({"ok": False, "error": "missing config_path"}), 400

        base_cfg = (
            payload.get("base_config")
            or load_config_file(config_path)
        )
        cfg = merge_stage2_edits(
            base_cfg,
            scale_vertical=payload.get("scale_vertical"),
            scale_horizontal=payload.get("scale_horizontal"),
            offset_rows=payload.get("offset_rows"),
            robot_root_name=payload.get("robot_root_name"),
            human_root_name=payload.get("human_root_name"),
            ground_height=payload.get("ground_height"),
            human_height_assumption=payload.get("human_height_assumption"),
            bvh_unit=payload.get("bvh_unit"),
        )

        filename = payload.get("filename")
        if not filename:
            # overwrite same file name when saving back
            filename = pathlib.Path(config_path).name if config_path else "stage2_config.json"

        dest_dir = payload.get("dest_dir")
        if payload.get("save_to_ik_configs"):
            dest_dir = "general_motion_retargeting/ik_configs"
        elif payload.get("overwrite") and config_path:
            dest_dir = str(pathlib.Path(config_path).parent)

        saved = save_config_file(cfg, filename, dest_dir=dest_dir)
        return jsonify({"ok": True, "path": saved, "config": cfg})
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.post("/api/stage2/rot_convert")
def api_stage2_rot_convert():
    """Convert between rotation matrix and quaternion (wxyz)."""
    try:
        payload = request.get_json(force=True) or {}
        if "matrix" in payload:
            quat = matrix_to_quat_wxyz(payload["matrix"])
            mat = quat_wxyz_to_matrix(quat)
            return jsonify({"ok": True, "quat_wxyz": quat, "matrix": mat})
        if "quat_wxyz" in payload:
            mat = quat_wxyz_to_matrix(payload["quat_wxyz"])
            return jsonify({"ok": True, "quat_wxyz": list(payload["quat_wxyz"]), "matrix": mat})
        return jsonify({"ok": False, "error": "provide matrix or quat_wxyz"}), 400
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.post("/api/stage2/auto_estimate")
def api_stage2_auto_estimate():
    """T-pose auto estimate of scale tables + rotation offsets."""
    try:
        payload = request.get_json(force=True) or {}
        xml = payload.get("xml") or DEFAULT_XML
        bvh = payload.get("bvh") or DEFAULT_BVH
        config_path = payload.get("config_path")
        if payload.get("config"):
            cfg = payload["config"]
        elif config_path:
            cfg = load_config_file(config_path)
        else:
            return jsonify({"ok": False, "error": "missing config"}), 400

        # Merge current UI edits into config before estimating
        cfg = merge_stage2_edits(
            cfg,
            scale_vertical=payload.get("scale_vertical"),
            scale_horizontal=payload.get("scale_horizontal"),
            offset_rows=payload.get("offset_rows"),
            robot_root_name=payload.get("robot_root_name"),
            human_root_name=payload.get("human_root_name"),
            bvh_unit=payload.get("bvh_unit"),
        )

        result = auto_estimate(
            xml_path=xml,
            bvh_path=bvh,
            config=cfg,
            bvh_unit=payload.get("bvh_unit", cfg.get("bvh_unit")),
            resolutions=payload.get("resolutions"),
        )
        return jsonify({"ok": True, **result})
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(exc)}), 400


# ---------------------------------------------------------------------------
# Stage-3 APIs: fine-tune aux_* body local positions in robot XML
# ---------------------------------------------------------------------------

@app.post("/api/stage3/aux_scene")
def api_stage3_aux_scene():
    """Robot (+ optional Stage-2 human) with optional in-memory aux overrides."""
    try:
        payload = request.get_json(force=True) or {}
        xml = payload.get("xml") or DEFAULT_XML
        bvh = payload.get("bvh")
        config_path = payload.get("config_path")
        cfg = payload.get("config")
        if cfg is None and config_path:
            cfg = load_config_file(config_path)
        if cfg is not None and payload.get("bvh_unit") is not None:
            unit = int(payload["bvh_unit"])
            if unit in BVH_UNIT_OPTIONS:
                cfg = dict(cfg)
                cfg["bvh_unit"] = unit
                cfg["bvh_scale"] = float(BVH_UNIT_OPTIONS[unit]["scale"])

        aux_positions = payload.get("aux_positions") or None
        scene = aux_logic.build_aux_scene(
            xml_path=xml,
            bvh_path=bvh if cfg is not None else None,
            config=cfg,
            bvh_unit=(cfg or {}).get("bvh_unit") if cfg else payload.get("bvh_unit"),
            aux_positions=aux_positions,
        )
        return jsonify({"ok": True, **scene})
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.post("/api/stage3/aux_save")
def api_stage3_aux_save():
    """Write aux_* local positions back to robot XML (overwrite or copy)."""
    try:
        payload = request.get_json(force=True) or {}
        xml = payload.get("xml") or DEFAULT_XML
        aux_positions = payload.get("aux_positions") or {}
        if not aux_positions:
            return jsonify({"ok": False, "error": "missing aux_positions"}), 400
        dest = payload.get("dest_path")
        saved = aux_logic.write_xml_with_aux_positions(
            xml_path=xml,
            aux_positions=aux_positions,
            dest_path=dest,
        )
        return jsonify({"ok": True, "path": saved})
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(exc)}), 400


# ---------------------------------------------------------------------------
# Stage-4 APIs: pos/rot weight tuning + short retarget preview
# ---------------------------------------------------------------------------

@app.post("/api/stage4/weights")
def api_stage4_weights():
    """Load weight rows from a config (no retarget)."""
    try:
        payload = request.get_json(force=True) or {}
        config_path = payload.get("config_path") or DEFAULT_STAGE2_CONFIG
        if payload.get("config"):
            cfg = payload["config"]
        else:
            cfg = load_config_file(config_path)
        rows = extract_weight_rows(cfg.get("ik_match_table1", {}))
        return jsonify({
            "ok": True,
            "config_path": config_path,
            "config": cfg,
            "weight_rows": rows,
        })
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.post("/api/stage4/retarget")
def api_stage4_retarget_preview():
    """Retarget first N frames with current weights for preview."""
    try:
        payload = request.get_json(force=True) or {}
        xml = payload.get("xml") or DEFAULT_XML
        bvh = payload.get("bvh") or DEFAULT_BVH
        config_path = payload.get("config_path")
        max_frames = int(payload.get("max_frames", 50))
        sample = str(payload.get("sample") or "uniform")

        if payload.get("config"):
            base_cfg = payload["config"]
        elif config_path:
            base_cfg = load_config_file(config_path)
        else:
            base_cfg = load_config_file(DEFAULT_STAGE2_CONFIG)
            config_path = DEFAULT_STAGE2_CONFIG

        cfg = merge_stage3_weights(base_cfg, payload.get("weight_rows"))
        cfg = apply_solver_flags(cfg, payload)
        if payload.get("bvh_unit") is not None:
            unit = int(payload["bvh_unit"])
            if unit in BVH_UNIT_OPTIONS:
                cfg["bvh_unit"] = unit
                cfg["bvh_scale"] = float(BVH_UNIT_OPTIONS[unit]["scale"])

        result = run_preview_retarget(
            xml_path=xml,
            bvh_path=bvh,
            config=cfg,
            bvh_unit=cfg.get("bvh_unit"),
            max_frames=max_frames,
            sample=sample,
        )
        return jsonify({
            "ok": True,
            "config": cfg,
            "config_path": config_path,
            **result,
        })
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.post("/api/stage4/config/save")
def api_stage4_save():
    try:
        payload = request.get_json(force=True) or {}
        config_path = payload.get("config_path") or payload.get("base_config_path")
        if not config_path and not payload.get("base_config"):
            return jsonify({"ok": False, "error": "missing config_path"}), 400

        base_cfg = payload.get("base_config") or load_config_file(config_path)
        cfg = merge_stage3_weights(base_cfg, payload.get("weight_rows"))
        cfg = apply_solver_flags(cfg, payload)

        filename = payload.get("filename")
        if not filename:
            filename = pathlib.Path(config_path).name if config_path else "stage4_config.json"

        dest_dir = payload.get("dest_dir")
        if payload.get("save_to_ik_configs"):
            dest_dir = "general_motion_retargeting/ik_configs"
        elif payload.get("overwrite") and config_path:
            dest_dir = str(pathlib.Path(config_path).parent)

        saved = save_config_file(cfg, filename, dest_dir=dest_dir)
        return jsonify({"ok": True, "path": saved, "config": cfg})
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(exc)}), 400


# ---------------------------------------------------------------------------
# Stage-5 APIs: full-trajectory retarget + MuJoCo sync
# ---------------------------------------------------------------------------

@app.post("/api/stage5/retarget")
def api_stage5_retarget():
    """Start async full-trajectory retarget (+ MuJoCo viewer)."""
    try:
        payload = request.get_json(force=True) or {}
        xml = payload.get("xml") or DEFAULT_XML
        bvh = payload.get("bvh") or DEFAULT_BVH
        config_path = payload.get("config_path")

        if payload.get("config"):
            base_cfg = payload["config"]
        elif config_path:
            base_cfg = load_config_file(config_path)
        else:
            base_cfg = load_config_file(DEFAULT_STAGE2_CONFIG)
            config_path = DEFAULT_STAGE2_CONFIG

        cfg = merge_stage3_weights(base_cfg, payload.get("weight_rows"))
        cfg = apply_solver_flags(cfg, payload)
        if payload.get("bvh_unit") is not None:
            unit = int(payload["bvh_unit"])
            if unit in BVH_UNIT_OPTIONS:
                cfg["bvh_unit"] = unit
                cfg["bvh_scale"] = float(BVH_UNIT_OPTIONS[unit]["scale"])

        result = stage4.start_retarget_job(
            xml_path=xml,
            bvh_path=bvh,
            config=cfg,
            bvh_unit=cfg.get("bvh_unit"),
            max_frames=payload.get("max_frames"),
        )
        if not result.get("ok"):
            return jsonify(result), 400
        return jsonify({**result, "config_path": config_path, "config": cfg})
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.get("/api/stage5/status")
def api_stage5_status():
    return jsonify(stage4.get_job_status())


@app.get("/api/stage5/frames")
def api_stage5_frames():
    """Return web skeleton frames for the current Stage-5 session."""
    try:
        meta = stage4.get_session_meta()
        frames = stage4.get_web_frames()
        if not meta.get("ready") or frames is None:
            return jsonify({"ok": False, "error": "no session — run retarget first"}), 400
        with stage4._lock:
            cfg = stage4._session.get("config") if stage4._session else None
        return jsonify({
            "ok": True,
            "config": cfg,
            "n_frames": meta["n_frames"],
            "fps": meta["fps"],
            "xml_path": meta.get("xml_path"),
            "bvh_path": meta.get("bvh_path"),
            "actual_human_height": meta.get("actual_human_height"),
            "weight_rows": meta.get("weight_rows"),
            "robot_edges": meta.get("robot_edges"),
            "human_edges": meta.get("human_edges"),
            "viewer_alive": meta.get("viewer_alive"),
            "viewer_error": meta.get("viewer_error"),
            "frames": frames,
        })
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.post("/api/stage5/sync")
def api_stage5_sync():
    """Sync MuJoCo viewer to a frame index (and optional play flag)."""
    try:
        payload = request.get_json(force=True) or {}
        if "frame_idx" in payload:
            stage4.set_frame_idx(int(payload["frame_idx"]))
        if "playing" in payload:
            stage4.set_playing(bool(payload["playing"]))
        meta = stage4.get_session_meta()
        return jsonify({"ok": True, **{k: meta[k] for k in (
            "frame_idx", "playing", "viewer_alive", "n_frames", "fps",
        ) if k in meta}})
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(exc)}), 400


def main():
    ap = argparse.ArgumentParser(description="GMR retargeting config web app")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    print(f"GMR app → http://{args.host}:{args.port}")
    print(f"  Stage-1: http://{args.host}:{args.port}/")
    print(f"  Stage-2: http://{args.host}:{args.port}/stage2")
    print(f"  Stage-3: http://{args.host}:{args.port}/stage3  (Aux)")
    print(f"  Stage-4: http://{args.host}:{args.port}/stage4  (Weights)")
    print(f"  Stage-5: http://{args.host}:{args.port}/stage5  (Retarget)")
    print(f"Project root: {_PROJECT_ROOT}")
    # threaded: allow /sync while retarget job + MuJoCo viewer run
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
