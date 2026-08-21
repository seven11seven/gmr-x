"""Stage-4: full-trajectory retarget + web skeleton + MuJoCo mesh viewer sync."""

from __future__ import annotations

import copy
import tempfile
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

import mujoco as mj
import numpy as np

from tools.retargeting_app.loaders import (
    BVH_UNIT_OPTIONS,
    PROJECT_ROOT,
    resolve_under_project,
)
from tools.retargeting_app.stage2_logic import extract_offset_rows
from tools.retargeting_app.stage3_logic import (
    extract_weight_rows,
    _nodes_from_positions,
)

CUSTOM_ROBOT_KEY = "_retargeting_app_stage4"


# ---------------------------------------------------------------------------
# Global session / async job state
# ---------------------------------------------------------------------------

_lock = threading.RLock()
_job: dict[str, Any] | None = None
_session: dict[str, Any] | None = None
_viewer_thread: threading.Thread | None = None
_viewer_stop = threading.Event()


def get_job_status() -> dict[str, Any]:
    with _lock:
        if _job is None:
            return {"ok": True, "status": "idle"}
        out = {
            "ok": True,
            "status": _job["status"],
            "job_id": _job["job_id"],
            "progress": _job.get("progress", 0),
            "total": _job.get("total", 0),
            "message": _job.get("message", ""),
            "error": _job.get("error"),
        }
        if _job["status"] == "done" and _job.get("result_meta"):
            out.update(_job["result_meta"])
        return out


def get_session_meta() -> dict[str, Any]:
    with _lock:
        if not _session:
            return {"ok": True, "ready": False}
        return {
            "ok": True,
            "ready": True,
            "n_frames": _session["n_frames"],
            "fps": _session["fps"],
            "frame_idx": _session["frame_idx"],
            "playing": _session["playing"],
            "viewer_alive": _session.get("viewer_alive", False),
            "viewer_error": _session.get("viewer_error"),
            "xml_path": _session.get("xml_path"),
            "bvh_path": _session.get("bvh_path"),
            "actual_human_height": _session.get("actual_human_height"),
            "weight_rows": _session.get("weight_rows"),
            "robot_edges": _session.get("robot_edges"),
            "human_edges": _session.get("human_edges"),
        }


def get_web_frames() -> list[dict] | None:
    with _lock:
        if not _session:
            return None
        return _session["web_frames"]


def set_frame_idx(idx: int) -> dict[str, Any]:
    with _lock:
        if not _session:
            return {"ok": False, "error": "no session"}
        n = _session["n_frames"]
        idx = int(max(0, min(idx, n - 1)))
        _session["frame_idx"] = idx
        return {"ok": True, "frame_idx": idx, "playing": _session["playing"]}


def set_playing(playing: bool) -> dict[str, Any]:
    with _lock:
        if not _session:
            return {"ok": False, "error": "no session"}
        _session["playing"] = bool(playing)
        return {
            "ok": True,
            "playing": _session["playing"],
            "frame_idx": _session["frame_idx"],
        }


def stop_viewer() -> None:
    global _viewer_thread
    _viewer_stop.set()
    t = _viewer_thread
    if t and t.is_alive():
        t.join(timeout=2.0)
    _viewer_thread = None
    with _lock:
        if _session is not None:
            _session["viewer_alive"] = False


# ---------------------------------------------------------------------------
# Retarget (full trajectory, world frame — matches xsens_bvh_to_robot)
# ---------------------------------------------------------------------------

def _register_robot(xml: Path, config: dict) -> str:
    from general_motion_retargeting.params import (
        ROBOT_BASE_DICT,
        ROBOT_XML_DICT,
        VIEWER_CAM_DISTANCE_DICT,
    )

    ROBOT_XML_DICT[CUSTOM_ROBOT_KEY] = str(xml)
    base = config.get("robot_root_name") or "base_link"
    if base in ("aux_base", "pelvis"):
        ROBOT_BASE_DICT[CUSTOM_ROBOT_KEY] = "base_link"
    else:
        ROBOT_BASE_DICT[CUSTOM_ROBOT_KEY] = base
    VIEWER_CAM_DISTANCE_DICT.setdefault(CUSTOM_ROBOT_KEY, 2.5)
    return CUSTOM_ROBOT_KEY


def run_full_retarget(
    *,
    xml_path: str,
    bvh_path: str,
    config: dict,
    bvh_unit: int | None = None,
    max_frames: int | None = None,
    progress_cb=None,
) -> dict[str, Any]:
    """Retarget every BVH frame; return web skeletons + MuJoCo payloads.

    ``max_frames`` if set, only processes the first N frames (debug / smoke).
    """
    from general_motion_retargeting import GeneralMotionRetargeting as GMR
    from tools.retargeting_visualizer.visualize_retargeting import (
        get_human_edges,
        get_robot_body_info,
        get_robot_edges,
        load_bvh_all_frames,
    )

    unit = int(bvh_unit if bvh_unit is not None else config.get("bvh_unit", 1))
    if unit not in BVH_UNIT_OPTIONS:
        raise ValueError(f"Unsupported bvh_unit={unit}")
    scale = float(BVH_UNIT_OPTIONS[unit]["scale"])

    xml = resolve_under_project(xml_path)
    bvh = resolve_under_project(bvh_path)
    if not xml.exists():
        raise FileNotFoundError(f"Robot XML not found: {xml}")
    if not bvh.exists():
        raise FileNotFoundError(f"BVH file not found: {bvh}")

    end = int(max_frames) if max_frames is not None else None
    human_frames, actual_human_height, frame_time, parser = load_bvh_all_frames(
        bvh, start=None, end=end, scale=scale,
    )
    if not human_frames:
        raise RuntimeError("No BVH frames loaded")

    human_edges = [[a, b] for a, b in get_human_edges(parser)]
    n_total = len(human_frames)
    if progress_cb:
        progress_cb(0, n_total, "loading IK…")

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8",
    ) as fh:
        import json
        json.dump(config, fh, indent=2)
        tmp_ik = Path(fh.name)

    robot_key = _register_robot(xml, config)

    try:
        retargeter = GMR(
            src_human="bvh_xsens",
            tgt_robot=robot_key,
            actual_human_height=float(actual_human_height),
            ik_config_path=str(tmp_ik),
            verbose=False,
        )
        model = retargeter.model
        data0 = mj.MjData(model)
        mj.mj_forward(model, data0)
        robot_edges = [
            [a, b] for a, b in get_robot_edges(get_robot_body_info(model, data0))
            if a and b
        ]

        data_fk = mj.MjData(model)
        qpos_list: list[np.ndarray] = []
        human_mj_list: list[dict] = []
        web_frames: list[dict] = []

        print(f"\n  [Stage4] Retargeting {n_total} frames …")
        for fi, human_data in enumerate(human_frames):
            qpos = retargeter.retarget(human_data, offset_to_ground=True)
            qpos_list.append(qpos.copy())

            # MuJoCo overlay: same as xsens_bvh_to_robot (scaled_human_data)
            human_scaled = {
                name: [val[0].copy(), val[1].copy()]
                for name, val in retargeter.scaled_human_data.items()
            }
            human_mj_list.append(human_scaled)

            data_fk.qpos = qpos
            mj.mj_forward(model, data_fk)
            robot_positions = {}
            robot_xmat = {}
            for i in range(model.nbody):
                name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i)
                if name and name != "world":
                    robot_positions[name] = data_fk.xpos[i].copy()
                    robot_xmat[name] = data_fk.xmat[i].copy().reshape(3, 3)

            human_positions = {name: val[0] for name, val in human_scaled.items()}
            human_quats = {name: val[1] for name, val in human_scaled.items()}
            web_frames.append({
                "robot": {
                    "nodes": _nodes_from_positions(
                        robot_positions, xmats=robot_xmat,
                    ),
                },
                "human": {
                    "nodes": _nodes_from_positions(
                        human_positions, quats=human_quats,
                    ),
                },
            })

            if progress_cb and ((fi + 1) % 10 == 0 or fi + 1 == n_total):
                progress_cb(fi + 1, n_total, f"retarget {fi + 1}/{n_total}")
            if (fi + 1) % 100 == 0:
                print(f"    … {fi + 1}/{n_total}")

        print(f"  [Stage4] Done. {n_total} frames.")
    finally:
        try:
            tmp_ik.unlink(missing_ok=True)
        except Exception:
            pass

    fps = 1.0 / float(frame_time) if frame_time else 120.0
    xml_rel = (
        xml.relative_to(PROJECT_ROOT).as_posix()
        if xml.is_relative_to(PROJECT_ROOT) else str(xml)
    )
    bvh_rel = (
        bvh.relative_to(PROJECT_ROOT).as_posix()
        if bvh.is_relative_to(PROJECT_ROOT) else str(bvh)
    )

    return {
        "n_frames": n_total,
        "fps": float(fps),
        "actual_human_height": float(actual_human_height),
        "xml_path": xml_rel,
        "bvh_path": bvh_rel,
        "weight_rows": extract_weight_rows(config.get("ik_match_table1", {})),
        "offset_rows": extract_offset_rows(config.get("ik_match_table1", {})),
        "robot_edges": robot_edges,
        "human_edges": human_edges,
        "web_frames": web_frames,
        "qpos_list": qpos_list,
        "human_mj_list": human_mj_list,
        "robot_key": robot_key,
    }


# ---------------------------------------------------------------------------
# MuJoCo viewer thread (synced to session.frame_idx; web is playback master)
# ---------------------------------------------------------------------------

def _viewer_loop(robot_key: str, fps: float) -> None:
    from general_motion_retargeting import RobotMotionViewer

    viewer = None
    try:
        viewer = RobotMotionViewer(
            robot_type=robot_key,
            motion_fps=max(1.0, float(fps)),
            transparent_robot=1,
            record_video=False,
            camera_follow=True,
        )
        with _lock:
            if _session is not None:
                _session["viewer_alive"] = True
                _session["viewer_error"] = None

        last_idx = -1
        while not _viewer_stop.is_set():
            if not viewer.viewer.is_running():
                break
            with _lock:
                if not _session or not _session.get("qpos_list"):
                    time.sleep(0.05)
                    continue
                idx = int(_session["frame_idx"])
                qpos = _session["qpos_list"][idx]
                human = _session["human_mj_list"][idx]

            if idx != last_idx:
                viewer.step(
                    root_pos=qpos[:3],
                    root_rot=qpos[3:7],
                    dof_pos=qpos[7:],
                    human_motion_data=human,
                    rate_limit=False,
                    follow_camera=True,
                )
                last_idx = idx
            else:
                viewer.viewer.sync()

            time.sleep(0.008)

    except Exception as exc:
        traceback.print_exc()
        with _lock:
            if _session is not None:
                _session["viewer_error"] = str(exc)
                _session["viewer_alive"] = False
    finally:
        try:
            if viewer is not None:
                viewer.close()
        except Exception:
            pass
        with _lock:
            if _session is not None:
                _session["viewer_alive"] = False


def _start_viewer(robot_key: str, fps: float) -> None:
    global _viewer_thread
    stop_viewer()
    _viewer_stop.clear()
    _viewer_thread = threading.Thread(
        target=_viewer_loop,
        args=(robot_key, fps),
        name="stage4-mujoco-viewer",
        daemon=True,
    )
    _viewer_thread.start()


def _install_session(result: dict[str, Any], config: dict) -> None:
    global _session
    with _lock:
        _session = {
            "n_frames": result["n_frames"],
            "fps": result["fps"],
            "frame_idx": 0,
            "playing": False,
            "qpos_list": result["qpos_list"],
            "human_mj_list": result["human_mj_list"],
            "web_frames": result["web_frames"],
            "robot_edges": result["robot_edges"],
            "human_edges": result["human_edges"],
            "xml_path": result["xml_path"],
            "bvh_path": result["bvh_path"],
            "actual_human_height": result["actual_human_height"],
            "weight_rows": result["weight_rows"],
            "config": copy.deepcopy(config),
            "viewer_alive": False,
            "viewer_error": None,
            "robot_key": result["robot_key"],
        }
    _start_viewer(result["robot_key"], result["fps"])


def start_retarget_job(
    *,
    xml_path: str,
    bvh_path: str,
    config: dict,
    bvh_unit: int | None = None,
    max_frames: int | None = None,
) -> dict[str, Any]:
    """Kick off async full-trajectory retarget; returns job_id."""
    global _job

    with _lock:
        if _job and _job["status"] == "running":
            return {
                "ok": False,
                "error": "已有重定向任务在运行，请等待完成",
                "job_id": _job["job_id"],
            }
        job_id = uuid.uuid4().hex[:12]
        _job = {
            "job_id": job_id,
            "status": "running",
            "progress": 0,
            "total": 0,
            "message": "starting…",
            "error": None,
            "result_meta": None,
        }

    cfg = copy.deepcopy(config)

    def _progress(cur, total, msg):
        with _lock:
            if _job and _job["job_id"] == job_id:
                _job["progress"] = int(cur)
                _job["total"] = int(total)
                _job["message"] = str(msg)

    def _worker():
        global _job
        try:
            result = run_full_retarget(
                xml_path=xml_path,
                bvh_path=bvh_path,
                config=cfg,
                bvh_unit=bvh_unit,
                max_frames=max_frames,
                progress_cb=_progress,
            )
            _install_session(result, cfg)
            with _lock:
                if _job and _job["job_id"] == job_id:
                    _job["status"] = "done"
                    _job["progress"] = result["n_frames"]
                    _job["total"] = result["n_frames"]
                    _job["message"] = "done"
                    _job["result_meta"] = {
                        "n_frames": result["n_frames"],
                        "fps": result["fps"],
                        "xml_path": result["xml_path"],
                        "bvh_path": result["bvh_path"],
                        "actual_human_height": result["actual_human_height"],
                        "weight_rows": result["weight_rows"],
                        "viewer_error": (
                            _session.get("viewer_error") if _session else None
                        ),
                    }
        except Exception as exc:
            traceback.print_exc()
            with _lock:
                if _job and _job["job_id"] == job_id:
                    _job["status"] = "error"
                    _job["error"] = str(exc)
                    _job["message"] = "error"

    threading.Thread(
        target=_worker, name=f"stage4-retarget-{job_id}", daemon=True,
    ).start()
    return {"ok": True, "job_id": job_id, "status": "running"}
