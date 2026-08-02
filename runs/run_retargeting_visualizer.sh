#!/bin/bash
# Stage 2: Retargeting Visualization Tool
#
# Dual-panel frame-by-frame comparison of robot vs human skeletons.
#   Panel A: interactive plotly HTML with frame slider (links view)
#   Panel B: MuJoCo viewer with mesh rendering + human skeleton overlay
#
# Usage:
#   bash runs/run_retargeting_visualizer.sh
#   bash runs/run_retargeting_visualizer.sh --max-frames 100  (limit frames)
#   bash runs/run_retargeting_visualizer.sh --no-panel-a        (skip HTML)
#   bash runs/run_retargeting_visualizer.sh --no-panel-b        (skip MuJoCo)

conda run -n gmr_cpython python tools/retargeting_visualizer/visualize_retargeting.py \
  --xml assets/D20_29dof/mjcf/D20.xml \
  --bvh datas/resource/fallAndGetUp2/Xsens2/5.bvh \
  --ik_config general_motion_retargeting/ik_configs/xsens_bvh_to_d20_29dof_getup.json \
  "$@"
