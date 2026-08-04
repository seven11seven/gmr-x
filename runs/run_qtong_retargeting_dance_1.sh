#!/bin/bash
# Stage 2: Retargeting Visualization Tool
#
# Dual-panel frame-by-frame comparison of robot vs human skeletons.
#   Panel A: interactive plotly HTML with frame slider (links view)
#   Panel B: MuJoCo viewer with mesh rendering + human skeleton overlay
#
# Usage:
#   bash runs/run_retargeting_visualizer.sh                          (default)
#   bash runs/run_retargeting_visualizer.sh --max-frames 100         (limit)
#   bash runs/run_retargeting_visualizer.sh --save cache.pkl         (save)
#   bash runs/run_retargeting_visualizer.sh --load cache.pkl         (replay)
#   bash runs/run_retargeting_visualizer.sh --no-panel-b             (web only)

conda run -n gmr_cpython python tools/retargeting_visualizer/visualize_retargeting.py \
  --xml assets/D20_29dof/mjcf/D20.xml \
  --bvh datas/qtong/dance/qt_dance1.bvh \
  --ik_config general_motion_retargeting/ik_configs/bvh_qtong_to_d20.json \
  --human-height 1.8 \
  --save outputs/d20_29dof/qt_dance_1.pkl \
  --save-csv outputs/d20_29dof/qt_dance_1.csv \
  "$@"
