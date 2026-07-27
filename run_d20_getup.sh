PYTHONPATH=. python scripts/xsens_bvh_to_robot.py \
  --bvh_file datas/resource/fallAndGetUp2/Xsens2/5.bvh \
  --robot d20_v2 \
  --scale 1 \
  --ik_config general_motion_retargeting/ik_configs/getup_xsens_bvh_to_d20_v2.json \
  --save_path outputs/getup_5.csv
  # --bvh_file datas/resource/fallAndGetUp2/Xsens2/5.bvh \
  # --bvh_file datas/resource/fallAndGetup1/example.bvh \
