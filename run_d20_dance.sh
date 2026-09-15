PYTHONPATH=. python scripts/xsens_bvh_to_robot.py \
  --bvh_file datas/resource/dance1/dance_6.bvh \
  --robot d20_v2 \
  --scale 1 \
  --ik_config general_motion_retargeting/ik_configs/xsens_bvh_to_d20_v2_dance.json
