conda run -n gmr python tools/links_mapping/visualize_links_mapping.py \
  --xml assets/D20_29dof/mjcf/D20_fz.xml \
  --ik_config general_motion_retargeting/ik_configs/bvh_fz_motion_to_d20.json \
  --bvh datas/fz/dance_13.bvh \
  --bvh-scale 1.0 \
