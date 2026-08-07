# conda run -n gmr 
python tools/links_mapping/visualize_links_mapping.py \
  --xml assets/D20_29dof/mjcf/D20.xml \
  --ik_config general_motion_retargeting/ik_configs/bvh_qtong_to_d20.json \
  --bvh datas/qtong/kongfu/tachi_21.bvh \
  --human-height 1.8 \
