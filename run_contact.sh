python scripts/hiphi_to_robot.py \
    --clip Change_posture-sit_down_0002 \
    --ik-config general_motion_retargeting/ik_configs/hiphi_sit.json \
    --object-contact-cost 20 \
    --object-contact-threshold 0.10 \
    --object-contact-max-points 128 \
    --export-dir data/SitChair \
    --export-slot 3
