python tools/motion_speed_warp/warp_speed.py \
    outputs/d20_25dof/qt_dance_2.csv \
    --fps 120 --start-frame 1140 --end-frame 3595 --factor 0.8 \
    -o outputs/d20_25dof/dance_2_slow_08.csv

python tools/motion_speed_warp/warp_speed.py \
    outputs/d20_25dof/qt_dance_2.csv \
    --fps 120 --start-frame 1140 --end-frame 3595 --factor 0.9 \
    -o outputs/d20_25dof/dance_2_slow_09.csv
