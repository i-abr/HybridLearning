#!/bin/zsh

for i in {1..2}
do
    python3 sac_bm.py \
            --env "HopperBulletEnv" \
            --max_steps 1000 \
            --max_frames 40000 \
            --frame_skip 4 \
            --no_render
    echo "trial $i out of 2"
done
