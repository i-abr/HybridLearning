#!/bin/zsh

for i in {1..1}
do
    python3 h_sac.py \
                --env "HopperBulletEnv" \
                --max_steps 1000 \
                --max_frames 80000 \
                --horizon 10 \
                --frame_skip 4 \
                --no_render
    echo "trial $i out of 2"
done
