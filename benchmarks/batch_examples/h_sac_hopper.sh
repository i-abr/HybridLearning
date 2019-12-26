#!/bin/zsh

for i in {1..2}
do
    python3 h_sac.py \
                --env "HopperBulletEnv" \
                --max_steps 400 \
                --max_frames 10000 \
                --horizon 5 \
                --frame_skip 4 \
                --no_render
    echo "trial $i out of 2"
done
