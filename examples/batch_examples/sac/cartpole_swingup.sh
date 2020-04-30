#!/bin/zsh

trials=2
for i in {1..$trials}
do
    python3 sac_bm.py \
                --env 'InvertedPendulumSwingupBulletEnv' \
                --max_frames 20000 \
                --frame_skip 4 \
                --max_steps 1000 \
                --no_render
    echo "trial $i out of $trials"
done
