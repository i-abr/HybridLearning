#!/bin/zsh

trials=2
for i in {1..$trials}
do
    python3 hd_sac.py \
                --env 'InvertedPendulumSwingupBulletEnv' \
                --max_frames 10000 \
                --frame_skip 4 \
                --render
    echo "trial $i out of $trials"
done