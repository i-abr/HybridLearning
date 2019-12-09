#!/bin/zsh

trials=2
for i in {1..$trials}
do
    python3 ddpg_bm.py \
                --env 'InvertedPendulumSwingupBulletEnv' \
                --max_frames 10000 \
                --frame_skip 4 \
                --render \
                --model_iter 3e-3
    echo "trial $i out of $trials"
done
