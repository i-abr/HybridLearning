#!/bin/zsh

trials=2
for i in {1..$trials}
do
    python3 hybrid_ddpg.py \
                --env 'InvertedPendulumSwingupBulletEnv' \
                --max_frames 10000 \
                --frame_skip 4 \
                --render
                --model_lr 3e-4
    echo "trial $i out of $trials"
done
