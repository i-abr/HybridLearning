#!/bin/zsh

trials=2
for i in {1..$trials}
do
    python3 h_sac.py \
                --env 'InvertedPendulumSwingupBulletEnv' \
                --max_frames 10000 \
                --frame_skip 4 \
                --max_steps 1000 \
                --lam 0.1 \
                --model_iter 5 \
                --render
    echo "trial $i out of $trials"
done
