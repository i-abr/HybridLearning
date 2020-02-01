#!/bin/zsh

for i in {1..1}
do
    python3 h_sac.py \
                --env CubeManipEnv \
                --max_steps 150 \
                --max_frames 80000 \
                --horizon 40 \
                --frame_skip 1 \
                --trajectory_samples 20 \
                --model_lr 0.01 \
                --policy_lr 0.003 \
                --lam 0.2 \
                --render
    echo "trial $i out of 2"
done
