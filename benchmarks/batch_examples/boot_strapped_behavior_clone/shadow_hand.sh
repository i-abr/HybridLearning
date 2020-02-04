#!/bin/zsh

for i in {1..1}
do
    python3 shadow_hand/boot_strapped_behavior_cloning.py \
                --env CubeManipEnv \
                --max_steps 200 \
                --max_frames 10000 \
                --horizon 30 \
                --trajectory_samples 20 \
                --frame_skip 1 \
                --lam 1.0 \
                --render
    echo "trial $i out of 2"
done
