#!/bin/zsh

for i in {1..1}
do
    python3 h_sac.py \
                --env CubeManipEnv \
                --max_steps 100 \
                --max_frames 10000 \
                --horizon 5 \
                --frame_skip 2 \
                --model_lr 3e-4 \
                --lam 0.9 \
                --render
    echo "trial $i out of 2"
done
