#!/bin/zsh

for i in {1..1}
do
    python3 stoch_model_based_learning.py \
                --env Reacher3d \
                --max_steps 150 \
                --max_frames 15000 \
                --horizon 10 \
                --frame_skip 1 \
                --model_lr 3e-3 \
                --render
    echo "trial $i out of 2"
done
