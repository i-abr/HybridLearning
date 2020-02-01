#!/bin/zsh

for i in {1..1}
do
    python3 stoch_model_based_learning.py \
                --env Pusher \
                --max_steps 100 \
                --max_frames 15000 \
                --horizon 40 \
                --frame_skip 1 \
                --trajectory_samples 40 \
                --model_lr 0.01\
                --render
    echo "trial $i out of 2"
done
