#!/bin/zsh

for i in {1..1}
do
    python3 h_sac.py \
                --env Reacher3d \
                --max_steps 150 \
                --max_frames 15000 \
                --horizon 10 \
                --frame_skip 2 \
                --model_lr 3e-3 \
                --render
    echo "trial $i out of 2"
done
