#!/bin/zsh

for i in {1..1}
do
    python3 h_sac.py \
                --env Reacher3d \
                --max_steps 100 \
                --max_frames 15000 \
                --horizon 10 \
                --frame_skip 1 \
                --render
    echo "trial $i out of 2"
done
