#!/bin/zsh

for i in {1..1}
do
    python3 h_sac.py \
                --env Striker \
                --max_steps 20 \
                --max_frames 80000 \
                --horizon 5 \
                --frame_skip 4 \
                --render
    echo "trial $i out of 2"
done
