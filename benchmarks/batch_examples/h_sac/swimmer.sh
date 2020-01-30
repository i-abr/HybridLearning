#!/bin/zsh

for i in {1..1}
do
    python3 h_sac.py \
                --env Swimmer-v3 \
                --max_steps 250 \
                --max_frames 10000 \
                --horizon 10 \
                --frame_skip 4 \
                --no_render
    echo "trial $i out of 2"
done
