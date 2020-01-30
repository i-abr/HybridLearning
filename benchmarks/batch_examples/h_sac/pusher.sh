#!/bin/zsh

for i in {1..1}
do
    python3 h_sac.py \
                --env Pusher \
                --max_steps 50 \
                --max_frames 15000 \
                --horizon 10 \
                --frame_skip 2 \
                --lam 0.8 \
                --render
    echo "trial $i out of 2"
done
