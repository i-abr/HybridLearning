#!/bin/zsh

for i in {1..2}
do
    python3 hybrid_sac.py \
        --env 'PendulumEnv' \
        --max_frames 6000 \
        --frame_skip 4 \
        --render
    echo "trial $i out of 5"
done
