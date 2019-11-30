#!/bin/zsh

for i in {1..5}
do
    python3 sac_bm.py --env 'PendulumEnv' --max_frames 10000 --frame_skip 4
    echo "trial $i out of 5"
done

