#!/bin/zsh

for i in {1..5}
do
    python3 sac_bm.py --env 'LunarLanderContinuousEnv' --max_frames 20000 --frame_skip 2
    echo "trial $i out of 5"
done

