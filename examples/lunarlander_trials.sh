#!/bin/zsh

for i in {1..5}
do
    python3 hybrid_sac.py --env 'LunarLanderContinuous' --max_frames 10000 --frame_skip 2
    echo "trial $i out of 5"
done

