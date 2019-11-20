#!/bin/zsh

for i in {1..5}
do
    python3 hybrid_sac.py --env 'LunarLanderContinuousEnv' --max_steps 400 --max_frames 10000 --frame_skip 2
    echo "trial $i out of 5"
done

