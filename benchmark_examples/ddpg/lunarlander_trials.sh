#!/bin/zsh

for i in {1..2}
do
    python3 ddpg_bm.py --env 'LunarLanderContinuousEnv' --policy_lr 3e-3 --max_steps 150 --max_frames 10000 --frame_skip 2 --render
    echo "trial $i out of 2"
done

