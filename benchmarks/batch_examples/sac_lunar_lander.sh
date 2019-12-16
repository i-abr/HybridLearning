#!/bin/zsh
trials=2
for i in {1..$trials}
do
    python3 sac_bm.py --env 'LunarLanderContinuousEnv' --max_steps 100 --policy_lr 3e-4 --max_frames 10000 --frame_skip 2 --render
    echo "trial $i out of $trials"
done
