#!/bin/zsh
trials=2
for i in {1..$trials}
do
    python3 hd_sac.py --env 'LunarLanderContinuousEnv' --max_steps 200 --max_frames 20000 --frame_skip 1 --horizon 12 --no_render --model_lr 3e-4 --policy_lr 3e-4
    echo "trial $i out of $trials"
done