#!/bin/zsh
trials=2
for i in {1..$trials}
do
    python3 h_sac.py --env 'LunarLanderContinuousEnv' --max_steps 100 --max_frames 6000 --frame_skip 2 --horizon 10 --render --model_lr 3e-3 --policy_lr 3e-4
    echo "trial $i out of $trials"
done
