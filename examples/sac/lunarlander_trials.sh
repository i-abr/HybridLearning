#!/bin/zsh

for i in {1..5}
do
    python3 hybrid_sac.py --env 'LunarLanderContinuousEnv' --policy_lr 3e-3 --horizon 10 --model_iter 1 --max_steps 150 --max_frames 10000 --frame_skip 2 --render true
    echo "trial $i out of 5"
done

