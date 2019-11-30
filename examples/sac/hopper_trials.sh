#!/bin/zsh

for i in {1..5}
do
    python3 hybrid_sac.py --env 'HopperBulletEnv' --policy_lr 3e-4 --model_lr 3e-4 --horizon 10 --model_iter 1 --max_steps 200 --max_frames 20000 --frame_skip 4 
    echo "trial $i out of 5"
done

