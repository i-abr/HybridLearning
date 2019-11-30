#!/bin/zsh

for i in {1..5}
do
    python3 sac_bm.py --env 'HopperBulletEnv' --policy_lr 3e-4 --max_steps 200 --max_frames 20000 --frame_skip 4
    echo "trial $i out of 5"
done

