#!/bin/zsh

for i in {1..2}
do
    python3 ddpg_bm.py \
            --env "HopperBulletEnv" \
            --max_steps 400 \
            --max_frames 10000 \
            --frame_skip 4 \
            --no_render \
            --value_lr 3e-4 \
            --policy_lr 3e-4
    echo "trial $i out of 2"
done
