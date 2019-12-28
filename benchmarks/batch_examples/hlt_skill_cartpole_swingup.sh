#!/bin/zsh
trials=2
for i in {1..$trials}
do
    python3 hlt_skill.py \
            --env 'InvertedPendulumSwingupBulletEnv' \
            --max_frames 6000 \
            --frame_skip 4 \
            --render \
            --policy_lr 3e-3 \
            --horizon 10
    echo "trial $i out of $trials"
done
