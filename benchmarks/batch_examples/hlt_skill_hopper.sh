#!/bin/zsh
trials=2
for i in {1..$trials}
do
    python3 hlt_skill.py \
            --env 'HopperBulletEnv' \
            --max_steps 1000 \
            --max_frames 10000 \
            --frame_skip 4 \
            --no_render \
            --horizon 10
    echo "trial $i out of $trials"
done
