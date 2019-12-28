#!/bin/zsh
trials=2
for i in {1..$trials}
do
    python3 hlt_skill.py \
                --env 'PendulumEnv' \
                --max_frames 6000 \
                --frame_skip 4 \
                --render \
                --model_iter 2
    echo "trial $i out of $trials"
done