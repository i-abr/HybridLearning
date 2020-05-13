#!/bin/sh

for seed in 0 1 42 666 1234
do
    echo "trial $seed"
    python3 sac.py \
                --env "AntBulletEnv" \
                --max_steps 2000 \
                --max_frames 100000 \
                --frame_skip 5 \
                --seed $seed \
		--reward_scale 10 \
		--log \
                --no_render
done
