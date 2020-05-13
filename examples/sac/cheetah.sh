#!/bin/sh

for seed in 0 1 42 666 1234
do
    echo "trial $seed"
    python3 sac.py \
                --env "HalfCheetahBulletEnv" \
                --max_steps 2000 \
                --max_frames 80000 \
                --frame_skip 5 \
                --seed $seed \
		--log \
                --no_render
done
