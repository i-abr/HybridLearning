#!/bin/sh

for seed in 0 1 42 666 1234
do
    echo "trial $seed"
    python3 hlt_deter.py \
                --env 'InvertedPendulumSwingupBulletEnv' \
                --max_frames 10000 \
                --frame_skip 4 \
                --max_steps 1000 \
                --model_iter 5 \
                --seed $seed \
		--log \
                --no_render
done
