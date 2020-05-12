#!/bin/sh

for seed in 0 1 42 666 1234
do

    echo "trial $seed"
    python3 hlt_stoch.py \
                --env "HalfCheetahBulletEnv" \
                --max_steps 2000 \
                --max_frames 40000 \
                --horizon 10 \
                --frame_skip 5 \
                --lam 0.1 \
                --seed $seed \
		--log \
                --no_render
done
