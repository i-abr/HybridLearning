#!/bin/sh

for seed in 0 1 42 666 1234
do

    echo "trial $seed"
    python3 enjoy.py \
                --env "AntBulletEnv" \
                --max_steps 2000 \
                --max_frames 80000 \
                --horizon 10 \
                --frame_skip 5 \
                --lam 0.1 \
		        --reward_scale 10 \
                --seed $seed \
                --render
done
