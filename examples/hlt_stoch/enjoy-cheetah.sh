#!/bin/sh

for seed in 42
do

    echo "trial $seed"
    python3 enjoy.py \
                --env "HalfCheetahBulletEnv" \
                --max_steps 2000 \
                --max_frames 40000 \
                --horizon 10 \
                --frame_skip 5 \
                --lam 0.1 \
                --seed $seed \
                --render \
                --record
done
