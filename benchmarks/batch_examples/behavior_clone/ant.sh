#!/bin/zsh

for i in {1..5}
do
    python3 behavior_cloning.py \
                --env AntBulletEnv \
                --max_steps 400 \
                --max_frames 6000 \
                --horizon 20 \
                --frame_skip 1 \
                --lam 1.0 \
                --no_render
    echo "trial $i out of 2"
done
