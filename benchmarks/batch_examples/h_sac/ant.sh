#!/bin/zsh

for i in {1..1}
do
    python3 h_sac.py \
                --env AntBulletEnv \
                --max_steps 400 \
                --max_frames 10000 \
                --horizon 40 \
                --frame_skip 1 \
                --lam 0.1 \
                --render
    echo "trial $i out of 2"
done
