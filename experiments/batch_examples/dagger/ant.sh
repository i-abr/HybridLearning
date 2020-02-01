#!/bin/zsh

for i in {1..1}
do
    python3 dagger.py \
                --env AntBulletEnv \
                --max_steps 400 \
                --max_frames 6000 \
                --horizon 20 \
                --frame_skip 1 \
                --lam 1.0 \
                --render
    echo "trial $i out of 2"
done
