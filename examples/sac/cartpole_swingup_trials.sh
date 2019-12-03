#!/bin/zsh

for i in {1..2}
do
    python3 hybrid_sac.py --env 'InvertedPendulumSwingupBulletEnv' --max_frames 10000 --frame_skip 4 \
        --render
    echo "trial $i out of 5"
done

