#!/bin/bash

for i in {1..2}
do
    python3 h_sac.py --env "HalfCheetahBulletEnv" --max_steps 200 --max_frames 200000 --frame_skip 1 --render
    echo "trial $i out of 2"
done
