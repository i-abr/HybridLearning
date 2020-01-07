#!/bin/bash

for i in {1..2}
do
    python3 sac_bm.py --env "HalfCheetahBulletEnv" --max_steps 200 --max_frames 200000 --frame_skip 1 --render
    echo "trial $i out of 2"
done
