#!/bin/bash

for i in {1..2}
do
    python3 sac_bm.py --env "HopperBulletEnv" --max_steps 200 --max_frames 10000 --frame_skip 4 --render
    echo "trial $i out of 2"
done
