#!/bin/bash

for i in {1..2}
do
    python3 h_sac.py --env "HalfCheetahBulletEnv" --max_steps 200 --horizon 20 --max_frames 20000 --frame_skip 5 --render
    echo "trial $i out of 2"
done
