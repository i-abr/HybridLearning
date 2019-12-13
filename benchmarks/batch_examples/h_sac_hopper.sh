#!/bin/zsh 

for i in {1..2}
do
    python3 h_sac.py --env "HopperBulletEnv" --max_steps 200 --max_frames 80000 --horizon 5 --frame_skip 2 --render
    echo "trial $i out of 2"
done
