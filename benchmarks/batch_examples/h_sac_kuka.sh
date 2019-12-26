#!/bin/zsh

for i in {1..2}
do
    python3 h_sac.py --env "KukaEnv" --max_steps 200 --max_frames 10000 --model_lr 3e-4 --horizon 20 --frame_skip 1 --no_render
    echo "trial $i out of 2"
done
