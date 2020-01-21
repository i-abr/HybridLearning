#!/bin/zsh

python3 policy_test.py \
            --env 'InvertedPendulumSwingupBulletEnv' \
            --max_frames 10000 \
            --frame_skip 4 \
            --render
