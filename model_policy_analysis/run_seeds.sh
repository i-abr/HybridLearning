#!/bin/sh


for env in 'HalfCheetahEnv'
# # for env in 'InvertedPendulumEnv' 'PendulumEnv' 'HalfCheetahEnv' 'HopperEnv'
do
    python3 model_test.py --env $env
    python3 policy_test.py --env $env
done
