#!/bin/sh

for env in 'AcrobotEnv' 'InvertedPendulumRoboschoolEnv' 'HopperEnv'  'HalfCheetahEnv'
do
    python3 mig_hlt.py --env $env
done
