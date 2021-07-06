#!/bin/sh

for env in 'AcrobotEnv' 'InvertedPendulumRoboschoolEnv' 'HopperEnv'  'HalfCheetahEnv'
# for env in 'PendulumEnv' 'InvertedPendulumBulletEnv'  'PendubotEnv'
do
    for i in $(seq 13 100 950)
    do
        for method in 'hlt_deter' 'mpc_deter' 'sac__' 'hlt_stoch' 'mpc_stoch'
        do
            echo $env $i $method
            python3 train_hlt.py --seed $i --env $env --method $method
        done
    done
done
