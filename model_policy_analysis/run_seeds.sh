#!/bin/sh

# test final hybrid method, model and policy (separately and combined)
for env in 'AcrobotEnv' 'InvertedPendulumRoboschoolEnv' 'HalfCheetahEnv' 'HopperEnv'
do
    for method in 'hlt_stoch' 'hlt_deter'
    do
        echo $env $method
        python3 model_test.py --env $env --method $method
        python3 policy_test.py --env $env --method $method
        python3 hybrid_test.py --env $env --method $method
    done
done


# test final model-based
for env in 'AcrobotEnv' 'InvertedPendulumRoboschoolEnv' 'HalfCheetahEnv' 'HopperEnv'
do
    for method in 'mpc_stoch' 'mpc_deter'
    do
        echo $env $method
        python3 model_test.py --env $env --method $method
    done
done


# test final model-free
for env in 'AcrobotEnv' 'InvertedPendulumRoboschoolEnv' 'HalfCheetahEnv' 'HopperEnv'
do
    echo $env
    python3 policy_test.py --env $env --method 'sac__'
done
