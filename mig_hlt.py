import numpy as np
import random
import pickle
from datetime import datetime

import sys
import os
import yaml
import glob
# local imports
import envs
import gym
from gym import wrappers

import torch
from sac_lib import SoftActorCritic
from sac_lib import PolicyNetwork
from sac_lib import ReplayBuffer
from sac_lib import NormalizedActions
from hlt_lib import StochPolicyWrapper, DetPolicyWrapper
from model import ModelOptimizer, Model, SARSAReplayBuffer

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='InvertedPendulumEnv')
parser.add_argument('--done_util', dest='done_util', action='store_true')
parser.add_argument('--no_done_util', dest='done_util', action='store_false')
parser.set_defaults(done_util=True)
parser.add_argument('--render', dest='render', action='store_true')
parser.add_argument('--no_render', dest='render', action='store_false')
parser.set_defaults(render=False)

args = parser.parse_args()


if __name__ == '__main__':

    config_path = './config/hlt_deter.yaml'
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
        config = config_dict['default']
        if args.env in list(config_dict.keys()):
            config.update(config_dict[args.env])
        else:
            raise ValueError('env not found config file')

    env_name = args.env
    try:
        env = NormalizedActions(envs.env_list[env_name](render=args.render))
    except TypeError as err:
        print('no argument render, assumping env.render will just work')
        env = NormalizedActions(envs.env_list[env_name]())

    assert np.any(np.abs(env.action_space.low) <= 1.) and  np.any(np.abs(env.action_space.high) <= 1.), 'Action space not normalizd'


    action_dim = env.action_space.shape[0]
    state_dim  = env.observation_space.shape[0]
    hidden_dim = 128

    device ='cpu'
    if torch.cuda.is_available():
        device  = 'cuda:0'
        print('Using GPU Accel')

    policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim,AF=config['activation_fun']).to(device)
    model = Model(state_dim, action_dim, def_layers=[200],AF=config['activation_fun']).to(device)

    state_dict_path = './data/hlt_deter/' + env_name
    mig_log = []
    for seed_dir in glob.glob(state_dict_path + '/seed_*'):
        seed = int(seed_dir.split('_')[-1])
        env.reset()
        env.seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        if (args.env == 'PendulumEnv') or (args.env == 'InvertedPendulumBulletEnv'):
            frame_evals = [1000, 2000, 3000, 4000, 5000, 
                           6000, 7000, 8000, 9000, 10000]
        else:
            frame_evals = [5000, 10000, 20000, 30000, 40000, 50000]
        for _frame in frame_evals:
            policy_path = seed_dir + '/policy_{}.pt'.format(_frame)
            model_path = seed_dir + '/model_{}.pt'.format(_frame)
#             print(policy_path, model_path)
            policy_net.load_state_dict(torch.load(policy_path, map_location=device))
            model.load_state_dict(torch.load(model_path, map_location=device))

            hybrid_policy = DetPolicyWrapper(model, policy_net,
                                        T=config['horizon'],
                                        lr=config['planner_lr'])

            max_frames  = config['max_frames']
            max_steps   = config['max_steps']
            frame_skip  = config['frame_skip']

            state = env.reset()
            hybrid_policy.reset()

            episode_reward = 0
            done = False
            mig = []
            rho = 0.
            for step in range(max_steps):

                action, _rho = hybrid_policy(state)
                rho += _rho
                for _ in range(frame_skip):
                    state, _, _, _ = env.step(action.copy())

                if args.render:
                    env.render("human")

                if args.done_util:
                    if done:
                        break
            mig.append(rho/step)
            print(seed,rho/step)

            mig_log.append(mig)

        print('savig mig log for ' + seed_dir)
        pickle.dump(mig_log, open(state_dict_path + '/mig_log.pkl', 'wb'))
