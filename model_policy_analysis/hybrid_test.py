import numpy as np
import random
import pickle
from datetime import datetime

import sys
import os
import yaml
sys.path.append('../')

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
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--env',   type=str,   default='InvertedPendulumEnv')
parser.add_argument('--frame', type=int, default=-1)
# parser.add_argument('--seed', type=int, default=13)
parser.add_argument('--done_util', dest='done_util', action='store_true')
parser.add_argument('--no_done_util', dest='done_util', action='store_false')
parser.set_defaults(done_util=True)
parser.add_argument('--render', dest='render', action='store_true')
parser.add_argument('--no_render', dest='render', action='store_false')
parser.set_defaults(render=False)
parser.add_argument('--method', type=str, default='hlt_stoch')

args = parser.parse_args()
print(args)

if __name__ == '__main__':

    config_path = '../config/' + args.method + '.yaml'
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
        print('no argument render,  assuming env.render will just work')
        env = NormalizedActions(envs.env_list[env_name]())
    assert np.any(np.abs(env.action_space.low) <= 1.) and  np.any(np.abs(env.action_space.high) <= 1.), 'Action space not normalizd'

    action_dim = env.action_space.shape[0]
    state_dim  = env.observation_space.shape[0]
    hidden_dim = 128

    device ='cpu'
    if torch.cuda.is_available():
        device  = 'cuda:0'
        print('Using GPU Accel')


    max_frames  = config['max_frames']
    max_steps   = config['max_steps']
    frame_skip  = config['frame_skip']

    trials     = []
    state_dict_path = '../data/'+ args.method +'/' + env_name
#     state_dict_path = './data/' + args.method + '/' + env_name + '/seed_{}/'.format(args.seed)

    for seed_dir in glob.glob(state_dict_path + '/seed_*'):
        seed = int(seed_dir.split('_')[-1])
        print(env_name, seed)
        env.reset()
        env.seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        rewards = []
        if args.frame == -1:
            m_path = seed_dir +'/model_final.pt'
            p_path = seed_dir +'/policy_final.pt'

            policy_net = PolicyNetwork(state_dim, action_dim,
                                       hidden_dim,AF=config['activation_fun']).to(device)
            model = Model(state_dim, action_dim,
                          def_layers=[200],AF=config['activation_fun']).to(device)
            policy_net.load_state_dict(torch.load(p_path, map_location=device))
            model.load_state_dict(torch.load(m_path, map_location=device))

            if config['method'] == 'hlt_stoch':
                hybrid_policy = StochPolicyWrapper(model, policy_net,
                                        samples=config['trajectory_samples'],
                                        t_H=config['horizon'],
                                        lam=config['lam'])
            elif config['method'] == 'hlt_deter':
                hybrid_policy = DetPolicyWrapper(model, policy_net,
                                            T=config['horizon'],
                                            lr=config['planner_lr'])
            else:
                raise ValueError('method not found in config')

            for _ in range(10):
                state = env.reset()
                hybrid_policy.reset()

                action,_ = hybrid_policy(state)

                episode_reward = 0
                done = False
                for step in range(max_steps):
                    for _ in range(frame_skip):
                        next_state, reward, done, _ = env.step(action.copy())

                    next_action,_ = hybrid_policy(next_state)

                    state = next_state
                    action = next_action
                    episode_reward += reward

                    if args.render:
                        env.render(mode="human")

                    if args.done_util:
                        if done:
                            break
                rewards.append(episode_reward)
        else:
            model_paths = glob.glob(seed_dir +'/model_*')
            policy_paths = glob.glob(seed_dir +'/policy_*')
            for m_path,p_path in zip(model_paths,policy_paths):
                final = False
                try:
                    frame_idx = int(m_path.split('model_')[-1].split('.pt')[0])
                except:
                    final = True
                if not final:
                    policy_net = PolicyNetwork(state_dim, action_dim,
                                               hidden_dim,AF=config['activation_fun']).to(device)
                    model = Model(state_dim, action_dim,
                                  def_layers=[200],AF=config['activation_fun']).to(device)
                    policy_net.load_state_dict(torch.load(p_path, map_location=device))
                    model.load_state_dict(torch.load(m_path, map_location=device))

                    if config['method'] == 'hlt_stoch':
                        hybrid_policy = StochPolicyWrapper(model, policy_net,
                                                samples=config['trajectory_samples'],
                                                t_H=config['horizon'],
                                                lam=config['lam'])
                    elif config['method'] == 'hlt_deter':
                        hybrid_policy = DetPolicyWrapper(model, policy_net,
                                                    T=config['horizon'],
                                                    lr=config['planner_lr'])
                    else:
                        raise ValueError('method not found in config')


                    state = env.reset()
                    hybrid_policy.reset()

                    action,_ = hybrid_policy(state)

                    episode_reward = 0
                    done = False
                    for step in range(max_steps):
                        for _ in range(frame_skip):
                            next_state, reward, done, _ = env.step(action.copy())

                        next_action,_ = hybrid_policy(next_state)

                        state = next_state
                        action = next_action
                        episode_reward += reward

                        if args.render:
                            env.render(mode="human")

                        if args.done_util:
                            if done:
                                break
                    rewards.append(episode_reward)
            rewards = sorted(rewards, key=lambda x: x[0])
            rewards = np.array(rewards)
        trials.append(rewards)
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d_%H-%M-%S/")

    path = '../data/'+args.method+'/' + env_name +  '/' + 'hybrid_test/'
    if os.path.exists(path) is False:
        os.makedirs(path)

    print('saving final data set')
    if args.frame == -1:
        pickle.dump(trials, open(path + 'reward_data_final'+ '.pkl', 'wb'))
    else:
        pickle.dump(trials, open(path + 'reward_data'+ '.pkl', 'wb'))
