import numpy as np
import random
import pickle
from datetime import datetime

import sys
import os
import yaml

# local imports
import envs
import gym
from gym import wrappers
from envs import Monitor

import torch
from sac_lib import SoftActorCritic
from sac_lib import PolicyNetwork
from sac_lib import ReplayBuffer
from sac_lib import NormalizedActions
from hlt_lib import StochPolicyWrapper, DetPolicyWrapper
from model import ModelOptimizer, Model, SARSAReplayBuffer

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--env',   type=str,   default='HopperEnv')
parser.add_argument('--method', type=str, default='hlt_stoch')
parser.add_argument('--frame', type=int, default=-1)
parser.add_argument('--seed', type=int, default=13)
parser.add_argument('--done_util', dest='done_util', action='store_true')
parser.add_argument('--no_done_util', dest='done_util', action='store_false')
parser.set_defaults(done_util=False)
parser.add_argument('--render', dest='render', action='store_true')
parser.add_argument('--no_render', dest='render', action='store_false')
parser.set_defaults(render=True)
parser.add_argument('--record', dest='record', action='store_true')
parser.add_argument('--no_record', dest='record', action='store_false')
parser.set_defaults(record=False)

args = parser.parse_args()

import pybullet as pb

if __name__ == '__main__':

    config_path = './config/' + args.method + '.yaml'
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
#     if args.env == 'PendulumEnv':
#         assert env.action_space.low == -env.action_space.high, 'Action space not symmetric'
#     else:
    assert np.any(np.abs(env.action_space.low) <= 1.) and  np.any(np.abs(env.action_space.high) <= 1.), 'Action space not normalizd'
    if args.record:
        if args.render:
            raise ValueError('cannot record while rendering, valid options are --render --no_record OR --no_record --render')
        video_path = './data/vid/hlt'
        if os.path.exists(video_path) == False:
            os.makedirs(video_path)
        if args.done_util:
            env = gym.wrappers.Monitor(env, video_path+'/{}-{}'.format(env_name, args.frame), force=True)
        else:
            env = Monitor(env, video_path+'/{}-{}'.format(env_name, args.frame), force=True)
        env.reset()

    # pb.configureDebugVisualizer(pb.STATE_LOGGING_VIDEO_MP4)

    env.seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    action_dim = env.action_space.shape[0]
    state_dim  = env.observation_space.shape[0]
    hidden_dim = 128

    device ='cpu'
    if torch.cuda.is_available():
        device  = 'cuda:0'
        print('Using GPU Accel')

    policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim,AF=config['activation_fun']).to(device)
    model = Model(state_dim, action_dim, def_layers=[200],AF=config['activation_fun']).to(device)

    state_dict_path = './data/' + args.method + '/' + env_name + '/seed_{}/'.format(args.seed)

    if args.frame == -1:
        test_frame = 'final'
    else:
        test_frame = args.frame

    policy_net.load_state_dict(torch.load(state_dict_path+'policy_{}.pt'.format(test_frame), map_location=device))
    model.load_state_dict(torch.load(state_dict_path+'model_{}.pt'.format(test_frame), map_location=device))

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

    max_frames  = config['max_frames']
    max_steps   = config['max_steps']
    frame_skip  = config['frame_skip']

    state = env.reset()
    hybrid_policy.reset()

    episode_reward = 0
    done = False
    for step in range(max_steps):
        action,_ = hybrid_policy(state)
        for _ in range(frame_skip):
            state, reward, done, _ = env.step(action.copy())
            if args.done_util:
                if done: break
        episode_reward += reward

        if args.render:
            try:
#                 env.render(mode="rgb_array", width=320*2, height=240*2)
                env.render(mode="human")
            except TypeError as err:
                env.render()

        if args.done_util:
            if done:
                break
    # print(episode_reward)
    print(step)
    env.close()
