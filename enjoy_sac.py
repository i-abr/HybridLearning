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

import torch
from sac_lib import SoftActorCritic
from sac_lib import PolicyNetwork
from sac_lib import ReplayBuffer
from sac_lib import NormalizedActions

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--env',   type=str,   default='InvertedPendulumBulletEnv')
parser.add_argument('--frame', type=int, default=-1)
parser.add_argument('--seed', type=int, default=666)
parser.add_argument('--done_util', dest='done_util', action='store_true')
parser.add_argument('--no_done_util', dest='done_util', action='store_false')
parser.set_defaults(done_util=True)
parser.add_argument('--render', dest='render', action='store_true')
parser.add_argument('--no_render', dest='render', action='store_false')
parser.set_defaults(render=True)
parser.add_argument('--record', dest='record', action='store_true')
parser.add_argument('--no_record', dest='record', action='store_false')
parser.set_defaults(record=False)

args = parser.parse_args()

import pybullet as pb

if __name__ == '__main__':

    config_path = './config/sac.yaml'
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
        print('no argument render,  assumping env.render will just work')
        env = NormalizedActions(envs.env_list[env_name]())
    assert np.any(np.abs(env.action_space.low) <= 1.) and  np.any(np.abs(env.action_space.high) <= 1.), 'Action space not normalizd'
    if args.record:

        env = gym.wrappers.Monitor(env, './data/vid/sac/{}-{}'.format(env_name, args.frame), force=True)
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

    policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)

    state_dict_path = './data/sac/' + env_name + '/seed_{}/'.format(args.seed)

    if args.frame == -1:
        test_frame = 'final'
    else:
        test_frame = args.frame

    policy_net.load_state_dict(torch.load(state_dict_path+'policy_{}.pt'.format(test_frame), map_location=device))

    max_frames  = config['max_frames']
    max_steps   = config['max_steps']
    frame_skip  = config['frame_skip']

    frame_idx   = 0
    rewards     = []

    ep_num = 0
    state = env.reset()

    episode_reward = 0
    done = False
    for step in range(max_steps):
        action = policy_net.get_action(state)
        for _ in range(frame_skip):
            state, reward, done, _ = env.step(action.copy())
            if done: break
        episode_reward += reward
        frame_idx += 1

        if args.render:
            env.render("human")

        if args.done_util:
            if done:
                break
    rewards.append([frame_idx, episode_reward])
    ep_num += 1
    env.close()
