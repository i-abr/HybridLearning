import numpy as np
import random
import pickle
from datetime import datetime

import sys
import os
import yaml

# local imports
import envs

import torch
from sac_lib import SoftActorCritic
from sac_lib import PolicyNetwork
from sac_lib import ReplayBuffer
from sac_lib import NormalizedActions
from hlt_lib import StochPolicyWrapper, DetPolicyWrapper
from model import ModelOptimizer, Model, SARSAReplayBuffer

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--env',   type=str,   default='InvertedPendulumBulletEnv')
parser.add_argument('--method', type=str, default='hlt_stoch')
parser.add_argument('--seed', type=int, default=666)
parser.add_argument('--done_util', dest='done_util', action='store_true')
parser.add_argument('--no_done_util', dest='done_util', action='store_false')
parser.set_defaults(done_util=True)
parser.add_argument('--log', dest='log', action='store_true')
parser.add_argument('--no-log', dest='log', action='store_false')
parser.set_defaults(log=False)
parser.add_argument('--render', dest='render', action='store_true')
parser.add_argument('--no_render', dest='render', action='store_false')
parser.set_defaults(render=False)

args = parser.parse_args()


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
        print('no argument render,  assumping env.render will just work')
        env = NormalizedActions(envs.env_list[env_name]())
    assert np.any(np.abs(env.action_space.low) <= 1.) and  np.any(np.abs(env.action_space.high) <= 1.), 'Action space not normalizd'
    env.reset()

    env.seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.log:
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d_%H-%M-%S/")
        dir_name = 'seed_{}/'.format(str(args.seed))
        path = './data/'  + config['method'] + '/' + env_name + '/' + dir_name
        if os.path.exists(path) is False:
            os.makedirs(path)

    action_dim = env.action_space.shape[0]
    state_dim  = env.observation_space.shape[0]
    hidden_dim = 128

    device ='cpu'
    if torch.cuda.is_available():
        device  = 'cuda:0'
        print('Using GPU Accel')

    policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)

    model = Model(state_dim, action_dim, def_layers=[200]).to(device)


    replay_buffer_size = 1000000
    replay_buffer = ReplayBuffer(replay_buffer_size)

    model_replay_buffer = SARSAReplayBuffer(replay_buffer_size)
    model_optim = ModelOptimizer(model, model_replay_buffer, lr=config['model_lr'])

    sac = SoftActorCritic(policy=policy_net,
                          state_dim=state_dim,
                          action_dim=action_dim,
                          replay_buffer=replay_buffer,
                          policy_lr=config['policy_lr'],
                          value_lr=config['value_lr'],
                          soft_q_lr=config['soft_q_lr'])
    if config['method'] == 'hlt_stoch':
        hybrid_policy = StochPolicyWrapper(model, policy_net,
                                samples=config['trajectory_samples'],
                                t_H=config['horizon'],
                                lam=config['lam'])
    elif config['method'] == 'hlt_deter':
        hybrid_policy = DetPolicyWrapper(model, policy_net, T=config['horizon'])
    else:
        ValueError('method not found in config')

    max_frames  = config['max_frames']
    max_steps   = config['max_steps']
    frame_skip  = config['frame_skip']
    reward_scale = config['reward_scale']
    frame_idx   = 0
    rewards     = []
    batch_size  = 256

    ep_num = 0
    while frame_idx < max_frames:
        state = env.reset()
        hybrid_policy.reset()

        action = hybrid_policy(state)

        episode_reward = 0
        done = False
        for step in range(max_steps):
            for _ in range(frame_skip):
                next_state, reward, done, _ = env.step(action.copy())

            next_action = hybrid_policy(next_state)

            replay_buffer.push(state, action, reward, next_state, done)
            model_replay_buffer.push(state, action, reward_scale * reward, next_state, next_action, done)

            if len(replay_buffer) > batch_size:
                sac.update(batch_size)
                model_optim.update_model(batch_size, mini_iter=config['model_iter'])

            state = next_state
            action = next_action
            episode_reward += reward
            frame_idx += 1

            if args.render:
                env.render("human")


            if frame_idx % (max_frames//10) == 0:
                last_reward = rewards[-1][1] if len(rewards)>0 else 0
                print(
                    'frame : {}/{}, \t last rew: {}'.format(
                        frame_idx, max_frames, last_reward
                    )
                )
                if args.log:
                    print('saving model and reward log')
                    pickle.dump(rewards, open(path + 'reward_data' + '.pkl', 'wb'))
                    torch.save(policy_net.state_dict(), path + 'policy_' + str(frame_idx) + '.pt')
                    torch.save(model.state_dict(), path + 'model_' + str(frame_idx) + '.pt')

            if args.done_util:
                if done:
                    break
        if len(replay_buffer) > batch_size:
            print('ep rew', ep_num, episode_reward)
        rewards.append([frame_idx, episode_reward])
        ep_num += 1
    if args.log:
        print('saving final data set')
        pickle.dump(rewards, open(path + 'reward_data'+ '.pkl', 'wb'))
        torch.save(policy_net.state_dict(), path + 'policy_' + 'final' + '.pt')
        torch.save(model.state_dict(), path + 'model_' + 'final' + '.pt')
