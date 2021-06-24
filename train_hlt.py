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
from mpc_lib import ModelBasedDeterControl, PathIntegral

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--env',   type=str,   default='InvertedPendulumRoboschoolEnv')
parser.add_argument('--method', type=str, default='hlt_stoch')
parser.add_argument('--seed', type=int, default=666)
parser.add_argument('--done_util', dest='done_util', action='store_true')
parser.add_argument('--no_done_util', dest='done_util', action='store_false')
parser.set_defaults(done_util=True)
parser.add_argument('--log', dest='log', action='store_true')
parser.add_argument('--no-log', dest='log', action='store_false')
parser.set_defaults(log=True)
parser.add_argument('--render', dest='render', action='store_true')
parser.add_argument('--no_render', dest='render', action='store_false')
parser.set_defaults(render=False)

args = parser.parse_args()
print(args)

if __name__ == '__main__':
    base_method = args.method[:3]
    if base_method == 'sac__':
        config_path = './config/sac.yaml'
    elif args.method[4:] == 'deter':
        config_path = './config/hlt_deter.yaml'
    else:
        config_path = './config/hlt_stoch.yaml'

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
    if args.render:
        try:
            env.render() # needed for InvertedDoublePendulumBulletEnv
        except:
            print('render not needed')
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
        path = './data/'  + args.method + '/' + env_name + '/' + dir_name
        if os.path.exists(path) == False:
            os.makedirs(path)

    action_dim = env.action_space.shape[0]
    state_dim  = env.observation_space.shape[0]
    hidden_dim = 128
    replay_buffer_size = 1000000
    replay_buffer = [None] # placeholder
    model_replay_buffer = [None] # placeholder

    device ='cpu'
    if torch.cuda.is_available():
        device  = 'cuda:0'
        print('Using GPU Accel')

    if base_method != 'sac':
        model = Model(state_dim, action_dim, def_layers=[200],AF=config['activation_fun']).to(device)
        model_replay_buffer = SARSAReplayBuffer(replay_buffer_size)
        model_optim = ModelOptimizer(model, model_replay_buffer, lr=config['model_lr'])

    if base_method != 'mpc':
        policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim,AF=config['activation_fun']).to(device)
        replay_buffer = ReplayBuffer(replay_buffer_size)
        sac = SoftActorCritic(policy=policy_net,
                              state_dim=state_dim,
                              action_dim=action_dim,
                              replay_buffer=replay_buffer,
                              policy_lr=config['policy_lr'],
                              value_lr=config['value_lr'],
                              soft_q_lr=config['soft_q_lr'])

    if args.method == 'hlt_stoch':
        hybrid_policy = StochPolicyWrapper(model, policy_net,
                                samples=config['trajectory_samples'],
                                t_H=config['horizon'],
                                lam=config['lam'])
    elif args.method == 'hlt_deter':
        hybrid_policy = DetPolicyWrapper(model, policy_net,
                                        T=config['horizon'],
                                        lr=config['planner_lr'])
    elif args.method == 'mpc_stoch':
        planner = PathIntegral(model,
                               samples=config['trajectory_samples'],
                               t_H=config['horizon'],
                               lam=config['lam'])
    elif args.method == 'mpc_deter':
        planner = ModelBasedDeterControl(model, T=config['horizon'])
    elif base_method == 'sac':
        pass
    else:
        raise ValueError('method not found')

    max_frames  = config['max_frames']
    max_steps   = config['max_steps']
    frame_skip  = config['frame_skip']
    reward_scale = config['reward_scale']

    frame_idx   = 0
    rewards     = []
    batch_size  = 128

    ep_num = 0
    while frame_idx < max_frames:
        state = env.reset()
        if base_method == 'sac':
            action = policy_net.get_action(state)
        elif base_method == 'mpc':
            planner.reset()
            action, _ = planner(state)
        else:
            hybrid_policy.reset()
            action, _ = hybrid_policy(state)

        episode_reward = 0
        done = False
        for step in range(max_steps):
            for _ in range(frame_skip):
                next_state, reward, done, _ = env.step(action.copy())

            if base_method == 'sac':
                next_action = policy_net.get_action(state)
                replay_buffer.push(state, action, reward, next_state, done)
                if len(replay_buffer) > batch_size:
                    sac.update(batch_size)
            elif base_method == 'mpc':
                next_action, _ = planner(next_state)
                model_replay_buffer.push(state, action, reward_scale * reward, next_state, next_action, done)
                if args.method == 'mpc_deter':
                    # print(step,next_action)
                    next_action += np.random.normal(0., 1.0*(0.999**(frame_idx+1)), size=(action_dim,))
                    # print(step,next_action)
                if len(model_replay_buffer) > batch_size:
                    model_optim.update_model(batch_size, mini_iter=config['model_iter'])
            elif base_method == 'hlt':
                next_action, rho = hybrid_policy(next_state)
                if args.method == 'hlt_deter':
                    next_action += np.random.normal(0., 1.0*(0.999**(frame_idx+1)), size=(action_dim,))
                model_replay_buffer.push(state, action, reward_scale * reward, next_state, next_action, done)
                replay_buffer.push(state, action, reward, next_state, done)
                if len(replay_buffer) > batch_size:
                    sac.update(batch_size)
                    model_optim.update_model(batch_size, mini_iter=config['model_iter'])

            state = next_state
            action = next_action
            episode_reward += reward
            frame_idx += 1

            if args.render:
                env.render(mode="human")


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
                    if base_method != 'mpc':
                        torch.save(policy_net.state_dict(), path + 'policy_' + str(frame_idx) + '.pt')
                    if base_method != 'sac':
                        torch.save(model.state_dict(), path + 'model_' + str(frame_idx) + '.pt')
#             print(episode_reward,done)
            if args.done_util:
                if done:
                    break
        if (len(replay_buffer) > batch_size) or (len(model_replay_buffer) > batch_size):
            print('ep rew', ep_num, episode_reward, frame_idx)
        rewards.append([frame_idx, episode_reward,ep_num])
        ep_num += 1
    env.close()
    if args.log:
        print('saving final data set')
        pickle.dump(rewards, open(path + 'reward_data'+ '.pkl', 'wb'))
        if base_method != 'mpc':
            torch.save(policy_net.state_dict(), path + 'policy_' + 'final' + '.pt')
        if base_method != 'sac':
            torch.save(model.state_dict(), path + 'model_' + 'final' + '.pt')
