#! /usr/bin/env python
"""
imports
"""
# general
import numpy as np
import pickle
from datetime import datetime

import sys
import os
import signal
import traceback
sys.path.append('../')

import argparse
from copy import copy, deepcopy
import time

# model
import torch

from saclib import SoftActorCritic, PolicyNetwork, ReplayBuffer

from hltlib import PathIntegral, ModelOptimizer, Model, SARSAReplayBuffer

# ros
import rospy
# from sawyer_reacher import sawyer_env # reacher
from sawyer_pusher import sawyer_env # pusher
# from sawyer_shapesorter import sawyer_env # shapesorter
# from sawyer_env_ShapeSorter import sawyer_env # shape sorter

"""
arg parse things
"""
parser = argparse.ArgumentParser()
# parser.add_argument('--env',        type=str,   help=envs.getlist())
parser.add_argument('--max_steps',  type=int,   default=100)
parser.add_argument('--max_frames', type=int,   default=6000)
# parser.add_argument('--frame_skip', type=int,   default=2)
parser.add_argument('--model_lr',   type=float, default=3e-3)
parser.add_argument('--policy_lr',  type=float, default=3e-3)
parser.add_argument('--value_lr',   type=float, default=3e-4)
parser.add_argument('--soft_q_lr',  type=float, default=3e-4)

parser.add_argument('--horizon', type=int, default=10)
parser.add_argument('--model_iter', type=int, default=2)
parser.add_argument('--trajectory_samples', type=int, default=20)
parser.add_argument('--lam',  type=float, default=0.01)


args = parser.parse_args()

if __name__ == '__main__':
    try:
        rospy.init_node('h_sac')

        env = sawyer_env()

        env_name = 'sawyer'
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d_%H-%M-%S_pusher/")

        path = './data/' + env_name +  '/' + 'h_sac/' + date_str
        # path = './data/' + env_name +  '/' + 'sac/' + date_str # policy only
        if os.path.exists(path) is False:
            os.makedirs(path)

        action_dim = 2 # 2 (reacher), 2 (pusher), 3 (shape sorter)
        state_dim  = 4 # 2 (reacher), 4 (pusher), 9 (shape sorter)
        hidden_dim = 128

        policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim)

        model = Model(state_dim, action_dim, def_layers=[200])

        replay_buffer_size = 10000
        replay_buffer = ReplayBuffer(replay_buffer_size)

        model_replay_buffer = SARSAReplayBuffer(replay_buffer_size)
        model_optim = ModelOptimizer(model, model_replay_buffer, lr=args.model_lr, lam=0.)

        sac = SoftActorCritic(policy=policy_net,
                              state_dim=state_dim,
                              action_dim=action_dim,
                              replay_buffer=replay_buffer,
                              policy_lr=args.policy_lr,
                              value_lr=args.value_lr,
                              soft_q_lr=args.soft_q_lr)

        planner = PathIntegral(model, policy_net, samples=args.trajectory_samples, t_H=args.horizon, lam=args.lam)

        max_frames = args.max_frames
        max_steps  = args.max_steps
        # frame_skip = args.frame_skip

        frame_idx  = 0
        rewards    = []
        success    = []
        batch_size = 128

        ep_num = 0

        rate=rospy.Rate(10)

        while frame_idx < max_frames:
            state = env.reset()
            planner.reset()
            action = planner(state.copy())
            # action = policy_net.get_action(state.copy()) # policy only
            episode_reward = 0
            episode_success = 0
            for step in range(max_steps):

                if np.isnan(action).any():
                    print('got nan')
                    # print(replay_buffer.buffer)
                    env.reset()
                    os._exit(0)
                next_state, reward, done = env.step(action.copy())

                # next_action = policy_net.get_action(next_state.copy())
                next_action = planner(next_state.copy())

                replay_buffer.push(state, action, reward, next_state, done)
                model_replay_buffer.push(state, action, reward, next_state, next_action, done)

                print(frame_idx,ep_num)
                # print(next_state, state)
                state = next_state.copy()
                action = next_action.copy()
                episode_reward += reward
                frame_idx += 1

                # if (frame_idx % int(max_frames/20) == 0) and (len(replay_buffer) > batch_size):
                #     pickle.dump(rewards, open(path + 'reward_data' + '.pkl', 'wb'))
                #     print('pickle elapsed time', start_time)
                if done:
                    episode_success = 1
                    print('done loop')
                    break
                else:
                    rate.sleep()

            rewards.append([frame_idx, episode_reward])
            success.append([frame_idx, ep_num, episode_success,reward])
            ep_num += 1
            if len(replay_buffer) > batch_size:
                for _ in range(max_steps/2):
                    sac.soft_q_update(batch_size)
                    model_optim.update_model(batch_size, mini_iter=args.model_iter, verbose=True)
        print('saving final data set')
        print(success)
        print(rewards)
        pickle.dump(success, open(path + 'success_data'+ '.pkl', 'wb'))
        pickle.dump(rewards, open(path + 'reward_data'+ '.pkl', 'wb'))
        torch.save(policy_net.state_dict(), path + 'policy_' + 'final' + '.pt')

    # except KeyboardInterrupt as e:
        # os._exit(0)
    except Exception as e:
        print(e)
        traceback.print_exc()
    finally:
        os.kill(os.getpid(),signal.SIGKILL)
