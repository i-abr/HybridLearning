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
sys.path.append('../')

import argparse
from copy import copy, deepcopy
import time

# model
import torch
from sac import SoftActorCritic
from sac import PolicyNetwork
from sac import ReplayBuffer
# from sac import NormalizedActions
from hybrid_stochastic import PathIntegral
from model import ModelOptimizer, Model, SARSAReplayBuffer
# from model import MDNModelOptimizer, MDNModel

# ros
import rospy
from sawyer_env import sawyer_env

"""
arg parse things
"""
parser = argparse.ArgumentParser()
# parser.add_argument('--env',        type=str,   help=envs.getlist())
parser.add_argument('--max_steps',  type=int,   default=500)
parser.add_argument('--max_frames', type=int,   default=10000)
# parser.add_argument('--frame_skip', type=int,   default=2)
parser.add_argument('--model_lr',   type=float, default=3e-4)
parser.add_argument('--policy_lr',  type=float, default=3e-3)
parser.add_argument('--value_lr',   type=float, default=3e-4)
parser.add_argument('--soft_q_lr',  type=float, default=3e-4)

parser.add_argument('--horizon', type=int, default=5)
parser.add_argument('--model_iter', type=int, default=1)
parser.add_argument('--trajectory_samples', type=int, default=20)
parser.add_argument('--lam',  type=float, default=1)


# parser.add_argument('--done_util', dest='done_util', action='store_true')
# parser.add_argument('--no_done_util', dest='done_util', action='store_false')
# parser.set_defaults(done_util=True)

# parser.add_argument('--render', dest='render', action='store_true')
# parser.add_argument('--no_render', dest='render', action='store_false')
# parser.set_defaults(render=False)

args = parser.parse_args()

if __name__ == '__main__':
    try:
        rospy.init_node('h_sac')

        env = sawyer_env()

        env_name = 'sawyer'
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d_%H-%M-%S/")

        path = './data/' + env_name +  '/' + 'h_sac/' + date_str
        if os.path.exists(path) is False:
            os.makedirs(path)

        action_dim = 2
        state_dim  = 4
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
        batch_size = 128

        ep_num = 0

        rate=rospy.Rate(20)

        while frame_idx < max_frames:
            rospy.loginfo("start loop")
            state = env.reset()
            rospy.loginfo("get state after reset")

            planner.reset()

            action = planner(state.copy())

            episode_reward = 0
            for step in range(max_steps):
                if np.isnan(action).any():
                    print('got nan')
                    print(replay_buffer.buffer)
                    env.reset()
                    os._exit(0)
                next_state, reward, done = env.step(action.copy())

                start_time = time.time()
                # print('state',next_state)
                next_action = planner(next_state)
                # print('elapsed time',time.time()-start_time)
                print(step)

                replay_buffer.push(state, action, reward, next_state, done)
                model_replay_buffer.push(state, action, reward, next_state, next_action, done)

                if len(replay_buffer) > batch_size:
                    # if frame_idx > 20:
                    # for _ in range(10):
                    sac.soft_q_update(batch_size);
                    model_optim.update_model(batch_size, mini_iter=args.model_iter)

                state = next_state
                action = next_action
                episode_reward += reward
                frame_idx += 1

                # if args.render:
                #     env.render("human"
                # print(len(rewards), len(model_optim.log['rew_loss']))
                # print(episode_reward)
                if (frame_idx % int(max_frames/20) == 0) and (len(replay_buffer) > batch_size):
                    # print(
                    #     'frame : {}/{}, \t last rew : {}, \t rew loss : {}'.format(
                    #         frame_idx, max_frames, rewards[-1][1], model_optim.log['rew_loss'][-1]
                    #     )
                    # )
                    start_time = time.time()
                    pickle.dump(rewards, open(path + 'reward_data' + '.pkl', 'wb'))
                    torch.save(policy_net.state_dict(), path + 'policy_' + str(frame_idx) + '.pt')
                    end_time = time.time()
                    print('pickle elapsed time', start_time)
                if done:
                    print('done loop')
                    break
                else:
                    rate.sleep()
            #if len(replay_buffer) > batch_size:
            #    for k in range(200):
            #        sac.soft_q_update(batch_size)
            #        model_optim.update_model(batch_size, mini_iter=1)#args.model_iter)

            if len(replay_buffer) > batch_size:
                print('ep rew', ep_num, episode_reward, model_optim.log['rew_loss'][-1], model_optim.log['loss'][-1])
                print('ssac loss', sac.log['value_loss'][-1], sac.log['policy_loss'][-1], sac.log['q_value_loss'][-1])
            rewards.append([frame_idx, episode_reward])
            ep_num += 1
        print('saving final data set')
        pickle.dump(rewards, open(path + 'reward_data'+ '.pkl', 'wb'))
        torch.save(policy_net.state_dict(), path + 'policy_' + 'final' + '.pt')

    except KeyboardInterrupt, rospy.ROSInterruptException:
        os._exit(0)
