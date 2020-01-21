import numpy as np
import pickle
from datetime import datetime

import sys
import os
sys.path.append('../')

# local imports
import envs

import torch
from sac import SoftActorCritic
from sac import PolicyNetwork
from sac import ReplayBuffer
from sac import NormalizedActions
from stoch_model_based_control import PathIntegral
from model import ModelOptimizer, Model, SARSAReplayBuffer
# from model import MDNModelOptimizer, MDNModel
# argparse things
import argparse

import glob

parser = argparse.ArgumentParser()
parser.add_argument('--env',        type=str,   help=envs.getlist())
parser.add_argument('--max_steps',  type=int,   default=200)
parser.add_argument('--max_frames', type=int,   default=10000)
parser.add_argument('--frame_skip', type=int,   default=2)
parser.add_argument('--model_lr',   type=float, default=3e-3)
parser.add_argument('--policy_lr',  type=float, default=3e-4)
parser.add_argument('--value_lr',   type=float, default=3e-4)
parser.add_argument('--soft_q_lr',  type=float, default=3e-4)

parser.add_argument('--horizon', type=int, default=5)
parser.add_argument('--model_iter', type=int, default=2)
parser.add_argument('--trajectory_samples', type=int, default=20)
parser.add_argument('--lam',  type=float, default=0.1)


parser.add_argument('--done_util', dest='done_util', action='store_true')
parser.add_argument('--no_done_util', dest='done_util', action='store_false')
parser.set_defaults(done_util=True)

parser.add_argument('--render', dest='render', action='store_true')
parser.add_argument('--no_render', dest='render', action='store_false')
parser.set_defaults(render=False)

args = parser.parse_args()



if __name__ == '__main__':

    env_name = args.env
    try:
        env = envs.env_list[env_name](render=args.render)
    except TypeError as err:
        print('no argument render,  assumping env.render will just work')
        env = envs.env_list[env_name]()
    env.reset()
    assert np.any(np.abs(env.action_space.low) <= 1.) and  np.any(np.abs(env.action_space.high) <= 1.), 'Action space not normalizd'

    # now = datetime.now()
    # date_str = now.strftime("%Y-%m-%d_%H-%M-%S/")
    #
    # path = './data/' + env_name +  '/' + 'model_test/' + date_str
    # if os.path.exists(path) is False:
    #     os.makedirs(path)

    model_paths = glob.glob('./data/InvertedPendulumSwingupBulletEnv/h_sac/2020-01-20_13-23-05/policy_*')

    action_dim = env.action_space.shape[0]
    state_dim  = env.observation_space.shape[0]
    hidden_dim = 128

    max_frames  = args.max_frames
    max_steps   = args.max_steps
    frame_skip = args.frame_skip
    trials = []
    for k in range(20):
        rewards = []
        for m_path in model_paths:

            policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim)

            policy_net.load_state_dict(torch.load(m_path))


            frame_idx = int(m_path.split('policy_')[-1].split('.pt')[0])

            state = env.reset()


            episode_reward = 0
            for step in range(max_steps):
                action = policy_net.get_action(state)

                for _ in range(frame_skip):
                    state, reward, done, _ = env.step(action.copy())

                episode_reward += reward

                if args.render:
                    env.render("human")

                if args.done_util:
                    if done:
                        break
            rewards.append([frame_idx, episode_reward])
        rewards = sorted(rewards, key=lambda x: x[0])
        rewards = np.array(rewards)
        trials.append(rewards)
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d_%H-%M-%S/")

    path = './data/' + env_name +  '/' + 'policy_test/'
    if os.path.exists(path) is False:
        os.makedirs(path)

    print('saving final data set')
    pickle.dump(trials, open(path + 'reward_data'+ '.pkl', 'wb'))

        #     rewards.append([frame_idx, episode_reward])
        #     if perf_rec is None:
        #         perf_rec = episode_reward
        #     elif episode_reward > perf_rec:
        #         torch.save(policy_net.state_dict(), path + 'policy_' + str(frame_idx) + '.pt')
        #         torch.save(model.state_dict(), path + 'model_' + str(frame_idx) + '.pt')
        #         perf_rec = episode_reward
        #     print('ep rew', ep_num, episode_reward)
        #
        #     ep_num += 1
        # print('saving final data set')
        # pickle.dump(rewards, open(path + 'reward_data'+ '.pkl', 'wb'))
        # torch.save(policy_net.state_dict(), path + 'policy_' + 'final' + '.pt')
