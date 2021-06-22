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

import torch
# from sac_lib import SoftActorCritic
# from sac_lib import PolicyNetwork
# from sac_lib import ReplayBuffer
from sac_lib import NormalizedActions
from mpc_lib import PathIntegral
from model import ModelOptimizer, Model, SARSAReplayBuffer
# from model import MDNModelOptimizer, MDNModel
# argparse things
import argparse

import glob

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='InvertedPendulumEnv', help=envs.getlist())
# parser.add_argument('--frame', type=int, default=-1)
# parser.add_argument('--seed', type=int, default=666)
parser.add_argument('--done_util', dest='done_util', action='store_true')
parser.add_argument('--no_done_util', dest='done_util', action='store_false')
parser.set_defaults(done_util=True)
parser.add_argument('--render', dest='render', action='store_true')
parser.add_argument('--no_render', dest='render', action='store_false')
parser.set_defaults(render=False)

args = parser.parse_args()


if __name__ == '__main__':

    config_path = '../config/hlt_stoch.yaml'
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
    env.reset()
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

    trials = []
    state_dict_path = '../data/hlt_stoch/' + env_name
#     model_paths = glob.glob('../data/hlt_stoch/'+ env_name +'/seed_'+ str(args.seed) +'/model_*')

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
#         for k in range(20):
#             print(k)
        rewards = []
        model_paths = glob.glob(seed_dir +'/model_*')
        for m_path in model_paths:
            final = False
            try: 
                frame_idx = int(m_path.split('model_')[-1].split('.pt')[0])
            except:
                final = True
            if not final: 
                model = Model(state_dim, action_dim, 
                              def_layers=[200],AF=config['activation_fun']).to(device)
                model.load_state_dict(torch.load(m_path, map_location=device))
                planner = PathIntegral(model,
                                       samples=config['trajectory_samples'],
                                       t_H= config['horizon'], 
                                       lam= config['lam'])


                state = env.reset()
                planner.reset()

                action, _ = planner(state)

                episode_reward = 0
                for step in range(max_steps):
                    # action = policy_net.get_action(state)
                    for _ in range(frame_skip):
                        next_state, reward, done, _ = env.step(action.copy())

                    next_action, _ = planner(next_state)


                    state = next_state
                    action = next_action
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

    path = '../data/hlt_stoch/' + env_name +  '/' + 'model_test/'
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
