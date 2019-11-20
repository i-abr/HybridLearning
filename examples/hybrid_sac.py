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
from hybrid_stochastic import PathIntegral
from model import ModelOptimizer, Model

# argparse things
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--env',        type=str,   help=envs.getlist())
parser.add_argument('--render',     type=bool,  default=False)
parser.add_argument('--max_steps',  type=int,   default=200)
parser.add_argument('--max_frames', type=int,   default=10000)
parser.add_argument('--frame_skip', type=int,   default=2)
parser.add_argument('--model_lr',   type=float, default=1e-3)

args = parser.parse_args()


def get_expert_data(env, replay_buffer, T=200):
    state = env.reset()
    for t in range(T):
        action = expert(env, state)
        next_state, reward, done, info = env.step(action)
        replay_buffer.push(state, action, reward, next_state, done)

        state = next_state

        if done:
            break

def test_with_planner(env, planner, max_steps=200):
        state = env.reset()
        planner.reset()
        episode_reward = 0
        for step in range(max_steps):
            action = planner(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            episode_reward += reward
            if done:
                break
        return episode_reward

if __name__ == '__main__':

    # env_name = 'KukaBulletEnv-v0'
    # env_name = 'InvertedPendulumSwingupBulletEnv-v0'
    # env_name = 'ReacherBulletEnv-v0'
    # env_name = 'HalfCheetahBulletEnv-v0'
    # env = pybullet_envs.make(env_name)
    # env.isRender = True
    # env = KukaGymEnv(renders=True, isDiscrete=False)
    # env.camera_adjust()


    env_name = args.env
    env = envs.env_list[env_name]()
    env.reset()


    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d_%H-%M-%S/")

    path = './data/' + env_name +  '/' + date_str
    if os.path.exists(path) is False:
        os.makedirs(path)

    action_dim = env.action_space.shape[0]
    state_dim  = env.observation_space.shape[0]
    hidden_dim = 128

    policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim)

    model = Model(state_dim, action_dim, def_layers=[200, 200])

    planner = PathIntegral(model, policy_net, samples=20, t_H=5, lam=0.1)

    replay_buffer_size = 1000000
    replay_buffer = ReplayBuffer(replay_buffer_size)

    model_optim = ModelOptimizer(model, replay_buffer, lr=args.model_lr)
    sac = SoftActorCritic(policy=policy_net,
                          state_dim=state_dim,
                          action_dim=action_dim,
                          replay_buffer=replay_buffer, policy_lr = 3e-3)

    max_frames  = args.max_frames
    max_steps   = args.max_steps
    frame_skip = args.frame_skip

    frame_idx   = 0
    rewards     = []
    batch_size  = 128

    while frame_idx < max_frames:
        state = env.reset()
        planner.reset()
        episode_reward = 0

        for step in range(max_steps):
            action = planner(state)
            # action = policy_net.get_action(state)
            for _ in range(frame_skip):
                next_state, reward, done, _ = env.step(action.copy())

            replay_buffer.push(state, action, reward, next_state, done)
            if len(replay_buffer) > batch_size:
                sac.soft_q_update(batch_size)
                model_optim.update_model(batch_size, mini_iter=2)

            state = next_state
            episode_reward += reward
            frame_idx += 1

            if args.render:
                env.render("human")


            if frame_idx % int(max_frames/10) == 0:
                print(
                    'frame : {}/{}, \t last rew : {}, \t model loss : {}'.format(
                        frame_idx, max_frames, rewards[-1], model_optim.log['loss'][-1]
                    )
                )

                pickle.dump(rewards, open(path + 'reward_data' + '.pkl', 'wb'))
                torch.save(policy_net.state_dict(), path + 'policy_' + str(frame_idx) + '.pt')

            if done:
                break
        # if len(replay_buffer) > 64:
        #     model_optim.update_model(64, mini_iter=10)
        # rewards.append(episode_reward)
        # rewards.append(test_with_planner(env, planner, max_steps))
        rewards.append(episode_reward)

    print('saving final data set')
    pickle.dump(rewards, open(path + 'reward_data'+ '.pkl', 'wb'))
    torch.save(policy_net.state_dict(), path + 'policy_' + 'final' + '.pt')
