import numpy as np
import pickle
import gym

import sys
import os

sys.path.append('../')

# local imports
import torch
from sac import SoftActorCritic
from sac import PolicyNetwork
from sac import ReplayBuffer
from sac import NormalizedActions

from gym.envs.box2d.lunar_lander import heuristic as expert

def get_expert_data(env, replay_buffer, T=200):
    state = env.reset()
    for t in range(T):
        action = expert(env, state)
        next_state, reward, done, info = env.step(action)
        replay_buffer.push(state, action, reward, next_state, done)

        state = next_state

        if done:
            break

# TODO: add arg parse

if __name__ == '__main__':

    env = gym.make("LunarLanderContinuous-v2")

    action_dim = env.action_space.shape[0]
    state_dim  = env.observation_space.shape[0]
    hidden_dim = 128

    policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim)

    replay_buffer_size = 1000000
    replay_buffer = ReplayBuffer(replay_buffer_size)

    sac = SoftActorCritic(policy=policy_net,
                          state_dim=state_dim,
                          action_dim=action_dim,
                          replay_buffer=replay_buffer, policy_lr = 3e-3)

    max_frames  = 10000
    max_steps   = 200
    frame_idx   = 0
    rewards     = []
    batch_size  = 128

    # for _ in range(5):
    #     get_expert_data(env, replay_buffer)
    frame_skip = 2
    while frame_idx < max_frames:
        state = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            action = policy_net.get_action(state)
            
            for _ in range(frame_skip):
                next_state, reward, done, _ = env.step(action)

            replay_buffer.push(state, action, reward, next_state, done)
            if len(replay_buffer) > batch_size:
                sac.soft_q_update(batch_size)

            state = next_state
            episode_reward += reward
            frame_idx += 1

            #env.render()

            if frame_idx % 200 == 0:
                print(
                    'frame : {}/{}, \t last rew : {}'.format(
                        frame_idx, max_frames, rewards[-1]
                    )
                )
                path = './data/lunar_lander/'
                if os.path.exists(path) is False:
                    os.mkdir(path)
                pickle.dump(rewards, open(path + 'reward_data4.pkl', 'wb'))
                torch.save(policy_net.state_dict(), path + 'policy4.pt')

            if done:
                break

        rewards.append(episode_reward)
    path = './data/lunar_lander/'
    if os.path.exists(path) is False:
        os.mkdir(path)
    pickle.dump(rewards, open(path + 'reward_data4.pkl', 'wb'))
    torch.save(policy_net.state_dict(), path + 'policy4.pt')
