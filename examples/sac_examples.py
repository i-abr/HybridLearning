import numpy as np
import pickle
import gym

import sys

sys.path.append('../')

# local imports
import torch
from sac import SoftActorCritic
from sac import PolicyNetwork
from sac import ReplayBuffer
from sac import NormalizedActions

from mppi import MPPI
from model_learning import ModelOptim, Model


# TODO: add arg parse

if __name__ == '__main__':

    env = NormalizedActions(gym.make("Pendulum-v0"))

    action_dim = env.action_space.shape[0]
    state_dim  = env.observation_space.shape[0]
    hidden_dim = 128

    policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim)

    model = Model(state_dim, action_dim)

    planner = MPPI(model, policy_net)

    replay_buffer_size = 1000000
    replay_buffer = ReplayBuffer(replay_buffer_size)

    model_optim = ModelOptim(model, replay_buffer, lr=1e-2)
    sac = SoftActorCritic(policy=policy_net,
                          state_dim=state_dim,
                          action_dim=action_dim,
                          replay_buffer=replay_buffer)

    max_frames  = 40000
    max_steps   = 500
    frame_idx   = 0
    rewards     = []
    batch_size  = 128


    while frame_idx < max_frames:
        state = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            action = planner(state)
            next_state, reward, done, _ = env.step(action)

            replay_buffer.push(state, action, reward, next_state, done)
            if len(replay_buffer) > batch_size:
                sac.soft_q_update(batch_size)

            state = next_state
            episode_reward += reward
            frame_idx += 1

            #env.render()

            if frame_idx % 1000 == 0:
                print(
                    'frame : {}/{}, \t last rew : {}'.format(
                        frame_idx, max_frames, rewards[-1]
                    )
                )
                pickle.dump(rewards,
                    open('rew_data.pkl'.format(frame_idx), 'wb')
                )

                torch.save(policy_net.state_dict(), 'policy.pkl'.format(frame_idx))

            if done:
                break
        if len(replay_buffer) > 128:
            model_optim.update_model(128, mini_iter=10)
        rewards.append(episode_reward)
