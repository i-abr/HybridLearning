import numpy as np
import pickle
# import gym

# from pybullet_envs.bullet.kukaGymEnv import KukaGymEnv
import pybullet_envs

print(pybullet_envs.getList())

import sys
import os
sys.path.append('../')

# local imports
import torch
from sac import SoftActorCritic
from sac import PolicyNetwork
from sac import ReplayBuffer
from sac import NormalizedActions

from mppi import MPPI
from model_learning import ModelOptim, Model


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
    env_name = 'InvertedPendulumSwingupBulletEnv-v0'
    # env_name = 'ReacherBulletEnv-v0'
    # env_name = 'HalfCheetahBulletEnv-v0'
    env = pybullet_envs.make(env_name)
    # env.isRender = True
    # env = KukaGymEnv(renders=True, isDiscrete=False)
    # env.camera_adjust()
    env.reset()

    path = './data/' + env_name +  '/'
    if os.path.exists(path) is False:
        os.mkdir(path)

    action_dim = env.action_space.shape[0]
    state_dim  = env.observation_space.shape[0]
    hidden_dim = 128

    policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim)

    model = Model(state_dim, action_dim, def_layers=[200, 200])

    planner = MPPI(model, policy_net, samples=20, t_H=5, lam=0.1)

    replay_buffer_size = 1000000
    replay_buffer = ReplayBuffer(replay_buffer_size)

    model_optim = ModelOptim(model, replay_buffer, lr=1e-3)
    sac = SoftActorCritic(policy=policy_net,
                          state_dim=state_dim,
                          action_dim=action_dim,
                          replay_buffer=replay_buffer, policy_lr = 3e-3)

    max_frames  = 20000
    max_steps   = 400
    frame_idx   = 0
    rewards     = []
    batch_size  = 128

    # for _ in range(5):
    #     get_expert_data(env, replay_buffer)
    frame_skip = 5
    print(env.action_space.low, env.action_space.high)
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

            env.render("human")


            if frame_idx % 500 == 0:
                print(
                    'frame : {}/{}, \t last rew : {}, \t model loss : {}'.format(
                        frame_idx, max_frames, rewards[-1], model_optim.log['loss'][-1]
                    )
                )

                pickle.dump(rewards, open(path + 'reward_data.pkl', 'wb'))
                torch.save(policy_net.state_dict(), path + 'policy.pt')

            if done:
                break
        # if len(replay_buffer) > 64:
        #     model_optim.update_model(64, mini_iter=10)
        # rewards.append(episode_reward)
        # rewards.append(test_with_planner(env, planner, max_steps))
        rewards.append(episode_reward)

    path = './data/' + env_name +  '/'
    if os.path.exists(path) is False:
        os.mkdir(path)
    pickle.dump(rewards, open(path + 'reward_data.pkl', 'wb'))
    torch.save(policy_net.state_dict(), path + 'policy.pt')
