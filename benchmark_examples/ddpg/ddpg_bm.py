import numpy as np
import pickle
import gym
from datetime import datetime

import sys
import os

sys.path.append('../../')

# local imports
import envs

import torch
from ddpg import DDPG
from ddpg import PolicyNetwork
from ddpg import ReplayBuffer
from ddpg import OUNoise

# argparse things
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, help=envs.getlist())
parser.add_argument('--max_steps', type=int, default=200)
parser.add_argument('--max_frames', type=int, default=10000)
parser.add_argument('--frame_skip', type=int, default=2)
parser.add_argument('--value_lr', type=float, default=3e-3)
parser.add_argument('--policy_lr', type=float, default=3e-4)



parser.add_argument('--done_util', dest='done_util', action='store_true')
parser.add_argument('--no_done_util', dest='done_util', action='store_false')
parser.set_defaults(done_util=True)

parser.add_argument('--render', dest='render', action='store_true')
parser.add_argument('--no_render', dest='render', action='store_false')
parser.set_defaults(render=False)

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

if __name__ == '__main__':

    # env_name = 'ReacherBulletEnv-v0'

    # env_name = 'KukaBulletEnv-v0'
    # env_name = 'InvertedPendulumSwingupBulletEnv-v0'
    # env = pybullet_envs.make(env_name)
    # env = pybullet_envs.make(env_name)
    # env = KukaGymEnv(renders=True, isDiscrete=False)

    env_name = args.env
    try:
        env = envs.env_list[env_name](render=args.render)
    except TypeError as err:
        print('no argument render,  assumping env.render will just work')
        env = envs.env_list[env_name]()
    env.reset()
    print(env.action_space.low, env.action_space.high)
    assert np.any(np.abs(env.action_space.low) <= 1.) and  np.any(np.abs(env.action_space.high) <= 1.), 'Action space not normalizd'

    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d_%H-%M-%S/")

    path = './data/' + env_name +  '/' + date_str
    if os.path.exists(path) is False:
        os.makedirs(path)

    action_dim = env.action_space.shape[0]
    state_dim  = env.observation_space.shape[0]
    hidden_dim = 128

    policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim)

    replay_buffer_size = 1000000
    replay_buffer = ReplayBuffer(replay_buffer_size)

    ddpg = DDPG(policy=policy_net,
                          state_dim=state_dim,
                          action_dim=action_dim,
                          replay_buffer=replay_buffer,
                          policy_lr=args.policy_lr,
                          value_lr=args.value_lr)
    ou_noise = OUNoise(env.action_space, decay_period=args.max_frames)

    max_frames  = args.max_frames
    max_steps   = args.max_steps
    frame_skip = args.frame_skip

    frame_idx   = 0
    rewards     = []
    batch_size  = 128

    # for _ in range(5):
    #     get_expert_data(env, replay_buffer)
    # env. render('human')

    while frame_idx < max_frames:
        state = env.reset()
        episode_reward = 0
        for step in range(max_steps):
            # action = ou_noise.get_action(policy_net.get_action(state))
            # action = ou_noise.get_action(policy_net.get_action(state))
            action = policy_net.get_action(state) + np.random.normal(0., 1.0*(0.999**(frame_idx)), size=(action_dim))
            for _ in range(frame_skip):
                next_state, reward, done, _ = env.step(action.copy())

            replay_buffer.push(state, action, reward, next_state, done)
            if len(replay_buffer) > batch_size:
                ddpg.update(batch_size)

            state = next_state
            episode_reward += reward
            frame_idx += 1

            if args.render:
                env.render()

            if frame_idx % int(max_frames/10) == 0:
                print(
                    'frame : {}/{}, \t last rew : {}'.format(
                        frame_idx, max_frames, rewards[-1][1]
                    )
                )
                pickle.dump(rewards, open(path + 'reward_data' + '.pkl', 'wb'))
                torch.save(policy_net.state_dict(), path + 'policy_' + str(frame_idx) + '.pt')

            if args.done_util:
                if done:
                    break

        rewards.append([frame_idx, episode_reward])

    print('saving final data set')
    pickle.dump(rewards, open(path + 'reward_data'+ '.pkl', 'wb'))
    torch.save(policy_net.state_dict(), path + 'policy_' + 'final' + '.pt')
