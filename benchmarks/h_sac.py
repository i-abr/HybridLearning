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
from model import MDNModelOptimizer, MDNModel
# argparse things
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--env',        type=str,   help=envs.getlist())
parser.add_argument('--max_steps',  type=int,   default=200)
parser.add_argument('--max_frames', type=int,   default=10000)
parser.add_argument('--frame_skip', type=int,   default=2)
parser.add_argument('--model_lr',   type=float, default=1e-3)
parser.add_argument('--policy_lr',  type=float, default=3e-3)
parser.add_argument('--value_lr',   type=float, default=3e-4)
parser.add_argument('--soft_q_lr',  type=float, default=3e-4)

parser.add_argument('--horizon', type=int, default=5)
parser.add_argument('--model_iter', type=int, default=2)
parser.add_argument('--trajectory_samples', type=int, default=20)


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

    # model = Model(state_dim, action_dim, def_layers=[200, 200])
    model = MDNModel(state_dim, action_dim, def_layers=[200, 200])

    planner = PathIntegral(model, policy_net, samples=args.trajectory_samples, t_H=args.horizon, lam=0.1)

    replay_buffer_size = 1000000
    replay_buffer = ReplayBuffer(replay_buffer_size)

    # model_optim = ModelOptimizer(model, replay_buffer, lr=args.model_lr)

    model_optim = MDNModelOptimizer(model, replay_buffer, lr=args.model_lr)


    sac = SoftActorCritic(policy=policy_net,
                          state_dim=state_dim,
                          action_dim=action_dim,
                          replay_buffer=replay_buffer,
                          policy_lr=args.policy_lr,
                          value_lr=args.value_lr,
                          soft_q_lr=args.soft_q_lr)


    max_frames  = args.max_frames
    max_steps   = args.max_steps
    frame_skip = args.frame_skip

    frame_idx   = 0
    rewards     = []
    batch_size  = 128

    # env.camera_adjust()

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
                model_optim.update_model(batch_size, mini_iter=args.model_iter)

            state = next_state
            episode_reward += reward
            frame_idx += 1

            if args.render:
                env.render("human")


            if frame_idx % int(max_frames/10) == 0:
                print(
                    'frame : {}/{}, \t last rew : {}, \t model loss : {}'.format(
                        frame_idx, max_frames, rewards[-1][1], model_optim.log['loss'][-1]
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
