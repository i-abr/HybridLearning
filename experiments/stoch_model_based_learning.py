import numpy as np
import pickle
from datetime import datetime

import sys
import os
sys.path.append('../')

# local imports
import envs

import torch
from stoch_model_based_control import PathIntegral
from model import ModelOptimizer, Model, SARSAReplayBuffer
# argparse things
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--env',        type=str,   help=envs.getlist())
parser.add_argument('--max_steps',  type=int,   default=200)
parser.add_argument('--max_frames', type=int,   default=10000)
parser.add_argument('--frame_skip', type=int,   default=2)
parser.add_argument('--model_lr',   type=float, default=3e-3)


parser.add_argument('--horizon', type=int, default=5)
parser.add_argument('--model_iter', type=int, default=1)
parser.add_argument('--trajectory_samples', type=int, default=20)


parser.add_argument('--done_util', dest='done_util', action='store_true')
parser.add_argument('--no_done_util', dest='done_util', action='store_false')
parser.set_defaults(done_util=True)

parser.add_argument('--render', dest='render', action='store_true')
parser.add_argument('--no_render', dest='render', action='store_false')
parser.set_defaults(render=False)

args = parser.parse_args()

class ActionWrapper(object):
    def __init__(self, action_space):
        self.action_space = action_space
        self.mid = np.mean([action_space.low, action_space.high], axis=0)
        self.rng = 0.5*(action_space.high-action_space.low)

    def __call__(self, action):
        # return action

        return self.mid + np.clip(action, -1,1)*self.rng

if __name__ == '__main__':



    env_name = args.env
    try:
        env = envs.env_list[env_name](render=args.render)
    except TypeError as err:
        print('no argument render,  assumping env.render will just work')
        env = envs.env_list[env_name]()
    env.reset()
    action_wrapper = ActionWrapper(env.action_space)

    # assert np.any(np.abs(env.action_space.low) <= 1.) and  np.any(np.abs(env.action_space.high) <= 1.), 'Action space not normalizd'

    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d_%H-%M-%S/")

    path = './data/' + env_name +  '/' + 'stoch_model_based_learning/' + date_str
    if os.path.exists(path) is False:
        os.makedirs(path)

    action_dim = env.action_space.shape[0]
    state_dim  = env.observation_space.shape[0]
    hidden_dim = 12

    model = Model(state_dim, action_dim, hidden_dim=128)

    replay_buffer_size = 100000

    model_replay_buffer = SARSAReplayBuffer(replay_buffer_size)
    model_optim = ModelOptimizer(model, model_replay_buffer, lr=args.model_lr)

    planner = PathIntegral(model, samples=args.trajectory_samples, t_H=args.horizon, lam=0.1)

    max_frames  = args.max_frames
    max_steps   = args.max_steps
    frame_skip = args.frame_skip

    frame_idx   = 0
    rewards     = []
    batch_size  = 128

    # env.camera_adjust()
    ep_num = 0
    while frame_idx < max_frames:
        state = env.reset()
        planner.reset()

        action = planner(state)

        episode_reward = 0
        for step in range(max_steps):

            for _ in range(frame_skip):
                next_state, reward, done, _ = env.step(action_wrapper(action.copy()))

            next_action = planner(next_state)

            model_replay_buffer.push(state, action, reward, next_state, next_action, done)

            if len(model_replay_buffer) > batch_size:
                model_optim.update_model(batch_size, mini_iter=args.model_iter)
                print('iter', frame_idx,
                    'model loss', model_optim.log['model_loss'][-1],
                    'rew_loss', model_optim.log['rew_loss'][-1])
            state = next_state
            action = next_action
            episode_reward += reward
            frame_idx += 1

            if args.render:
                env.render("human")


            if frame_idx % int(max_frames/10) == 0:
                print(
                    'frame : {}/{}, \t last rew : {}'.format(
                        frame_idx, max_frames, rewards[-1][1]
                    )
                )

                pickle.dump(rewards, open(path + 'reward_data' + '.pkl', 'wb'))
                torch.save(model.state_dict(), path + 'model_' + str(frame_idx) + '.pt')

            if args.done_util:
                if done:
                    break

        if len(model_replay_buffer) > batch_size:
            print('ep rew', ep_num, episode_reward)
        rewards.append([frame_idx, episode_reward])
        ep_num += 1
    print('saving final data set')
    pickle.dump(rewards, open(path + 'reward_data'+ '.pkl', 'wb'))
    torch.save(model.state_dict(), path + 'model_' + 'final' + '.pt')
