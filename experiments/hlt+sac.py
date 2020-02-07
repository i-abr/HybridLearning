import numpy as np
import pickle
from datetime import datetime

import sys
import os
sys.path.append('../')

# local imports
# import envs
import robosuite as suite

import torch
from saclib import SoftActorCritic
from saclib import PolicyNetwork
from saclib import ReplayBuffer
from saclib import NormalizedActions
from hltlib import HybridMPC
from modellib import ModelOptimizer, Model, SARSAReplayBuffer
# from model import MDNModelOptimizer, MDNModel
# argparse things
import argparse

parser = argparse.ArgumentParser()
# parser.add_argument('--env',        type=str,   help=envs.getlist())
parser.add_argument('--max_steps',  type=int,   default=200)
parser.add_argument('--max_frames', type=int,   default=10000)
parser.add_argument('--frame_skip', type=int,   default=1)
parser.add_argument('--model_lr',   type=float, default=3e-2)
parser.add_argument('--policy_lr',  type=float, default=3e-4)
parser.add_argument('--value_lr',   type=float, default=3e-4)
parser.add_argument('--soft_q_lr',  type=float, default=3e-4)

parser.add_argument('--horizon', type=int, default=40)
parser.add_argument('--model_iter', type=int, default=2)
parser.add_argument('--trajectory_samples', type=int, default=20)
parser.add_argument('--lam',  type=float, default=0.1)


parser.add_argument('--done_util', dest='done_util', action='store_true')
parser.add_argument('--no_done_util', dest='done_util', action='store_false')
parser.set_defaults(done_util=True)

parser.add_argument('--render', dest='render', action='store_true')
parser.add_argument('--no_render', dest='render', action='store_false')
parser.set_defaults(render=True)

args = parser.parse_args()


if __name__ == '__main__':


    # env_name = args.env
    # try:
    #     env = envs.env_list[env_name](render=args.render)
    # except TypeError as err:
    #     print('no argument render,  assumping env.render will just work')
    #     env = envs.env_list[env_name]()
    # env.reset()
    # print(env.action_space.low, env.action_space.high)
    # assert np.any(np.abs(env.action_space.low) <= 1.) and  np.any(np.abs(env.action_space.high) <= 1.), 'Action space not normalizd'


    env = suite.make(
        "SawyerLift",
        has_renderer=True,
        use_camera_obs=False,
        reward_shaping=True
    )
    # env.viewer.set_camera(camera_id=0)

    # now = datetime.now()
    # date_str = now.strftime("%Y-%m-%d_%H-%M-%S/")
    #
    # path = './data/' + env_name +  '/' + 'h_sac/' + date_str
    # if os.path.exists(path) is False:
    #     os.makedirs(path)
    obs = env.reset()
    action_dim = env.dof
    state_dim  = np.concatenate([obs['robot-state'], obs['object-state']]).shape[0]
    hidden_dim = 128

    policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim)

    model = Model(state_dim, action_dim, layers=[256, 128])
    # model = MDNModel(state_dim, action_dim, def_layers=[200, 200])


    replay_buffer_size = 1000000
    replay_buffer = ReplayBuffer(replay_buffer_size)

    model_replay_buffer = SARSAReplayBuffer(replay_buffer_size)
    model_optim = ModelOptimizer(model, model_replay_buffer, lr=args.model_lr)

    # model_optim = MDNModelOptimizer(model, replay_buffer, lr=args.model_lr)


    sac = SoftActorCritic(policy=policy_net,
                          state_dim=state_dim,
                          action_dim=action_dim,
                          replay_buffer=replay_buffer,
                          policy_lr=args.policy_lr,
                          value_lr=args.value_lr,
                          soft_q_lr=args.soft_q_lr)

    planner = HybridMPC(model, policy_net, samples=args.trajectory_samples, t_H=args.horizon, lam=args.lam)

    max_frames  = args.max_frames
    max_steps   = args.max_steps
    frame_skip = args.frame_skip

    frame_idx   = 0
    rewards     = []
    batch_size  = 128

    # env.camera_adjust()
    ep_num = 0
    while frame_idx < max_frames:

        obs = env.reset()
        state = np.concatenate([obs['robot-state'], obs['object-state']])
        planner.reset()

        action = planner(state)
        episode_reward = 0
        for step in range(max_steps):
            # action = policy_net.get_action(state)
            # for _ in range(frame_skip):
            next_obs, reward, done, _ = env.step(action.copy())
            # reward *= 0.01
            next_state = np.concatenate([next_obs['robot-state'], next_obs['object-state']])
            next_action = planner(next_state)

            replay_buffer.push(state, action, reward, next_state, done)
            model_replay_buffer.push(state, action, reward, next_state, next_action, done)

            state = next_state
            action = next_action
            episode_reward += reward
            frame_idx += 1

            if args.render:
                env.render()


            # if frame_idx % int(max_frames/10) == 0:
            #     print(
            #         'frame : {}/{}, \t last rew : {}, \t rew loss : {}'.format(
            #             frame_idx, max_frames, rewards[-1][1], model_optim.log['rew_loss'][-1]
            #         )
            #     )
            #
            #     # pickle.dump(rewards, open(path + 'reward_data' + '.pkl', 'wb'))
            #     # torch.save(policy_net.state_dict(), path + 'policy_' + str(frame_idx) + '.pt')

            if args.done_util:
                if done:
                    break
        #if len(replay_buffer) > batch_size:
        #    for k in range(200):
        #        sac.soft_q_update(batch_size)
        #        model_optim.update_model(batch_size, mini_iter=1)#args.model_iter)
        if len(replay_buffer) > batch_size:
            sac.soft_q_update(batch_size, epochs=100, verbose=True)
            model_optim.update_model(batch_size, epochs=100, verbose=True)
        # if len(replay_buffer) > batch_size:
        #     print('ep rew', ep_num, episode_reward, model_optim.log['rew_loss'][-1], model_optim.log['loss'][-1])
        #     print('ssac loss', sac.log['value_loss'][-1], sac.log['policy_loss'][-1], sac.log['q_value_loss'][-1])
        rewards.append([frame_idx, episode_reward])
        print('episode',ep_num, 'reward', episode_reward)
        ep_num += 1
    print('saving final data set')
    # pickle.dump(rewards, open(path + 'reward_data'+ '.pkl', 'wb'))
    # torch.save(policy_net.state_dict(), path + 'policy_' + 'final' + '.pt')
