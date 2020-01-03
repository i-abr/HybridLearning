import numpy as np
import pickle
from datetime import datetime

import sys
import os
sys.path.append('../')
import random

# local imports
import envs

import torch
import torch.optim as optim
from torch.distributions import Normal

from sac import SoftActorCritic
from sac import PolicyNetwork
from sac import ReplayBuffer
from sac import NormalizedActions
from hybrid_stochastic import PathIntegral
from model import ModelOptimizer, Model, SARSAReplayBuffer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--env',        type=str,   help=envs.getlist(), default="AntBulletEnv")
parser.add_argument('--max_steps',  type=int,   default=1000)
parser.add_argument('--max_frames', type=int,   default=10000)
parser.add_argument('--frame_skip', type=int,   default=1)
parser.add_argument('--model_lr',   type=float, default=3e-3)
parser.add_argument('--policy_lr',  type=float, default=3e-4)
parser.add_argument('--value_lr',   type=float, default=3e-4)
parser.add_argument('--soft_q_lr',  type=float, default=3e-4)

parser.add_argument('--horizon', type=int, default=20)
parser.add_argument('--model_iter', type=int, default=1)
parser.add_argument('--trajectory_samples', type=int, default=10)


parser.add_argument('--done_util', dest='done_util', action='store_true')
parser.add_argument('--no_done_util', dest='done_util', action='store_false')
parser.set_defaults(done_util=True)

parser.add_argument('--render', dest='render', action='store_true')
parser.add_argument('--no_render', dest='render', action='store_false')
parser.set_defaults(render=True)

args = parser.parse_args()

def update_models_from_demos(demos, model, policy, iters=200):
    p_optim = optim.Adam(policy.parameters(),  lr=3e-3)
    m_optim = optim.Adam(model.parameters(), lr=3e-3)

    for k in range(iters):

        expert_batch = random.sample(demos, 256)

        state, action, reward, next_state, next_action, done = map(np.stack, zip(*expert_batch))

        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        action = torch.FloatTensor(action)
        next_action = torch.FloatTensor(next_action)
        reward = torch.FloatTensor(reward).unsqueeze(1)
        done    = torch.FloatTensor(np.float32(done)).unsqueeze(1)

        mu, log_sig = policy(state)
        pi = Normal(mu, log_sig.exp())
        loss = -torch.mean(pi.log_prob(action))

        p_optim.zero_grad()
        loss.backward()
        p_optim.step()

        ns, ns_std, rew = model(state, action)
        next_vals = model.reward_fun(torch.cat([next_state, next_action], axis=1))
        beta = Normal(ns, ns_std)


        model_loss = -torch.mean(beta.log_prob(next_state)) \
                    + 0.5 * torch.mean(torch.pow((reward+0.95*(1-done)*next_vals).detach()-rew,2))

        m_optim.zero_grad()
        model_loss.backward()
        m_optim.step()


if __name__ == '__main__':

    env_name = args.env
    try:
        env = envs.env_list[env_name](render=args.render)
    except TypeError as err:
        print('no argument render,  assumping env.render will just work')
        env = envs.env_list[env_name]()
    env.reset()
    env.render("human")

    assert np.any(np.abs(env.action_space.low) <= 1.) and  np.any(np.abs(env.action_space.high) <= 1.), 'Action space not normalizd'

    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d_%H-%M-%S/")

    path = './data/' + env_name +  '/' + 'demo_h_sac/' + date_str
    if os.path.exists(path) is False:
        os.makedirs(path)

    action_dim = env.action_space.shape[0]
    state_dim  = env.observation_space.shape[0]
    hidden_dim = 128

    policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim)

    model = Model(state_dim, action_dim, hidden_size=200)


    replay_buffer_size = 1000000
    replay_buffer = ReplayBuffer(replay_buffer_size)

    model_replay_buffer = SARSAReplayBuffer(replay_buffer_size)

    model_optim = ModelOptimizer(model, model_replay_buffer, lr=args.model_lr)

    sac = SoftActorCritic(policy=policy_net,
                          state_dim=state_dim,
                          action_dim=action_dim,
                          replay_buffer=replay_buffer,
                          policy_lr=args.policy_lr,
                          value_lr=args.value_lr,
                          soft_q_lr=args.soft_q_lr)

    planner = PathIntegral(model, policy_net,
                samples=args.trajectory_samples, t_H=args.horizon, lam=0.1)

    demos = pickle.load(open('../expert_demonstrators/ant_demos.pkl', 'rb'))
    expert_batch = random.sample(demos, 256)
    for sarsa_pair in expert_batch:
        _s, _a, _r, _ns, _na, _d = sarsa_pair
        model_replay_buffer.push(*sarsa_pair)
        replay_buffer.push(_s, _a, _r,  _ns, _d)

    print('learning from experts')
    expert_iters = 500
    update_models_from_demos(demos, model, policy_net, iters=expert_iters)
        # sac.soft_q_update(128)
        # model_optim.update_model(128, mini_iter=args.model_iter)
    print('ok, enough, I am master now')

    max_frames  = args.max_frames
    max_steps   = args.max_steps
    frame_skip  = args.frame_skip

    frame_idx   = 0
    rewards     = []
    batch_size  = 128

    ep_num = 0
    while frame_idx < max_frames:
        state = env.reset()
        planner.reset()

        action = planner(state)

        episode_reward = 0
        for step in range(max_steps):

            for _ in range(frame_skip):
                next_state, reward, done, _ = env.step(action.copy())
            next_action = planner(next_state)
            replay_buffer.push(state, action, reward, next_state, done)
            model_replay_buffer.push(state, action, reward, next_state, next_action, done)

            if len(replay_buffer) > batch_size:
                sac.soft_q_update(batch_size)
                model_optim.update_model(batch_size, mini_iter=args.model_iter)

            state = next_state
            action = next_action
            episode_reward += reward
            frame_idx += 1

            if args.render:
                env.render("human")


            if frame_idx % int(max_frames/10) == 0 and len(rewards)>0:
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
        ep_num += 1

        if len(replay_buffer) > batch_size:
            print('ep rew', ep_num, episode_reward)

    print('saving final data set')
    pickle.dump(rewards, open(path + 'reward_data'+ '.pkl', 'wb'))
    torch.save(policy_net.state_dict(), path + 'policy_' + 'final' + '.pt')
