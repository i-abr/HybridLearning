import sys
import os
import numpy as np
import gym
import pybullet_envs
import pickle
import time
from datetime import datetime

import torch
import torch.optim as optim
from torch.distributions import Normal

from policynetwork import Policy
from model import Model

from hybrid_stochastic import PathIntegral

from jacobian import jacobian

import random

import matplotlib.pyplot as plt


def update_models_from_demos(demos, model, m_optim, policy, p_optim, iters=200, samples=32):


    for k in range(iters):

        expert_batch = random.sample(demos, samples)

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


def test(env, policy):
    ep_rew = 0.
    state = env.reset()
    for t in range(1000):
        a = policy.get_action(state)
        state, rew, done, _ = env.step(a)
        env.render("human")
        ep_rew += rew
        if done:
            break
    return ep_rew

if __name__ == "__main__":

    env = gym.make("AntBulletEnv-v0")
    # plt.ion()
    env.render(mode="human")

    demos = pickle.load(open('../expert_demonstrators/ant_demos.pkl', 'rb'))


    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    policy = Policy(state_dim, action_dim)
    model = Model(state_dim, action_dim)

    planner = PathIntegral(model, policy, samples=40, t_H=5)

    p_optim = optim.Adam(policy.parameters(),  lr=3e-3)
    m_optim = optim.Adam(model.parameters(),   lr=3e-3)

    log = []
    # ep_rew = 0.
    # state = env.reset()
    for t in range(20):
        # a = planner.get_action(state)
        # state, rew, done, _ = env.step(a)
        # env.render("human")
        # ep_rew += rew

        update_models_from_demos(demos, model, m_optim, policy, p_optim, iters=32, samples=128)
        ep_rew = test(env, planner)
        log.append(ep_rew)
        print(ep_rew)
        # if done:
        #     break

    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d_%H-%M-%S")

    path = './data/hybrid_control/'
    if os.path.exists(path) is False:
        os.makedirs(path)
    pickle.dump(log, open(path + 'rew_' + date_str + '.pkl', 'wb'))
    # batch_size = 256
    # for k in range(600):
    #
    #     batch = random.sample(demos, batch_size)
    #     state, action, reward, next_state, next_action, done = map(np.stack, zip(*batch))
    #
    #     state = torch.FloatTensor(state)
    #     next_state = torch.FloatTensor(next_state)
    #     action = torch.FloatTensor(action)
    #     next_action = torch.FloatTensor(next_action)
    #     reward = torch.FloatTensor(reward).unsqueeze(1)
    #     done    = torch.FloatTensor(np.float32(done)).unsqueeze(1)
    #
    #     mu, sig = policy(state)
    #     pi = Normal(mu, sig)
    #
    #     loss = -torch.mean(pi.log_prob(action))
    #
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    #
    #
    #     state.requires_grad = True
    #
    #     ns, ns_std, rew = model(state, action)
    #
    #     df = jacobian(ns, state)
    #
    #     next_vals = model.reward_fun(torch.cat([next_state, next_action], axis=1))
    #
    #     beta = Normal(ns, ns_std)
    #
    #     model_loss = -torch.mean(beta.log_prob(next_state)) \
    #                 + 0.5 * torch.mean(torch.pow((reward+0.95*(1-done)*next_vals).detach()-rew,2)) \
    #                 + 1e-1 * torch.norm(df, dim=[1,2]).mean()
    #
    #     m_optimizer.zero_grad()
    #     model_loss.backward()
    #     m_optimizer.step()
    #
    #     if k % 50 == 0:
    #         print(k, loss.item(), model_loss.item(), test(env, planner), test(env, policy))



    # state = env.reset()
    #
    # for t in range(1000):
    #     # time.sleep(1. / 60.)
    #
    #     # a = policy.get_action(state)
    #
    #     a = planner(state)
    #
    #     state, rew, done, _ = env.step(a)
    #
    #     still_open = env.render("human")
    #     # plt.clf()
    #     # plt.imshow(still_open)
    #     # plt.pause(0.1)
    #     if not done: continue
    #     if still_open == False:
    #         break
