#! /usr/bin/env python3
import sys
sys.path.append('../')
from envs.shadow_hand.cube_manip import CubeManipEnv
from envs.shadow_hand.cube_manip_multenv import CubeManipMultEnv
#from envs.cube_catch_multenv import CubeCatchMultEnv
#from envs.cube_catch import CubeCatchEnv
import numpy as np
from scipy.signal import savgol_filter

import pickle

def mppi(state, model, u_seq, horizon, lam=0.8, sig=0.1):
    assert len(u_seq) == horizon

    model.set_state(state)

    s   = []
    eps = []
    for t in range(horizon):
        eps.append(
            np.random.normal(0., sig, size=(model.num_sims, model.action_space.shape[0]))
            # np.stack([np.linspace(-1,1, model.num_sims) for i in range(model.action_space.shape[0])])
        )
        # eps[-1][:,:5] = np.zeros((20,5))
        rew = model.step(u_seq[t] + eps[-1])
        s.append(rew)

    s = np.cumsum(s[::-1], 0)[::-1, :]

    for t in range(horizon):
        s[t] -= np.min(s[t])
        #s[t] -= np.max(s[t] - sig * np.sum(eps[t]**2))
        w = np.exp(s[t]/lam) + 1e-4 # offset
        w /= np.sum(w)
        u_seq[t] = u_seq[t] + np.dot(w, eps[t])
    return savgol_filter(u_seq, horizon-1, 3, axis=0)

class MPPI(object):

    def __init__(self, nsims=10):
        self.model = CubeManipMultEnv(num_sims=10)
        self.horizon = 20
        self.u_seq = [np.zeros(self.model.action_space.shape[0]) for _ in range(self.horizon)]
        self.out = np.zeros(self.model.action_space.shape[0])
    def reset(self):
        self.u_seq = [np.zeros(self.model.action_space.shape[0]) for _ in range(self.horizon)]
        self.out = np.zeros(self.model.action_space.shape[0])

    def __call__(self, state, lam=0.4, sig=0.1):
        # assert len(u_seq) == horizon

        self.u_seq[:-1] = self.u_seq[1:]
        self.u_seq[-1]  = np.zeros(self.model.action_space.shape[0])
        self.model.set_state(state)

        s   = []
        eps = []
        for t in range(self.horizon):
            eps.append(
                np.random.normal(0., sig, size=(self.model.num_sims, self.model.action_space.shape[0]))
                # np.stack([np.linspace(-1,1, model.num_sims) for i in range(model.action_space.shape[0])])
            )
            # eps[-1][:,:5] = np.zeros((20,5))
            rew = self.model.step(self.u_seq[t] + eps[-1])
            s.append(rew)

        s = np.cumsum(s[::-1], 0)[::-1, :]

        for t in range(self.horizon):
            s[t] -= np.min(s[t])
            #s[t] -= np.max(s[t] - sig * np.sum(eps[t]**2))
            w = np.exp(s[t]/lam) + 1e-4 # offset
            w /= np.sum(w)
            self.u_seq[t] = self.u_seq[t] + np.dot(w, eps[t])
        # self.u_seq = savgol_filter(self.u_seq, self.horizon-1, 3, axis=0)
        self.out = 0.2 * self.out + (1-0.2) * self.u_seq[0]
        return self.out.copy()

def main():
    # env   = CubeCatchEnv()
    # model = CubeCatchMultEnv(num_sims=10)
    env   = CubeManipEnv()
    # model = CubeManipMultEnv(num_sims=10)

    horizon = 20
    max_iter= 100
    num_actions = env.action_space.shape[0]
    log = []
    # {
    #         'states' : [],
    #         'actions' : [],
    #         'next_states' : []
    # }
    succ_trials = 0
    attempts    = 0
    mppi = MPPI()
    while succ_trials < max_iter:
        print('succ attempt {} out of {}, max {}'.format(succ_trials, attempts, max_iter))
        obs = env.reset()
        print(obs.shape)
        # model.reset_model(env.target_pos, env.target_config)
        mppi.model.reset_model(env.target_config)
        # local_log = {'states' : [], 'actions' : [], 'next_states' : []}
        local_log = []
        state = env.get_state()
        # u_seq = [np.zeros(model.action_space.shape[0]) for _ in range(horizon)]
        # u_seq = mppi(state, model, u_seq, horizon, sig=0.4, lam=.2)
        # action = u_seq[0].copy()
        action = mppi(state, sig=0.6)
        for _ in range(150):

            #eps = np.random.normal(0., 0.1, size=(num_actions,))
            next_obs, rew, done, _ = env.step(action)
            env.render()
            if done: break

            state = env.get_state()
            # u_seq = mppi(state, model, u_seq, horizon, sig=0.4, lam=.2)
            # next_action = u_seq[0].copy()

            next_action = mppi(state, sig=0.6)

            # local_log['states'].append(obs + np.random.normal(0., 0.01, size=obs.shape))
            # local_log['next_states'].append(next_obs + np.random.normal(0., 0.01, size=obs.shape))
            # local_log['actions'].append(u_seq[0].copy())
            local_log.append((obs.copy(),action.copy(), rew, next_obs.copy(), next_action.copy(), done))

            obs = next_obs
            action = next_action
            # print(state)
            # print(obs)

            # env.render()
            # u_seq[:-1] = u_seq[1:]
            # u_seq[-1]  = np.zeros(env.action_space.shape[0])


        log.append(local_log)
        succ_trials += 1
        attempts += 1
        pickle.dump(log, open( "./data/shadow_hand/cube_manip/demonstrations.pkl", "wb" ) )


if __name__ == '__main__':
    main()
