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

def mppi(state, model, u_seq, horizon, lam=0.2, sig=0.1):
    assert len(u_seq) == horizon

    model.set_state(state)

    s   = []
    eps = []
    for t in range(horizon):
        eps.append(np.random.normal(0., sig, size=(model.num_sims, model.action_space.shape[0])))
        rew = model.step(u_seq[t] + eps[-1])
        s.append(rew)

    s = np.cumsum(s[::-1], 0)[::-1, :]

    for t in range(horizon):
        #s[t] -= np.min(s[t])
        s[t] -= np.max(s[t])
        w = np.exp(s[t]/lam) + 1e-4 # offset
        w /= np.sum(w)
        u_seq[t] = u_seq[t] + np.dot(w, eps[t])
    return savgol_filter(u_seq, horizon-1, 6, axis=0)

def main():
    # env   = CubeCatchEnv()
    # model = CubeCatchMultEnv(num_sims=10)
    env   = CubeManipEnv()
    model = CubeManipMultEnv(num_sims=10)

    horizon = 40
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
    while succ_trials < max_iter:
        print('succ attempt {} out of {}, max {}'.format(succ_trials, attempts, max_iter))
        obs = env.reset()
        print(obs.shape)
        # model.reset_model(env.target_pos, env.target_config)
        model.reset_model(env.target_config)
        # local_log = {'states' : [], 'actions' : [], 'next_states' : []}
        local_log = []
        state = env.get_state()
        u_seq = [np.zeros(model.action_space.shape[0]) for _ in range(horizon)]
        for _ in range(150):
            state = env.get_state()
            u_seq = mppi(state, model, u_seq, horizon, sig=0.4, lam=.2)
            #eps = np.random.normal(0., 0.1, size=(num_actions,))
            action = env.action_space.sample()
            next_obs, rew, done, _ = env.step(u_seq[0])
            #env.render()
            if done: break

            # local_log['states'].append(obs + np.random.normal(0., 0.01, size=obs.shape))
            # local_log['next_states'].append(next_obs + np.random.normal(0., 0.01, size=obs.shape))
            # local_log['actions'].append(u_seq[0].copy())
            local_log.append((obs.copy(), u_seq[0].copy(), rew, next_obs.copy(), u_seq[1].copy(), done))

            obs = next_obs

            # env.render()
            u_seq[:-1] = u_seq[1:]
            u_seq[-1]  = np.zeros(env.action_space.shape[0])


        log.append(local_log)
        succ_trials += 1
        attempts += 1
        pickle.dump(log, open( "./data/shadow_hand/cube_manip/demonstrations.pkl", "wb" ) )


if __name__ == '__main__':
    main()
