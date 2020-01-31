import os

from gym import error, spaces
from gym.utils import seeding
import numpy as np
from os import path
import gym

# try:
#     from mujoco_py import MjSim, MjSimPool, load_model_from_path
# except ImportError as e:
#     raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))
from mujoco_py import MjSim, MjSimPool, load_model_from_path


class MujocoMultEnv(object):
    """
    Custom class for MPPI which simulates a set of multiple mujoco envs
    in parallel
    """

    def __init__(self, model_path, frame_skip, num_sims):
        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)
        if not path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)

        self.frame_skip = frame_skip
        self.model = load_model_from_path(fullpath)

        self.act_mid = np.mean(self.model.actuator_ctrlrange, axis=1)
        self.act_rng = 0.5 * (self.model.actuator_ctrlrange[:,1]-self.model.actuator_ctrlrange[:,0])

        self.pool = [MjSim(self.model) for _ in range(num_sims)]
        self.pool = MjSimPool(self.pool, nsubsteps=frame_skip)

        self.data = [sim.data for sim in self.pool.sims]

        self.init_qpos = self.pool.sims[0].data.qpos.ravel().copy()
        self.init_qvel = self.pool.sims[0].data.qvel.ravel().copy()

        # observation, _reward, done, _info = self.step(np.zeros(self.model.nu))
        # assert not done
        # self.obs_dim = observation.size

        low    = -np.ones(self.act_rng.shape)
        high   = np.ones(self.act_rng.shape)
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # high = np.inf*np.ones(self.obs_dim)
        # low = -high
        # self.observation_space = spaces.Box(low, high, dtype=np.float32)

    def scale_action(self, a):
        ## first clip
        a = np.clip(a, -1.0, 1.0)
        ## now scale
        a = self.act_mid + a * self.act_rng
        return a

    def set_state(self, state):
        # reset each of the simulators initial state
        for sim in self.pool.sims:
            sim.set_state(state)
        self.pool.forward()

    def reset(self):
        self.pool.reset()
        self.pool.forward()

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def do_simulation(self, ctrl):
        for i, data in enumerate(self.data):
            data.ctrl[:] = self.scale_action(ctrl[i].copy())
        self.pool.step()
