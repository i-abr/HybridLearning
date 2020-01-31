import numpy as np
from gym import utils
from .mujoco_multenv import MujocoMultEnv

from .quatmath import quat2euler, euler2quat, mat2quat
from mujoco_py import MjViewer
import os

class CubeManipMultEnv(MujocoMultEnv):# TODO: verify the ->, utils.EzPickle): functionality

    def __init__(self, frame_skip=5, num_sims=10):

        self.num_sims = num_sims

        self.frame_skip = frame_skip

        self.target_obj_bid = 0
        self.target_obj_sid = 0
        self.obj_sid        = 0
        self.obj_bid        = 0
        self.palm_sid       = 0

        file_dir = os.path.dirname(os.path.abspath(__file__))
        MujocoMultEnv.__init__(self,
            file_dir + '/shadow_hand_assets/hand/manipulate_block.xml',
            frame_skip, num_sims)

        self.target_obj_bid = self.model.body_name2id('target')
        self.target_obj_sid = self.model.site_name2id('target:center')
        self.obj_sid        = self.model.site_name2id('object:center')
        self.obj_bid        = self.model.body_name2id('object')
        self.palm_sid       = self.model.site_name2id('robot0:Palm')

        # utils.EzPickle.__init__(self)

    def step(self, a):
        self.do_simulation(a)
        rewards = []
        for data in self.data:
            reward = self._get_cost(data)
            rewards.append(reward)
        return rewards

    def reset_model(self, target_config):

        for sim in self.pool.sims:
            sim.reset()
            sim.model.body_quat[self.target_obj_bid] = target_config
            #sim.model.body_pos[self.target_obj_bid] = target_pos
            sim.forward()

    def _get_cost(self, data):
        done = False
        obj_pos     = data.site_xpos[self.obj_sid].ravel()
        obj_config  = data.site_xmat[self.obj_sid].reshape((3,3))

        palm_pos = data.site_xpos[self.palm_sid].ravel()
        vel = data.qvel[-6:].ravel()
        dist = 0.5*np.linalg.norm(palm_pos - obj_pos)

        desired_config = data.body_xmat[self.target_obj_bid].reshape((3,3))
        desired_pos    = data.body_xpos[self.target_obj_bid].ravel()

        cost = dist
        tr_RtR = np.trace(obj_config.T.dot(desired_config))
        _arc_c_arg = (tr_RtR - 1)/2.0
        _th = np.arccos(_arc_c_arg)
        #_th = np.linalg.norm(obj_config.ravel() - desired_config.ravel())
        orien_err = _th**2

        #if dist < 0.015:
        #    cost += dist2targ
        #    if dist2targ < 0.02:
        #        cost += orien_err

        return -10.*cost - orien_err

    # TODO: verify that I even need to get a get_obs method for the model
    # def _get_obs(self):
    #
    #     qp   = self.data.qpos.ravel()
    #     qvel = self.data.qvel.ravel()
    #
    #     obj_pos     =  self.data.body_xpos[self.obj_bid].ravel()
    #     obj_orien   =  self.data.body_xmat[self.obj_bid].reshape(3,3)
    #     obj_vel     =  self.data.qvel[-6:].ravel()
    #
    #
    #     desired_orien = self.data.site_xmat[self.target_obj_sid].reshape(3,3)
    #     palm_pos      = self.data.site_xpos[self.palm_sid].ravel()
    #
    #
    #     return np.concatenate([qp[:-6], obj_pos, obj_vel, obj_orien.ravel(), desired_orien.ravel(),
    #                             obj_pos - palm_pos, np.ravel(obj_orien.T.dot(desired_orien))])
