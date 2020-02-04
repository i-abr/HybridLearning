import numpy as np
from gym import utils
from .mujoco_env import MujocoEnv
from .quatmath import quat2euler, euler2quat, mat2quat
from mujoco_py import MjViewer
import os

def robot_get_obs(sim):
    """Returns all joint positions and velocities associated with
    a robot.
    """
    if sim.data.qpos is not None and sim.model.joint_names:
        names = [n for n in sim.model.joint_names if n.startswith('robot')]
        return (
            np.array([sim.data.get_joint_qpos(name) for name in names]),
            np.array([sim.data.get_joint_qvel(name) for name in names]),
        )
    return np.zeros(0), np.zeros(0)

class CubeManipEnv(MujocoEnv, utils.EzPickle):

    def __init__(self, frame_skip=10, stabilize=True):

        self.frame_skip = frame_skip
        self.stabilize   = stabilize

        self.target_obj_bid = 0
        self.target_obj_sid = 0
        self.obj_sid        = 0
        self.obj_bid        = 0
        self.palm_sid       = 0

        self.target_config = np.array([0.,0.,0.,1.])

        file_dir = os.path.dirname(os.path.abspath(__file__))
        MujocoEnv.__init__(self, file_dir + '/shadow_hand_assets/hand/manipulate_block.xml', frame_skip)
        self.target_obj_bid = self.model.body_name2id('target')
        self.target_obj_sid = self.model.site_name2id('target:center')
        self.obj_sid        = self.model.site_name2id('object:center')
        self.obj_bid        = self.model.body_name2id('object')
        self.palm_sid       = self.model.site_name2id('robot0:Palm')

        utils.EzPickle.__init__(self)

        self.act_mid = np.mean(self.model.actuator_ctrlrange, axis=1)
        self.act_rng = 0.5 * (self.model.actuator_ctrlrange[:,1] - self.model.actuator_ctrlrange[:,0])

    def step(self, _a):
        a = np.clip(_a, -1.0, 1.0)
        try:
            a = self.act_mid + a * self.act_rng
        except:
            a = _a

        self.do_simulation(a, self.frame_skip)
        reward, done = self._get_reward()
        return self._get_obs(), reward, done, {}

    def reset_model(self):
        qp = self.init_qpos.copy()
        qv = self.init_qvel.copy()
        self.set_state(qp, qv)
        desired_orien = np.zeros(3)

        # rot = self.np_random.randint(-1,2, size=(3,)) * np.pi/2.0
        rot = np.array([-1.0,0.,0.]) * np.pi/2.0
        # desired_orien[0] = self.np_random.uniform(low=-1., high=1.)
        # desired_orien[1] = self.np_random.uniform(low=-1., high=1.)
        # desired_orien[2] = self.np_random.uniform(low=-0.2, high=0.2)
        desired_orien[:3] = rot[:]

        #self.model.body_pos[self.target_obj_bid][0] = target_pose[0]
        #self.model.body_pos[self.target_obj_bid][2] = target_pose[2]
        self.model.body_quat[self.target_obj_bid] = euler2quat(desired_orien)
        self.target_config = euler2quat(desired_orien)

        self.sim.forward()
        return self._get_obs()

    def is_successful(self):
        reward = 0.
        data = self.data

        done = False
        obj_pos     = data.site_xpos[self.obj_sid].ravel()
        obj_config  = data.site_xmat[self.obj_sid].reshape((3,3))

        if obj_pos[-1] < -0.2:
            done = True

        palm_pos = data.site_xpos[self.palm_sid].ravel()
        vel = data.qvel[-6:].ravel()
        dist = 0.5 * np.linalg.norm(palm_pos - obj_pos)

        desired_config = data.body_xmat[self.target_obj_bid].reshape((3,3))
        desired_pos    = data.body_xpos[self.target_obj_bid].ravel()



        tr_RtR = np.trace(obj_config.T.dot(desired_config))
        _arc_c_arg = (tr_RtR - 1)/2.0
        _th = np.arccos(_arc_c_arg)

        # print(_th)
        if _th < 0.08:
            return True
        else:
            return False

    def _get_reward(self):
        reward = 0.
        data = self.data

        done = False
        obj_pos     = data.body_xpos[self.obj_bid].ravel()
        # obj_config  = data.site_xmat[self.obj_sid].reshape((3,3))
        obj_config = data.body_xquat[self.obj_bid].ravel()

        # done = self.is_successful()
        done = False

        palm_pos = data.site_xpos[self.palm_sid].ravel()
        vel = data.qvel[-6:].ravel()
        dist = 0.5 * np.linalg.norm(palm_pos - obj_pos)

        # desired_config = data.body_xmat[self.target_obj_bid].reshape((3,3))
        desired_config = data.body_xquat[self.target_obj_bid].ravel()

        desired_pos    = data.body_xpos[self.target_obj_bid].ravel()


        # tr_RtR = np.trace(obj_config.T.dot(desired_config))
        # _arc_c_arg = (tr_RtR - 1)/2.0
        # _th = np.arccos(_arc_c_arg)
        _th = np.linalg.norm(desired_config - obj_config)
        reward = -10.*dist - _th

        #reward +=  -0.1*dist2targ
        #if dist2targ < 0.05:
        #    reward += 10.
        return reward, done

    def _get_obs(self):

        robot_qpos, robot_qvel = robot_get_obs(self.sim)
        object_qpos = self.sim.data.get_joint_qpos('object:joint')
        object_qvel = self.sim.data.get_joint_qvel('object:joint')
        # qp   = self.data.qpos.ravel()
        # qvel = self.data.qvel.ravel()

        # obj_pos     =  self.data.site_xpos[self.obj_sid].ravel()
        # obj_orien   =  self.data.site_xmat[self.obj_sid].reshape((3,3))
        # obj_vel     =  self.data.qvel[-6:].ravel()
        palm_pos = self.data.site_xpos[self.palm_sid].ravel()


        desired_orien = self.data.body_xmat[self.target_obj_bid].reshape((3,3))
        desired_pose  = self.data.body_xpos[self.target_obj_bid].ravel()
        palm_pos      = self.data.site_xpos[self.palm_sid].ravel()

        desired_config = self.data.body_xquat[self.target_obj_bid].ravel()

        desired_pos    = self.data.body_xpos[self.target_obj_bid].ravel()



        # tr_RtR = np.trace(obj_orien.T.dot(desired_orien))
        # _arc_c_arg = (tr_RtR - 1)/2.0
        # _th = np.arccos(_arc_c_arg)
        # return np.concatenate([robot_qpos, robot_qvel,
        #     object_qpos, object_qvel, self.target_config.copy()])
        return np.concatenate([self.data.qpos[:].copy(), self.data.qvel[:].copy(), palm_pos, desired_config])
        # return np.concatenate([qp[:-6], obj_pos-palm_pos, obj_vel, obj_orien.ravel()-desired_orien.ravel(), [_th]])
        #return np.concatenate([qp[:-6], obj_pos-desired_pose, obj_vel, obj_orien.ravel(), desired_orien.ravel(),
        #                        obj_pos-palm_pos, np.ravel(obj_orien.T.dot(desired_orien))])


    def gs(self):
        """
        Get state of hand as well as objects and targets in the scene
        """
        obj_pos     =  self.data.body_xpos[self.obj_bid].ravel()
        obj_orien   =  self.data.body_xmat[self.obj_bid].reshape(3,3)
        obj_vel     =  self.data.qvel[-6:].ravel()
        return dict(obj_pos=obj_pos, obj_vel=obj_vel, obj_orien=obj_orien)

    def mj_viewer_setup(self):
        self.viewer = MjViewer(self.sim)
        self.viewer._render_every_frame = True
