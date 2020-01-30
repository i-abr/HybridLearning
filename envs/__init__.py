from pybullet_envs import gym_pendulum_envs, gym_manipulator_envs, gym_locomotion_envs
from gym.envs import classic_control, box2d
from pybullet_envs.bullet.racecarGymEnv import RacecarGymEnv
from pybullet_envs.bullet.minitaur_gym_env import MinitaurBulletEnv
from rex_gym.envs.gym.galloping_env import RexReactiveEnv

from gym.envs.mujoco import ant_v3, half_cheetah_v3, swimmer_v3, pusher, striker, thrower, reacher3d

from .cube_manip import CubeManipEnv

env_list = {
    'InvertedPendulumSwingupBulletEnv' : gym_pendulum_envs.InvertedPendulumSwingupBulletEnv,
    'HalfCheetahBulletEnv' : gym_locomotion_envs.HalfCheetahBulletEnv,
    'HopperBulletEnv' : gym_locomotion_envs.HopperBulletEnv,
    'AntBulletEnv' : gym_locomotion_envs.AntBulletEnv,
    'ReacherBulletEnv' : gym_manipulator_envs.ReacherBulletEnv,
    'PendulumEnv' : classic_control.PendulumEnv,
    'LunarLanderContinuousEnv' : box2d.LunarLanderContinuous,
    'RexEnv' : RexReactiveEnv,
    'Swimmer-v3' : swimmer_v3.SwimmerEnv,
    'Pusher' : pusher.PusherEnv,
    'Striker' : striker.StrikerEnv,
    'Thrower' : thrower.ThrowerEnv,
    'Reacher3d' : reacher3d.Reacher3DEnv,
    'CubeManipEnv' : CubeManipEnv
}

def getlist():
    out_str = ''
    for env_name in env_list.keys():
        out_str += env_name + '\n'
    return out_str
