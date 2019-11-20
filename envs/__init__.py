from pybullet_envs import gym_pendulum_envs, gym_manipulator_envs, gym_locomotion_envs
from gym.envs import classic_control, box2d

env_list = {
    'InvertedPendulumSwingupBulletEnv' : gym_pendulum_envs.InvertedPendulumSwingupBulletEnv,
    'HalfCheetahBulletEnv' : gym_locomotion_envs.HalfCheetahBulletEnv,
    'HopperBulletEnv' : gym_locomotion_envs.HopperBulletEnv,
    'AntBulletEnv' : gym_locomotion_envs.AntBulletEnv,
    'ReacherBulletEnv' : gym_manipulator_envs.ReacherBulletEnv,
    'StrikerBulletEnv' : gym_manipulator_envs.StrikerBulletEnv,
    'PendulumEnv' : classic_control.PendulumEnv,
    'LunarLanderContinuousEnv' : box2d.LunarLanderContinuous
}

def getlist():
    out_str = ''
    for env_name in env_list.keys():
        out_str += env_name + '\n'
    return out_str
