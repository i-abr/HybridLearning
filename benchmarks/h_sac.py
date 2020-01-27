import numpy as np
import pickle
from datetime import datetime

import sys
import os
sys.path.append('../')

# local imports
# import envs

import torch
from sac import SoftActorCritic
from sac import PolicyNetwork
from sac import ReplayBuffer
# from sac import NormalizedActions
from hybrid_stochastic import PathIntegral
from model import ModelOptimizer, Model, SARSAReplayBuffer
# from model import MDNModelOptimizer, MDNModel
# argparse things
import argparse

parser = argparse.ArgumentParser()
# parser.add_argument('--env',        type=str,   help=envs.getlist())
parser.add_argument('--max_steps',  type=int,   default=1000)
parser.add_argument('--max_frames', type=int,   default=10000)
parser.add_argument('--frame_skip', type=int,   default=2)
parser.add_argument('--model_lr',   type=float, default=3e-3)
parser.add_argument('--policy_lr',  type=float, default=3e-4)
parser.add_argument('--value_lr',   type=float, default=3e-4)
parser.add_argument('--soft_q_lr',  type=float, default=3e-4)

parser.add_argument('--horizon', type=int, default=5)
parser.add_argument('--model_iter', type=int, default=2)
parser.add_argument('--trajectory_samples', type=int, default=20)
parser.add_argument('--lam',  type=float, default=0.1)


parser.add_argument('--done_util', dest='done_util', action='store_true')
parser.add_argument('--no_done_util', dest='done_util', action='store_false')
parser.set_defaults(done_util=True)

# parser.add_argument('--render', dest='render', action='store_true')
# parser.add_argument('--no_render', dest='render', action='store_false')
# parser.set_defaults(render=False)

args = parser.parse_args()

# sawyer
import rospy
from sawyer.msg import RelativeMove
from geometry_msgs.msg import Pose2D
from std_srvs.srv import Empty, EmptyResponse
import tf
import time

class sawer_env(object):
    def __init__(self):
        self.move = rospy.Publisher('/puck/relative_move',RelativeMove,queue_size=1)
        self.reset_arm = rospy.ServiceProxy('/puck/reset', Empty)
        rospy.wait_for_service('/puck/reset', 5.0)
        self.listener = tf.TransformListener()

        # set up tf
        target_transform = self.setup_transform_between_frames( 'target','block2')
        ee_transform = self.setup_transform_between_frames('target','ee')
        self.state = np.array([ee_transform[0],ee_transform[1],target_transform[0],target_transform[1]])

    def setup_transform_between_frames(self, reference_frame, target_frame):
        time_out = 0.5
        start_time = time.time()
        while(True):
            try:
                translation, rot_quaternion = self.listener.lookupTransform(reference_frame, target_frame, rospy.Time(0))
                break
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                if((time.time()- start_time) > time_out):
                    return None
        return translation

    def get_transforms(self):
        try:
            target_transform, _ = self.listener.lookupTransform( 'target','block2', rospy.Time(0))
            ee_transform, _ = self.listener.lookupTransform( 'target','ee', rospy.Time(0))

            self.state = np.array([ee_transform[0],ee_transform[1],target_transform[0],target_transform[1]])
            # state = np.array([dx_targetToArm, dy_targetToArm, dx_targetToBlock, dy_targetToBlock])
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            pass

    def reset(self):
        resp = self.reset_arm()
        self.get_transforms()
        return self.state

    def step(self, _a):
        action = 0.2*np.clip(_a, -1,1)
        # publishes action input
        pose = RelativeMove()
        pose.dx = action[0]
        pose.dy = action[1]
        self.move.publish(pose)
        # gets the new state
        self.get_transforms()
        reward = self.reward_function()
        if (reward > -0.25):
            done = 1
        else:
            done = 0
        return self.state, reward, done

        # returns reward state and if it's outside bounds
    def reward_function(self):
        [dx_targetToArm, dy_targetToArm, dx_targetToBlock, dy_targetToBlock] = self.state

        arm_to_block = -((dx_targetToArm-dx_targetToBlock)**2+
                        (dy_targetToArm-dy_targetToBlock)**2)*100

        block_to_target = -(dx_targetToBlock**2+dy_targetToBlock**2)*100

        if (arm_to_block > -.25):
            reward = arm_to_block
        else:
            reward = block_to_target

        return reward

class h_sac(object):
    def doneCallback(self,req):
        self.done = 1
        print('manual done called')
        return EmptyResponse()

    def main(self):
        env = sawer_env()
        manual_done = rospy.Service('/puck/done', Empty, self.doneCallback)

        env_name = 'sawyer'
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d_%H-%M-%S/")

        path = './data/' + env_name +  '/' + 'h_sac/' + date_str
        if os.path.exists(path) is False:
            os.makedirs(path)


        action_dim = 2 # env.action_space.shape[0]
        state_dim  = 4 # env.observation_space.shape[0]
        hidden_dim = 128

        policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim)

        model = Model(state_dim, action_dim, def_layers=[200])

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

        planner = PathIntegral(model, policy_net, samples=args.trajectory_samples, t_H=args.horizon, lam=args.lam)

        max_frames  = args.max_frames
        max_steps   = args.max_steps
        frame_skip = args.frame_skip

        frame_idx   = 0
        rewards     = []
        batch_size  = 128

        ep_num = 0

        rate=rospy.Rate(10)

        while frame_idx < max_frames:
            state = env.reset()
            planner.reset()

            action = planner(state)

            episode_reward = 0
            for step in range(max_steps):
                next_state, reward, self.done = env.step(action.copy())


                next_action = planner(next_state)

                replay_buffer.push(state, action, reward, next_state, self.done)
                model_replay_buffer.push(state, action, reward, next_state, next_action, self.done)

                if len(replay_buffer) > batch_size:
                    sac.soft_q_update(batch_size)
                    model_optim.update_model(batch_size, mini_iter=args.model_iter)

                state = next_state
                action = next_action
                episode_reward += reward
                frame_idx += 1

                # if args.render:
                #     env.render("human")

                if frame_idx % int(max_frames/10) == 0:
                    print(
                        'frame : {}/{}, \t last rew : {}, \t rew loss : {}'.format(
                            frame_idx, max_frames, rewards[-1][1], model_optim.log['rew_loss'][-1]
                        )
                    )

                    pickle.dump(rewards, open(path + 'reward_data' + '.pkl', 'wb'))
                    torch.save(policy_net.state_dict(), path + 'policy_' + str(frame_idx) + '.pt')

                if self.done:
                    print('done loop')
                    break
                else:
                    rate.sleep()
            #if len(replay_buffer) > batch_size:
            #    for k in range(200):
            #        sac.soft_q_update(batch_size)
            #        model_optim.update_model(batch_size, mini_iter=1)#args.model_iter)
            if len(replay_buffer) > batch_size:
                print('ep rew', ep_num, episode_reward, model_optim.log['rew_loss'][-1], model_optim.log['loss'][-1])
                print('ssac loss', sac.log['value_loss'][-1], sac.log['policy_loss'][-1], sac.log['q_value_loss'][-1])
            rewards.append([frame_idx, episode_reward])
            ep_num += 1
        print('saving final data set')
        pickle.dump(rewards, open(path + 'reward_data'+ '.pkl', 'wb'))
        torch.save(policy_net.state_dict(), path + 'policy_' + 'final' + '.pt')


if __name__ == '__main__':
    rospy.init_node('h_sac')
    test = h_sac()
    test.main()
