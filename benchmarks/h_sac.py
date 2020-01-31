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

from copy import copy, deepcopy

parser = argparse.ArgumentParser()
# parser.add_argument('--env',        type=str,   help=envs.getlist())
parser.add_argument('--max_steps',  type=int,   default=500)
parser.add_argument('--max_frames', type=int,   default=10000)
parser.add_argument('--frame_skip', type=int,   default=2)
parser.add_argument('--model_lr',   type=float, default=3e-4)
parser.add_argument('--policy_lr',  type=float, default=3e-3)
parser.add_argument('--value_lr',   type=float, default=3e-4)
parser.add_argument('--soft_q_lr',  type=float, default=3e-4)

parser.add_argument('--horizon', type=int, default=5)
parser.add_argument('--model_iter', type=int, default=1)
parser.add_argument('--trajectory_samples', type=int, default=20)
parser.add_argument('--lam',  type=float, default=1)


# parser.add_argument('--done_util', dest='done_util', action='store_true')
# parser.add_argument('--no_done_util', dest='done_util', action='store_false')
# parser.set_defaults(done_util=True)

# parser.add_argument('--render', dest='render', action='store_true')
# parser.add_argument('--no_render', dest='render', action='store_false')
# parser.set_defaults(render=False)

args = parser.parse_args()

# sawyer additions
import rospy
from sawyer.msg import RelativeMove
from geometry_msgs.msg import Pose2D
from std_srvs.srv import Empty, EmptyResponse
import tf
import time
from intera_core_msgs.msg import EndpointState
# from intera_interface import Limb

class sawer_env(object):
    def __init__(self):
        # set up ros
        self.move = rospy.Publisher('/puck/relative_move',RelativeMove,queue_size=1)
        self.reset_arm = rospy.ServiceProxy('/puck/reset', Empty)
        rospy.wait_for_service('/puck/reset', 5.0)
        self.listener = tf.TransformListener()
        s = rospy.Service('/puck/done', Empty, self.doneCallback)

        # set up sawyer
        self.limb = rospy.Subscriber("/robot/limb/right/endpoint_state", EndpointState, self.check_workspace)
        self.wall = False

        # set up tf
        target_transform = self.setup_transform_between_frames( 'target','block2')
        ee_transform = self.setup_transform_between_frames('target','ee')
        try:
            self.state = np.array([ee_transform[0],ee_transform[1],target_transform[0],target_transform[1]])
        except:
            print("Check that all april tags are visible")

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

            self.state = (1-0.8)*self.state + 0.8*np.array([ee_transform[0],ee_transform[1],target_transform[0],target_transform[1]])
            # state = np.array([dx_targetToArm, dy_targetToArm, dx_targetToBlock, dy_targetToBlock])
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            pass


    def reset(self):
        resp = self.reset_arm()
        self.get_transforms()
        return self.state.copy()

    def step(self, _a):
        # check if in workspace
        # wall = self.check_workspace()
        # wall = False
        if (self.wall == False):
            theta = (np.pi/4)*np.clip(_a[2],-1,1)  # keep april tags in view
            action = 0.2*np.clip(_a, -1,1)
            # publishes action input
            pose = RelativeMove()
            pose.dx = action[0]
            pose.dy = action[1]
            pose.dtheta = theta
            self.move.publish(pose)
            # gets the new state
            self.get_transforms()
            reward, done = self.reward_function()
        else:
            done = True
            reward = -100
        return self.state.copy(), reward, done

        # returns reward state and if it's outside bounds
    def reward_function(self):
        [dx_targetToArm, dy_targetToArm, dx_targetToBlock, dy_targetToBlock] = self.state.copy()

        arm_to_block = np.sqrt((dx_targetToArm-dx_targetToBlock)**2+
                        (dy_targetToArm-dy_targetToBlock)**2)

        # block_to_target = np.sqrt(dx_targetToBlock**2+dy_targetToBlock**2)
        reward = 0
        done = False
        thresh = 0.08
        # if (arm_to_block > thresh):
        reward += -arm_to_block
        if (arm_to_block < 0.15):
            reward += 1
        # reward += -block_to_target

        if (arm_to_block < thresh):
        # if (block_to_target < thresh):
            done = True
            reward += 10
            print('Reached goal!')

        # rospy.loginfo("arm_to_block: %f, block_to_target: %f, reward: %f", arm_to_block, block_to_target, reward)
        rospy.loginfo("action reward: %f", reward)
        rospy.loginfo("block dist: %f", arm_to_block)
        # rospy.loginfo("target dist: %f", block_to_target)

        return reward, done

    def check_workspace(self,current_pose):
        # current_pose = Limb().tip_state('right_hand').pose # get current state
        # make sure ee stays in workspace
        if ((current_pose.pose.position.x > 0.85) or (current_pose.pose.position.x < 0.45)
        or (current_pose.pose.position.y > 0.3) or (current_pose.pose.position.y < -0.25)):
            wall = True
            print('edge of workspace')
        else:
            wall = False
        self.wall = wall
        # return wall

    def doneCallback(self,req):
        self.wall = True
        print('manual done called')
        return EmptyResponse()

if __name__ == '__main__':
    try:
        rospy.init_node('h_sac')

        env = sawer_env()

        env_name = 'sawyer'
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d_%H-%M-%S/")

        path = './data/' + env_name +  '/' + 'h_sac/' + date_str
        if os.path.exists(path) is False:
            os.makedirs(path)

        action_dim = 3 # env.action_space.shape[0]
        state_dim  = 4 # env.observation_space.shape[0]
        hidden_dim = 128

        policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim)

        model = Model(state_dim, action_dim, def_layers=[200])

        replay_buffer_size = 10000
        replay_buffer = ReplayBuffer(replay_buffer_size)

        model_replay_buffer = SARSAReplayBuffer(replay_buffer_size)
        model_optim = ModelOptimizer(model, model_replay_buffer, lr=args.model_lr, lam=0.)

        sac = SoftActorCritic(policy=policy_net,
                              state_dim=state_dim,
                              action_dim=action_dim,
                              replay_buffer=replay_buffer,
                              policy_lr=args.policy_lr,
                              value_lr=args.value_lr,
                              soft_q_lr=args.soft_q_lr)

        planner = PathIntegral(model, policy_net, samples=args.trajectory_samples, t_H=args.horizon, lam=args.lam)

        max_frames = args.max_frames
        max_steps  = args.max_steps
        frame_skip = args.frame_skip

        frame_idx  = 0
        rewards    = []
        batch_size = 128

        ep_num = 0

        rate=rospy.Rate(20)

        while frame_idx < max_frames:
            state = env.reset()
            planner.reset()

            action = planner(state.copy())

            episode_reward = 0
            for step in range(max_steps):
                if np.isnan(action).any():
                    print('got nan')
                    print(replay_buffer.buffer)
                    env.reset()
                    os._exit(0)
                next_state, reward, done = env.step(action.copy())

                start_time = time.time()
                # print('state',next_state)
                next_action = planner(next_state)
                # print('elapsed time',time.time()-start_time)
                print(step)

                replay_buffer.push(state, action, reward, next_state, done)
                model_replay_buffer.push(state, action, reward, next_state, next_action, done)

                if len(replay_buffer) > batch_size:
                    # if frame_idx > 20:
                    # for _ in range(10):
                    sac.soft_q_update(batch_size);
                    model_optim.update_model(batch_size, mini_iter=args.model_iter)

                state = next_state
                action = next_action
                episode_reward += reward
                frame_idx += 1

                # if args.render:
                #     env.render("human"
                print(len(rewards), len(model_optim.log['rew_loss']))
                print(episode_reward)
                if (frame_idx % int(max_frames/20) == 0) and (len(replay_buffer) > batch_size):
                    # print(
                    #     'frame : {}/{}, \t last rew : {}, \t rew loss : {}'.format(
                    #         frame_idx, max_frames, rewards[-1][1], model_optim.log['rew_loss'][-1]
                    #     )
                    # )
                    start_time = time.time()
                    pickle.dump(rewards, open(path + 'reward_data' + '.pkl', 'wb'))
                    torch.save(policy_net.state_dict(), path + 'policy_' + str(frame_idx) + '.pt')
                    end_time = time.time()
                    print('pickle elapsed time', start_time)
                if done:
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

    except KeyboardInterrupt, rospy.ROSInterruptException:
        # env.reset()
        os._exit(0)
