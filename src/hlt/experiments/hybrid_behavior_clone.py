#!/usr/bin/env python

import rospy
from copy import deepcopy, copy
import sys
import os
import signal
import traceback
import pickle
sys.path.append('../')
from hlt.msg import BlockStackState
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Joy
from bclib import BCOptimizer, Policy, ReplayBuffer
import numpy as np
from block_stack_env import BlockStackEnv
from hltlib import HybridPlanner, Model, ModelOptimizer, SARSAReplayBuffer

class HybridBehaviorCloneExperiment(object):
    """
    Experiment for running the behavior cloning
    """
    def __init__(self):

        state_dim = 7
        action_dim = 3

        self.policy = Policy(state_dim, action_dim, layers=[32, 24])
        capacity = 100000
        self.buffer     = ReplayBuffer(capacity)
        self.optimizer  = BCOptimizer(self.policy, self.buffer, lr=0.01)
        self.model = Model(state_dim, action_dim, hidden_dim=64)
        self.model_replay_buffer = SARSAReplayBuffer(capacity)
        self.model_optimizer = ModelOptimizer(self.model, self.model_replay_buffer, lr=0.01)
        self.planner = HybridPlanner(self.model, self.policy, samples=40, t_H=40, lam=1.0)
        self.cmd = Float32MultiArray()
        self.cmd.data = [0., 0., 0., 0., 0., 0.]
        self.raw_expert_cmd = np.array([0., 0., 0.])
        self.raw_ready_button = 0
        self.ready = False
        self.env = BlockStackEnv()

        rospy.Subscriber('/joy', Joy, self.expert_callback)

        self.rate = rospy.Rate(20)
        self.log = {'reward' : []}

        self.expert_demonstrations = pickle.load(open('./data/human_demos/demonstrations.pkl', 'rb'))

    def __filter(self, cmd):
        for i in range(3): # we are only controlling xyz
            self.cmd.data[i] = self.alpha * self.cmd.data[i] \
                        + (1-self.alpha) * (self.velocity_scale * cmd[i])

    def expert_callback(self, msg):
        self.raw_expert_cmd[0]= -msg.axes[1]
        self.raw_expert_cmd[1]= -msg.axes[0]
        self.raw_expert_cmd[2]=  msg.axes[4]
        self.raw_ready_button = int(msg.buttons[2])
        self.expert_interevention = True

    def run(self):
        frame_idx = 0
        max_frames = 10000
        max_steps = 200
        batch_size = 128
        episode = 0
        while not rospy.is_shutdown() and frame_idx < max_frames:

            ##### expert demonstration #####
            # wait for expert to be ready
            # state = self.env.reset()
            print('running expert demo')
            done = False
            for (state, action, next_state, action) in self.expert_demonstrations[episode]:
                self.buffer.push(state, action, next_state, action)
                reward = self.env.get_reward(state, action)
                self.model_replay_buffer.push(state, action, reward, next_state, action, done)
            # rospy.sleep(1)
            # action = self.raw_expert_cmd.copy()
            # ep_rew = 0.
            # for step in range(max_steps):
            #     next_state, reward, done, _ = self.env.step(action.copy())
            #     print next_state
            #     ep_rew += reward
            #     next_action = self.raw_expert_cmd.copy()
            #     self.buffer.push(state, action, next_state, action)
            #     self.model_replay_buffer.push(state, action, reward, next_state, next_action, done)
            #     state = next_state.copy()
            #     action = next_action.copy()
            #     self.rate.sleep()
            #     if done:
            #         self.ready = not self.ready
            #         break
            # frame_idx += 1
            # print('frame', frame_idx, 'ep rew', ep_rew, 'steps taken in demo', step)

            self.optimizer.update_policy(batch_size, epochs=100, verbose=True)
            self.model_optimizer.update_model(batch_size, epochs=100, verbose=True)

            state = self.env.reset()
            print('running learn demo')
            rospy.sleep(2)
            self.expert_interevention = False
            action = self.planner(state)
            ep_rew = 0.
            for step in range(max_steps):

                action = self.planner(state)

                next_state, reward, done, _ = self.env.step(action.copy())
                print next_state
                ep_rew += reward

                self.model_replay_buffer.push(state, action, reward, next_state, action, done)
                state = next_state
                self.rate.sleep()
                # self.optimizer.update_policy(batch_size, epochs=1, verbose=True)
                # self.model_optimizer.update_model(batch_size, epochs=1, verbose=True)
                if self.expert_interevention:
                    print 'human intervention'
                    self.ready = not self.ready
                    break
                    # action = self.raw_expert_cmd.copy()
                    # self.expert_interevention = False
                if done:
                    self.ready = not self.ready
                    break
            frame_idx += 1
            episode += 1
            print('frame', frame_idx, 'ep rew', ep_rew, 'steps taken in demo', step)

            # self.log['reward'].append([episode, ep_rew])
            self.log['reward'].append([episode, ep_rew, done, copy(self.expert_interevention)])
            pickle.dump(self.log, open('./data/hybrid_behavior_cloning/rewards3.pkl', 'wb'))

if __name__ == '__main__':

    rospy.init_node('behavior_clone_experiment')
    try:
        bc_experiment = HybridBehaviorCloneExperiment()
        bc_experiment.run()
    except Exception as e:
        print(e)
        traceback.print_exc()
    finally:
        os.kill(os.getpid(), signal.SIGKILL)
