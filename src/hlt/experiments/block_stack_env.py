#!/usr/bin/env python

import rospy
import numpy as np
import tf
import actionlib
from franka_gripper.msg import GraspAction, GraspGoal, GraspEpsilon
from hlt.msg import BlockStackState
from geometry_msgs.msg import WrenchStamped
from sensor_msgs.msg import Joy
from std_msgs.msg import Float32MultiArray
from copy import deepcopy

class BlockStackEnv(object):

    def __init__(self):

        self.tf_listener = tf.TransformListener()
        self.init_ee_pose = np.array([0.4, 0., 0.3])
        self.block_rel_pose = np.array([0.043, 0.205, -0.254])
        self.block_target_pose = np.array([0.02102451, -0.09910298, -0.03374503])
        self.state = np.zeros(7)
        self.ee_force = WrenchStamped()
        self.raw_ready_button = False

        rospy.Subscriber('/joy', Joy, self.callback)
        rospy.Subscriber('/franka_state_controller/F_ext', WrenchStamped, self.ee_force_callback)
        self.cmd_pub = rospy.Publisher('/vel_cmd', Float32MultiArray, queue_size=1)
        self.cmd = Float32MultiArray()
        self.cmd.data = [0., 0., 0., 0., 0., 0.]
        self.reset_cmd = [0., 0., 0.]
        self.grasp_cmd = GraspGoal()
        self.grasp_cmd.width = 0.1
        self.grasp_cmd.speed = 0.1
        self.grasp_cmd.force = 0.1
        self.grasp_cmd.epsilon.inner = 0.005
        self.grasp_cmd.epsilon.outer = 0.005
        self.open_grasp = True
        self.ee_pose = None
        self.velocity_scale = 0.2
        self.alpha = 0.4
        self.client = actionlib.SimpleActionClient('/franka_gripper/grasp', GraspAction)
        self.client.wait_for_server()

        # self.state_pub = rospy.Publisher('/state', BlockStackState, queue_size=1)
        self.tf_listener = tf.TransformListener()
        self.rate = rospy.Rate(10)

    def ee_force_callback(self, msg):
        self.ee_force = deepcopy(msg)

    def callback(self, msg):
        self.reset_cmd[0]= -msg.axes[1]
        self.reset_cmd[1]= -msg.axes[0]
        self.reset_cmd[2]=  msg.axes[4]
        self.raw_ready_button = int(msg.buttons[2])

    def __filter(self, cmd):
        for i in range(3): # we are only controlling xyz
            self.cmd.data[i] = self.alpha * self.cmd.data[i] \
                        + (1-self.alpha) * (self.velocity_scale * cmd[i])
    def __zero_cmd(self):
        for i in range(3):
            self.cmd.data[i] = 0.
    def reset(self):
        print "waiting for reset"
        ready = False
        self.grasp_cmd.width = 0.1
        self.client.send_goal(self.grasp_cmd)
        self.client.wait_for_result()
        self.open_grasp = True
        while not rospy.is_shutdown() and not ready:
            # ready = self.step(self.reset_cmd, resetting=True)
            self.__filter(self.reset_cmd)
            self.cmd_pub.publish(self.cmd)
            state, reward, done = self.get_state()
            if self.raw_ready_button == 1:
                ready = True
            self.rate.sleep()

        print "successful reset"
        return self.state.copy()

    def get_reward(self, state, action):
        reward = 0.
        if int(self.state[-1]):
            reward = -np.linalg.norm(state[:3]-self.block_rel_pose)
        else:
            reward = -1.25 * np.linalg.norm(state[:3]-self.block_target_pose)
        reward += -1e-6*np.linalg.norm([state[3],state[4],state[4]])
        reward += np.linalg.norm(action) * 1e-6
        return reward
    def get_state(self):
        done = False
        reward = 0.
        try:
            (trans, rot) = self.tf_listener.lookupTransform('/panda_link0', '/panda_EE', rospy.Time(0))
            self.ee_pose = trans
            rel_pose = trans - self.init_ee_pose
            if self.open_grasp:
                if np.linalg.norm(rel_pose - self.block_rel_pose) < 0.04 and np.linalg.norm(self.cmd.data) < 0.001:
                    self.grasp_cmd.width = 0.05
                    self.client.send_goal(self.grasp_cmd)
                    self.client.wait_for_result()
                    self.__zero_cmd()
                    self.open_grasp = not self.open_grasp
            if not self.open_grasp:
                if np.linalg.norm(rel_pose - self.block_target_pose) < 0.04 and np.linalg.norm(self.cmd.data) < 0.001:
                    self.grasp_cmd.width = 0.1
                    self.client.send_goal(self.grasp_cmd)
                    self.client.wait_for_result()
                    self.__zero_cmd()
                    self.open_grasp = not self.open_grasp
                    done = True




            self.state[0] = rel_pose[0]
            self.state[1] = rel_pose[1]
            self.state[2] = rel_pose[2]
            self.state[3] = self.ee_force.wrench.force.x
            self.state[4] = self.ee_force.wrench.force.y
            self.state[5] = self.ee_force.wrench.force.z
            self.state[6] = int(self.open_grasp)


            return self.state.copy(), 0, done

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            print 'bad data'
            return self.state.copy(), reward, done

    def step(self, action):
        print(self.ee_pose)
        if self.ee_pose[2] < 0.06:
            action[2] = np.clip(action[2], 0., 1.0)
        self.__filter(np.clip(action, -1., 1.))
        state, reward, done = self.get_state()
        reward = self.get_reward(self.state.copy(), action)

        if np.abs(self.ee_force.wrench.force.z)>25.:
            done = True
        self.cmd_pub.publish(self.cmd)
        return state.copy(), reward, done, {}
