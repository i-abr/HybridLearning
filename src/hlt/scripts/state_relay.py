#!/usr/bin/env python

import rospy
import numpy as np
import tf
import actionlib
from franka_gripper.msg import GraspAction, GraspGoal, GraspEpsilon
from hlt.msg import BlockStackState
from geometry_msgs.msg import WrenchStamped
from sensor_msgs.msg import Joy
from copy import deepcopy

class StateRelay(object):

    def __init__(self):

        self.tf_listener = tf.TransformListener()
        self.init_ee_pose = np.array([0.4, 0., 0.3])
        self.block_rel_pose = np.array([0.043, 0.205, -0.254])
        self.block_target_pose = np.array([0.02102451, -0.09910298, -0.13374503])
        self.state = BlockStackState()
        self.ee_force = WrenchStamped()
        self.state.done = False

        self.rate = rospy.Rate(10)

        rospy.Subscriber('/joy', Joy, self.callback)
        rospy.Subscriber('/franka_state_controller/F_ext', WrenchStamped, self.ee_force_callback)
        # self.cmd_pub = rospy.Publisher('/vel_cmd', Float32MultiArray, queue_size=1)
        # self.cmd = Float32MultiArray()
        # self.cmd.data = [0., 0., 0., 0., 0., 0.]
        # self.raw_cmd = [0., 0., 0., 0., 0., 0.]
        self.grasp_cmd = GraspGoal()
        self.grasp_cmd.width = 0.1
        self.grasp_cmd.speed = 0.1
        self.grasp_cmd.force = 0.1
        self.grasp_cmd.epsilon.inner = 0.005
        self.grasp_cmd.epsilon.outer = 0.005
        self.open_grasp = True

        self.velocity_scale = 0.25
        self.alpha = 0.5
        self.client = actionlib.SimpleActionClient('/franka_gripper/grasp', GraspAction)
        self.client.wait_for_server()

        self.state_pub = rospy.Publisher('/state', BlockStackState, queue_size=1)
        # self.tf_listener = tf.TransformListener()

    def ee_force_callback(self, msg):
        self.state.ee_force = deepcopy(msg)

    def callback(self, msg):
        if int(msg.buttons[2]) == 1:
            self.state.done = False

    def run(self):

        while not rospy.is_shutdown():
            try:
                (trans, rot) = self.tf_listener.lookupTransform('/panda_link0', '/panda_EE', rospy.Time(0))
                rel_pose = trans - self.init_ee_pose
                if self.open_grasp:
                    if np.linalg.norm(rel_pose - self.block_rel_pose) < 0.01:
                        self.grasp_cmd.width = 0.06
                        self.client.send_goal(self.grasp_cmd)
                        self.client.wait_for_result()
                        self.open_grasp = not self.open_grasp
                if not self.open_grasp:
                    if np.linalg.norm(rel_pose - self.block_target_pose) < 0.02:
                        self.grasp_cmd.width = 0.1
                        self.client.send_goal(self.grasp_cmd)
                        self.client.wait_for_result()
                        self.open_grasp = not self.open_grasp
                        self.state.done = True

                reward = -np.linalg.norm(rel_pose - self.block_rel_pose) \
                            - 1.25 * np.linalg.norm(rel_pose - self.block_target_pose)

                self.state.position.x = rel_pose[0]
                self.state.position.y = rel_pose[1]
                self.state.position.z = rel_pose[2]
                self.state.open_grasp = self.open_grasp
                self.state.reward = reward
                if np.abs(self.state.ee_force.wrench.force.z)>12:
                    self.state.done = True
                    self.state.reward = -100.
                else:
                    self.state.done = False
                self.state_pub.publish(self.state)
                print self.state
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue

            self.rate.sleep()

if __name__=='__main__':
    rospy.init_node('expert')
    state_relay = StateRelay()
    state_relay.run()
