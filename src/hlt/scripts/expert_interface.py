#!/usr/bin/env python

import rospy
import numpy as np
import tf
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Joy
import actionlib
from franka_gripper.msg import GraspAction, GraspGoal, GraspEpsilon


class PSListener(object):

    def __init__(self):
        rospy.Subscriber('/joy', Joy, self.callback)
        self.cmd_pub = rospy.Publisher('/vel_cmd', Float32MultiArray, queue_size=1)
        self.cmd = Float32MultiArray()
        self.cmd.data = [0., 0., 0., 0., 0., 0.]
        self.raw_cmd = [0., 0., 0., 0., 0., 0.]
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

        # self.tf_listener = tf.TransformListener()

        self.rate = rospy.Rate(10)
    def callback(self, msg):
        if int(msg.buttons[0])==1:
            if self.open_grasp:
                self.grasp_cmd.width = 0.06
                self.client.send_goal(self.grasp_cmd)
                self.client.wait_for_result()
            if self.open_grasp is False:
                self.grasp_cmd.width = 0.1
                self.client.send_goal(self.grasp_cmd)
                self.client.wait_for_result()
            self.open_grasp = not self.open_grasp
        self.raw_cmd[0]=-self.velocity_scale * msg.axes[1]
        self.raw_cmd[1]=-self.velocity_scale * msg.axes[0]
        self.raw_cmd[2]=self.velocity_scale * msg.axes[4]

    def run(self):

        while not rospy.is_shutdown():
            xcmd = self.raw_cmd[0]
            ycmd = self.raw_cmd[1]
            zcmd = self.raw_cmd[2]
            self.cmd.data[0] = self.alpha * self.cmd.data[0] + (1-self.alpha) * xcmd
            self.cmd.data[1] = self.alpha * self.cmd.data[1] + (1-self.alpha) * ycmd
            self.cmd.data[2] = self.alpha * self.cmd.data[2] + (1-self.alpha) * zcmd

            self.cmd_pub.publish(self.cmd)
            self.rate.sleep()

if __name__=='__main__':
    rospy.init_node('expert')
    expert_listener = PSListener()
    expert_listener.run()
