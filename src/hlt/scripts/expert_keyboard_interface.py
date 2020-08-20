#!/usr/bin/env python

import rospy
import numpy as np
import tf 
from std_msgs.msg import Float32MultiArray
#from sensor_msgs.msg import Joy
import pygame


class KeyboardListener(object):

    def __init__(self):
        self.cmd_pub = rospy.Publisher('/vel_cmd', Float32MultiArray, queue_size=1)
        self.cmd = Float32MultiArray()
        self.cmd.data = [0., 0., 0., 0., 0., 0.]
        self.rate = rospy.Rate(10)
        self.alpha = 0.8
        self.velocity_scale = 0.5
    def listen(self):
        while not rospy.is_shutdown():
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if pygame.K_a: # left
                        self.cmd.data[0] = self.alpha * self.cmd.data[0] \
                            + (1-self.alpha) * self.velocity_scale
                    if pygame.K_d: # right
                        self.cmd.data[0] = self.alpha * self.cmd.data[0] \
                            + (1-self.alpha) * (-self.velocity_scale)
                    if pygame.K_w: # up
                        self.cmd.data[1] = self.alpha * self.cmd.data[1] \
                            + (1-self.alpha) * self.velocity_scale
                    if pygame.K_s: # down
                        self.cmd.data[1] = self.alpha * self.cmd.data[1] \
                            + (1-self.alpha) * (-self.velocity_scale)
                if event.type == pygame.KEYUP:
                    if pygame.K_a: # left
                        self.cmd.data[0] = self.alpha * self.cmd.data[0]
                    if pygame.K_d: # right
                        self.cmd.data[0] = self.alpha * self.cmd.data[0]
                    if pygame.K_w: # up
                        self.cmd.data[1] = self.alpha * self.cmd.data[1]
                    if pygame.K_s: # down
                        self.cmd.data[1] = self.alpha * self.cmd.data[1]
            self.cmd_pub.publish(self.cmd)
                
if __name__=='__main__':
    rospy.init_node('expert')
    expert_listener = KeyboardListener()
    expert_listener.listen()

