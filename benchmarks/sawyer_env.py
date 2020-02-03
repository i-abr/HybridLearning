#! /usr/bin/env python
"""
Set up sawyer environment

Used by h_sac.py
"""
# general
import numpy as np
import time
from copy import copy, deepcopy

# ros
import rospy
import tf
from geometry_msgs.msg import Pose2D
from std_srvs.srv import Trigger, TriggerResponse

# sawyer
from sawyer.msg import RelativeMove

class sawyer_env(object):
    def __init__(self):
        # set up ros
        self.move = rospy.Publisher('/puck/relative_move',RelativeMove,queue_size=1)
        self.reset_arm = rospy.ServiceProxy('/puck/reset', Trigger)
        rospy.wait_for_service('/puck/reset', 5.0)
        self.listener = tf.TransformListener()
        rospy.Service('/puck/done', Trigger, self.doneCallback)

        # set up flags
        self.reset_test = False

        # set up tf
        self.state = self.setup_transforms()

    def setup_transforms(self):
        target_transform = self.setup_transform_between_frames( 'target','top')
        ee_transform = self.setup_transform_between_frames('target','ee')
        try:
            self.state = np.array([ee_transform[0],ee_transform[1],target_transform[0],target_transform[1]])
        except:
            print("Check that all april tags are visible")

        return self.state


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
        lookups = ['top', 'block1','block2','block3','block4']
        try:
            ee_transform, _ = self.listener.lookupTransform( 'target','ee', rospy.Time(0))
            found_tf = True
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            found_tf = False
            pass
        if (found_tf == True):
            for lookup in lookups:
                try:
                    target_transform, _ = self.listener.lookupTransform( 'target',lookup, rospy.Time(0))
                    self.state = (1-0.8)*copy(self.state) + 0.8*np.array([ee_transform[0],ee_transform[1],target_transform[0],target_transform[1]])
                    print(lookup)
                    break
                except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                    pass

    def reset(self):
        self.reset_test = False
        resp = self.reset_arm()
        self.state = self.setup_transforms()
        return self.state.copy()

    def step(self, _a):
        if (self.reset_test == False):
            # theta = (np.pi/4)*np.clip(_a[2],-1,1)  # keep april tags in view
            action = 0.3*np.clip(_a, -1,1)
            # publish action input
            pose = RelativeMove()
            pose.dx = action[0]
            pose.dy = action[1]
            # pose.dtheta = theta
            self.move.publish(pose)

            # get new state
            self.get_transforms()
            reward, done = self.reward_function()
        else:
            done = True
            reward = -100
        return self.state.copy(), reward, done

    def reward_function(self):
        [dx_targetToArm, dy_targetToArm, dx_targetToBlock, dy_targetToBlock] = self.state.copy()

        arm_to_block = np.sqrt((dx_targetToArm-dx_targetToBlock)**2+
                        (dy_targetToArm-dy_targetToBlock)**2)

        block_to_target = np.sqrt(dx_targetToBlock**2+dy_targetToBlock**2)

        reward = 0
        done = False
        thresh = 0.08
        # if (arm_to_block > thresh):
        reward += -arm_to_block
        # if (arm_to_block < thresh):
        if (arm_to_block < thresh*2):
            reward += 1
            # reward += -block_to_target
        # if (self.wall == True):
        #     reward += -1

        if (arm_to_block < thresh):
            # if (block_to_target < thresh):
            done = True
            reward += 10
            print('Reached goal!')

        # rospy.loginfo("arm_to_block: %f, block_to_target: %f, reward: %f", arm_to_block, block_to_target, reward)
        rospy.loginfo("action reward: %f", reward)
        rospy.loginfo("block dist: %f", arm_to_block)
        rospy.loginfo("target dist: %f", block_to_target)

        return reward, done

    def doneCallback(self,req):
        self.reset_test = True
        print('manual done called')
        return TriggerResponse(success=True,
                               message="Done callback complete")
