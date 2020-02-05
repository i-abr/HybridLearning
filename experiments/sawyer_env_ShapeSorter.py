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
from geometry_msgs.msg import Pose2D
from std_srvs.srv import Trigger, TriggerResponse

# sawyer
from sawyer.msg import RelativeMove
from intera_core_msgs.msg import EndpointState,EndpointStates
from intera_interface import Gripper

class sawyer_env(object):
    def __init__(self):
        # set up ros
        self.move = rospy.Publisher('/puck/relative_move',RelativeMove,queue_size=1)
        self.reset_arm = rospy.ServiceProxy('/puck/reset', Trigger)
        rospy.wait_for_service('/puck/reset', 5.0)
        rospy.Service('/puck/done', Trigger, self.doneCallback)

        # set up flags
        self.reset_test = False

        # set up sawyer
        self.tip_name = "right_hand"
        self._tip_states = None
        _tip_states_sub = rospy.Subscriber('/robot/limb/right/tip_states',EndpointStates,self._on_tip_states,queue_size=1,tcp_nodelay=True)
        limb = "right"
        self.gripper = Gripper(limb + '_gripper')

        # set up tf
        self.got_pose = False
        while (self.got_pose == False):
            # print('waiting')
            time.sleep(0.2)

        self.state = np.zeros(9)
        self.update_state()

    def _on_tip_states(self, msg):
        self.got_pose = True
        self._tip_states = deepcopy(msg)

    def tip_state(self, tip_name):
        try:
            return deepcopy(self._tip_states.states[self._tip_states.names.index(tip_name)])
        except ValueError:
            return None

    def update_state(self):
        target = np.zeros(9)
        # target[0:3] = np.array([0.636175521396, -0.0225218216069, 0.14]) # (distance) [x,y,z], trapezoid
        # target[0:3] = np.array([0.608951905342, 0.0370819526536, 0.1436]) # diamond
        # target[0:3] = np.array([ 0.618630950525, 0.0360745993657, 0.142493648115])
        target[0:3] = np.array([0.618718108914, 0.0361612427719, 0.143426261079])

        ee = np.array([self.tip_state(self.tip_name).pose.position.x, self.tip_state(self.tip_name).pose.position.y,self.tip_state(self.tip_name).pose.position.z,
                       self.tip_state(self.tip_name).wrench.force.x,self.tip_state(self.tip_name).wrench.force.y,self.tip_state(self.tip_name).wrench.force.z,
                       self.tip_state(self.tip_name).wrench.torque.x,self.tip_state(self.tip_name).wrench.torque.y,self.tip_state(self.tip_name).wrench.torque.z])
        self.state = ee-target

    def reset(self):
        self.reset_test = False
        resp = self.reset_arm()
        o = raw_input("Enter '0' to open gripper, otherwise press enter to continue ")
        if (o == '0'):
            self.gripper.open()
            raw_input("Press enter to close gripper")
            self.gripper.close("Press enter to continue")
            raw_input()
        self.update_state()
        return self.state.copy()

    def step(self, _a):
        if (self.reset_test == False):
            # theta = (np.pi/4)*np.clip(_a[3],-1,1)  # keep april tags in view
            action = 0.2*np.clip(_a, -1,1)
            # publish action input
            pose = RelativeMove()
            pose.dx = action[0]
            pose.dy = action[1]
            pose.dz = action[2]
            # pose.dtheta = theta
            self.move.publish(pose)

            # get new state
            self.update_state()
            reward, done = self.reward_function()
        else:
            done = True
            reward, _ = self.reward_function()
        return self.state.copy(), reward, done

    def reward_function(self):
        [distance_dx, distance_dy, distance_dz,
         force_dx, force_dy, force_dz,
         torque_dx, torque_dy, torque_dz] = self.state.copy()

        l2norm = distance_dx**2+distance_dy**2+distance_dz**2
        distance = np.sqrt(distance_dx**2+distance_dy**2+distance_dz**2+1e-3)
        force = np.sqrt(force_dx**2+force_dy**2+force_dz**2)
        torque = np.sqrt(torque_dx**2+torque_dy**2+torque_dz**2)

        reward = 0
        done = False
        thresh = 0.032

        reward += -distance
        reward += -l2norm
        reward += -torque*1e-2
        reward += -force*1e-2

        if (distance < thresh):
            done = True
            # reward += 10
            print('Reached goal!')

        # rospy.loginfo("action reward: %f", reward)
        # rospy.loginfo("distance: %f", distance)
        # rospy.loginfo("torque: %f", torque)
        # rospy.loginfo("force: %f", force)

        return reward, done

    def doneCallback(self,req):
        self.reset_test = True
        print('manual done called')
        return TriggerResponse(success=True, message="Done callback complete")
