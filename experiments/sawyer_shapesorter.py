#! /usr/bin/env python
"""
imports
"""
# general
import numpy as np
import time
from copy import copy, deepcopy
import warnings

# ros
import rospy
# import tf
from geometry_msgs.msg import PoseStamped, Pose, Quaternion
from std_msgs.msg import Header, Float64
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger, TriggerResponse
from tf_conversions import transformations

# sawyer
from intera_core_msgs.srv import SolvePositionIK, SolvePositionIKRequest
from intera_core_msgs.msg import JointCommand, EndpointState
from sawyer.msg import RelativeMove, Reward
from intera_interface import RobotParams, settings, Gripper
import intera_dataflow

# pykdl
from sawyer_pykdl import sawyer_kinematics

class sawyer_env(object):
    def __init__(self):
        '''
        controller
        '''
        # set up ik solver
        self.iksvc = rospy.ServiceProxy("ExternalTools/right/PositionKinematicsNode/IKService", SolvePositionIK)
        rospy.wait_for_service("ExternalTools/right/PositionKinematicsNode/IKService", 5.0)

        # set up py_kdl
        self.py_kdl = sawyer_kinematics("right")

        # set up sawyer (ijn place of limb class)
        self._joint_names = RobotParams().get_joint_names("right")
        self._joint_angle = dict()
        self._joint_velocity = dict()
        # self._joint_effort = dict()
        self._tip_states = None
        self.gripper = Gripper('right_gripper')

        # self._command_msg = JointCommand()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._pub_joint_cmd = rospy.Publisher('/robot/limb/right/joint_command',JointCommand,tcp_nodelay=True,queue_size=1)
        self._pub_joint_cmd_timeout = rospy.Publisher('/robot/limb/right/joint_command_timeout',Float64,latch=True,queue_size=1)
        self._pub_speed_ratio = rospy.Publisher('/robot/limb/right/set_speed_ratio', Float64, latch=True, queue_size=1)
        _joint_state_sub = rospy.Subscriber('robot/joint_states',JointState,self._on_joint_states,queue_size=1,tcp_nodelay=True)
        _tip_states_sub = rospy.Subscriber('/robot/limb/right/endpoint_state',EndpointState,self._on_tip_states,queue_size=1,tcp_nodelay=True)

        # set up controller
        self.alpha = 0.8 # [0,1]
        self.reset_joint_dict = dict()
        for name in self._joint_names:
            self.reset_joint_dict[name] = 0.0
        self.delta_theta = deepcopy(self.reset_joint_dict)

        # initalize home configuration
        home_orientation = (-0.464134980767, 0.885626012962, -0.0064449616952, 0.0142740065387)
        self.home_orientation_rpy = np.asarray(transformations.euler_from_quaternion(home_orientation))

        home_pose = [0.4321650390625, -0.1204091796875, -1.06283984375, 1.6995654296875, 1.5470302734375, 1.049638671875, -0.458474609375]
        self.home_joints = dict(zip(self._joint_names, home_pose))

        # wait for first response to joint subscriber
        self.got_joints = False
        while (self.got_joints == False):
            time.sleep(0.2)
        self.desired_theta_dot = deepcopy(self.reset_joint_dict)

        self.raw_command = RelativeMove()
        self.filtered_command = RelativeMove()
        self.reset_test = False
        print('controller setup complete')
        '''
        sawyer_env
        '''
        # set up ros
        self.move = rospy.Publisher('/test/relative_move',RelativeMove,queue_size=1)
        self.reward = rospy.Publisher('/test/reward',Reward,queue_size=1)
        rospy.Service('/test/done', Trigger, self.doneCallback)

        # set up tf
        self.got_pose = False
        while (self.got_pose == False):
            time.sleep(0.2)

        self.state = np.zeros(9)
        self.update_state()
        self.update_velocities()
        print('sawyer env setup complete')

    '''
    from limb class
    '''
    def _on_joint_states(self, msg):
        self.got_joints = True
        for idx, name in enumerate(msg.name):
            if name in self._joint_names:
                self._joint_angle[name] = msg.position[idx]
                self._joint_velocity[name] = msg.velocity[idx]
                # self._joint_effort[name] = msg.effort[idx]

    def joint_angles(self):
        return deepcopy(self._joint_angle)

    def _on_tip_states(self, msg):
        self.got_pose = True
        self._tip_states = deepcopy(msg)

    def set_joint_positions(self, positions):
        _command_msg = JointCommand()
        _command_msg.names = positions.keys()
        _command_msg.position = positions.values()
        _command_msg.mode = JointCommand.POSITION_MODE
        _command_msg.header.stamp = rospy.Time.now()
        self._pub_joint_cmd.publish(_command_msg)

    def move_to_joint_positions(self, positions, timeout=15.0,
                                threshold=settings.JOINT_ANGLE_TOLERANCE,
                                test=None):
        cmd = self.joint_angles()

        def genf(joint, angle):
            def joint_diff():
                return abs(angle - self._joint_angle[joint])
            return joint_diff

        diffs = [genf(j, a) for j, a in positions.items() if
                 j in self._joint_angle]
        fail_msg = "limb failed to reach commanded joint positions."
        def test_collision():
            # if self.has_collided():
            #     rospy.logerr(' '.join(["Collision detected.", fail_msg]))
            #     return True
            return False
        self.set_joint_positions(positions)
        intera_dataflow.wait_for(
            test=lambda: test_collision() or \
                         (callable(test) and test() == True) or \
                         (all(diff() < threshold for diff in diffs)),
            timeout=timeout,
            timeout_msg=fail_msg,
            rate=100,
            raise_on_error=False,
            body=lambda: self.set_joint_positions(positions)
            )

    def set_joint_velocities(self, velocities):
        _command_msg = JointCommand()
        _command_msg.names = velocities.keys()
        _command_msg.velocity = velocities.values()
        _command_msg.mode = JointCommand.VELOCITY_MODE
        _command_msg.header.stamp = rospy.Time.now()
        self._pub_joint_cmd.publish(_command_msg)

    # def set_joint_torques(self, torques):
    #     _command_msg = JointCommand()
    #     _command_msg.names = torques.keys()
    #     _command_msg.effort = torques.values()
    #     _command_msg.mode = JointCommand.TORQUE_MODE
    #     _command_msg.header.stamp = rospy.Time.now()
    #     self._pub_joint_cmd.publish(_command_msg)

    '''
    controller
    '''
    def ee_vel_to_joint_vel(self,_data,orientation):
        # calculate jacobian_pseudo_inverse
        data = self.clip_velocities(_data)

        jacobian_ps = self.py_kdl.jacobian_pseudo_inverse(joint_values=None)

        xdot = np.zeros(6)
        xdot[0] = data.dx
        xdot[1] = data.dy
        xdot[2] = data.dz
        xdot[3] = orientation[0]
        xdot[4] = orientation[1]
        xdot[5] = orientation[2]

        desired_theta_dot = np.matmul(jacobian_ps,xdot)

        for i in range(len(self._joint_names)):
            self.desired_theta_dot[self._joint_names[i]] = desired_theta_dot[0,i]

    def check_orientation(self):
        current_orientation = self._tip_states.pose.orientation
        quaternion = (current_orientation.x, current_orientation.y,
                      current_orientation.z, current_orientation.w)
        current_orientation_rpy = np.asarray(transformations.euler_from_quaternion(quaternion))
        correction = current_orientation_rpy-self.home_orientation_rpy
        correction[2] *= -1
        for i in range(3):
            if correction[i] < -np.pi:
                correction[i] += 2*np.pi
            elif correction[i] > np.pi:
                correction[i] -= 2*np.pi
        return correction

    def clip_velocities(self,action):
        max_x = 0.75
        min_x = 0.5 # 0.45
        max_y = 0.25 #0.3
        min_y = -0.2 #-0.25

        current_pose = deepcopy(self._tip_states)

        if (current_pose.pose.position.x > max_x):
            action.dx = np.clip(action.dx, -1,0)
        elif (current_pose.pose.position.x < min_x):
            action.dx = np.clip(action.dx, 0,1)
        if (current_pose.pose.position.y > max_y):
            action.dy = np.clip(action.dy, -1,0)
        elif(current_pose.pose.position.y < min_y):
            action.dy = np.clip(action.dy, 0,1)

        return action

    def move_up(self):
        # solve inverse kinematics
        ikreq = SolvePositionIKRequest()

        current_pose = self._tip_states.pose # get current state

        pose = current_pose
        pose.position.z += 0.06

        # create stamped pose with updated pose
        poseStamped = PoseStamped()
        poseStamped.header = Header(stamp=rospy.Time.now(), frame_id='base')
        poseStamped.pose = pose

        # Add desired pose for inverse kinematics
        ikreq.pose_stamp.append(poseStamped)
        ikreq.tip_names.append('right_hand') # for each pose in IK

        limb_joints = copy(self.home_joints)
        try:
            resp = self.iksvc(ikreq)

            # Check if result valid, and type of seed ultimately used to get solution
            if (resp.result_type[0] > 0):
                limb_joints = dict(zip(resp.joints[0].name, resp.joints[0].position))
                # rospy.loginfo("Response Message:\n%s", resp)
                self.desired_theta = limb_joints
            else:
                rospy.logerr("INVALID POSE - No Valid Joint Solution Found.")
                rospy.logerr("Result Error %d", resp.result_type[0])

        except (rospy.ServiceException, rospy.ROSException), e:
            rospy.logerr("Service call failed: %s" % (e,))

        return limb_joints

    def update_velocities(self):
        if self.reset_test == True:
            # move up
            vertical_joints = self.move_up()
            self._pub_speed_ratio.publish(Float64(0.1))
            self.move_to_joint_positions(vertical_joints)
            time.sleep(2)

            # move to home
            self._pub_speed_ratio.publish(Float64(0.2))
            self.move_to_joint_positions(copy(self.home_joints))
            time.sleep(2)

            # reset parameters
            # self.delta_theta = deepcopy(self.reset_joint_dict)
            self.desired_theta_dot = deepcopy(self.reset_joint_dict)  # copy(self.home_joints)
            self.raw_command = RelativeMove()
            self.filtered_command = RelativeMove()
            self.reset_test = False
            print("Reset Pose")

        raw_orientation_correction = self.check_orientation()
        self.filtered_command.dx = self.alpha*self.filtered_command.dx+(1-self.alpha)*self.raw_command.dx
        self.filtered_command.dy = self.alpha*self.filtered_command.dy+(1-self.alpha)*self.raw_command.dy
        self.filtered_command.dz = self.alpha*self.filtered_command.dz+(1-self.alpha)*self.raw_command.dz
        self.ee_vel_to_joint_vel(self.filtered_command,raw_orientation_correction*.25)
        self.set_joint_velocities(self.desired_theta_dot)

    '''
    sawery_env
    '''
    def update_state(self):
        target = np.zeros(9)
        target[0:3] = np.array([0.618718108914, 0.0361612427719, 0.143426261079])

        c_tip_state = copy(self._tip_states)

        ee = np.array([c_tip_state.pose.position.x, c_tip_state.pose.position.y, c_tip_state.pose.position.z,
                       c_tip_state.wrench.force.x,  c_tip_state.wrench.force.y,  c_tip_state.wrench.force.z,
                       c_tip_state.wrench.torque.x, c_tip_state.wrench.torque.y, c_tip_state.wrench.torque.z])

        self.state = ee-target

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

        # reward = -distance-l2norm
        reward = -distance
        # reward += -torque*1e-6
        # reward += -force*1e-6

        if (distance < thresh):
            done = True
            # reward += 10
            print('Reached goal!')

        return reward, done

    def reset(self):
        self.reset_test = True
        self.update_velocities()
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
            # theta = (np.pi/4)*np.clip(_a[2],-1,1)  # keep april tags in view
            action = 0.4*np.clip(_a, -1,1)
            # publish action input
            pose = RelativeMove()
            pose.dx = action[0]
            pose.dy = action[1]
            # pose.dtheta = theta
            self.move.publish(pose)

            self.raw_command = pose
            print('step function',self.raw_command)

            # get new state
            self.update_state()
            reward, done = self.reward_function()
        else:
            done = True
            reward, _ = self.reward_function()

        next_reward = Reward()
        next_reward.reward = reward
        self.reward.publish(next_reward)

        self.update_velocities()

        return self.state.copy(), reward, done

    def doneCallback(self,req):
        self.reset_test = True
        print('manual done called')
        return TriggerResponse(success=True,message="Done callback complete")
