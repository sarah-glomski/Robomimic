#!/usr/bin/env python3
"""
XArm Velocity Controller Node

This node handles commanding the XArm using velocity control:
- Velocity commands based on position error (P-controller)
- Gripper commands
- Reset commands
- Pause/resume commands

It does NOT publish any state information.
"""

import time
import math
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32, Bool
from xarm.wrapper import XArmAPI
from scipy.spatial.transform import Rotation as R


class XArmVelocityController(Node):
    """
    ROS node that controls XArm using velocity control with P-loop.
    """
    def __init__(self):
        super().__init__('xarm_velocity_controller')
        
        # Declare parameters
        self.ip = self.declare_parameter('xarm_ip', '192.168.1.213').value
        self.control_rate = self.declare_parameter('control_rate', 30.0).value  # Hz
        self.p_gain_pos = self.declare_parameter('p_gain_position', 5.0).value  # Position P-gain
        self.p_gain_ori = self.declare_parameter('p_gain_orientation', 1.0).value  # Orientation P-gain
        self.max_linear_vel = self.declare_parameter('max_linear_velocity', 100.0).value  # mm/s
        self.max_angular_vel = self.declare_parameter('max_angular_velocity', 1.0).value  # rad/s
        self.position_tolerance = self.declare_parameter('position_tolerance', 5.0).value  # mm
        
        self.get_logger().info(f'Connecting to xArm at IP: {self.ip}')

        # Initialize XArm
        self.arm = XArmAPI(self.ip)
        self.setup_xarm()

        # Control state
        self.target_position = None  # Target pose in mm and radians
        self.gripper = 0.0
        self.is_paused = False
        self.is_resetting = False
        self.velocity_control_active = False

        # Subscribers for commands
        self.position_sub = self.create_subscription(
            PoseStamped, 'xarm_position', self.position_callback, 1)
        self.gripper_cmd_sub = self.create_subscription(
            Float32, 'gripper_position', self.gripper_callback, 1)
        self.reset_sub = self.create_subscription(
            Bool, '/reset_xarm', self.reset_callback, 10)
        self.pause_sub = self.create_subscription(
            Bool, '/pause_xarm', self.pause_callback, 10)

        # Control loop timer
        self.control_timer = self.create_timer(
            1.0 / self.control_rate, self.control_loop)

        # Initialize gripper and reset position
        self.initialize_robot()

        self.get_logger().info('XArm Velocity Controller initialized successfully')

    def setup_xarm(self):
        """
        Initialize the XArm for velocity control.
        """
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)  # Start in position mode
        self.arm.set_state(state=0)  # Ready state
        time.sleep(1)
        self.get_logger().info('XArm initialized')

    def initialize_robot(self):
        """
        Initialize gripper and reset robot to home position.
        """
        # Setup gripper
        self.arm.set_gripper_mode(0)      # location mode
        self.arm.set_gripper_enable(True) # power the driver
        self.arm.set_gripper_speed(5000)  # speed (1-5000)
        self.arm.clean_gripper_error()    # clear residual errors
        
        # Open gripper and reset position
        self.gripper_callback(Float32(data=0.0))
        self.reset_callback(Bool(data=True))

    def switch_to_velocity_mode(self):
        """
        Switch XArm to velocity control mode.
        """
        if not self.velocity_control_active:
            self.arm.set_mode(5)  # Cartesian velocity control mode
            self.arm.set_state(0)
            time.sleep(0.1)
            self.velocity_control_active = True
            self.get_logger().info('Switched to velocity control mode')

    def switch_to_position_mode(self):
        """
        Switch XArm to position control mode.
        """
        if self.velocity_control_active:
            # Stop any ongoing velocity
            self.arm.vc_set_cartesian_velocity([0, 0, 0, 0, 0, 0])
            time.sleep(0.1)
            
            self.arm.set_mode(0)  # Position control mode
            self.arm.set_state(0)
            time.sleep(0.1)
            self.velocity_control_active = False
            self.get_logger().info('Switched to position control mode')

    def position_callback(self, pose_msg: PoseStamped):
        """
        Callback for receiving commanded positions.
        """
        if self.is_resetting:
            return

        try:
            # Extract position (convert from m to mm)
            x_mm = pose_msg.pose.position.x * 1000.0
            y_mm = pose_msg.pose.position.y * 1000.0
            z_mm = pose_msg.pose.position.z * 1000.0

            # Extract quaternion orientation
            qw = pose_msg.pose.orientation.w
            qx = pose_msg.pose.orientation.x
            qy = pose_msg.pose.orientation.y
            qz = pose_msg.pose.orientation.z

            # Convert quaternion to Euler angles (roll, pitch, yaw) in radians
            ar = R.from_quat([qx, qy, qz, qw])
            roll_rad, pitch_rad, yaw_rad = ar.as_euler('xyz', degrees=False)

            # Update target position
            self.target_position = {
                'x': x_mm, 'y': y_mm, 'z': z_mm,
                'roll': roll_rad, 'pitch': pitch_rad, 'yaw': yaw_rad
            }

            # Make sure we're in velocity mode
            if not self.is_paused:
                self.switch_to_velocity_mode()

        except Exception as e:
            self.get_logger().error(f'Error in position callback: {e}')

    def control_loop(self):
        """
        Main control loop that computes and sends velocity commands.
        """
        if (self.is_paused or self.is_resetting or 
            self.target_position is None or not self.velocity_control_active):
            return

        try:
            # Get current position
            code, current_pos = self.arm.get_position(is_radian=True)
            if code != 0:
                self.get_logger().warn(f'Failed to get position, code: {code}')
                return

            # Extract current position and orientation
            current_x, current_y, current_z = current_pos[0:3]  # mm
            current_roll, current_pitch, current_yaw = current_pos[3:6]  # radians

            # Calculate position errors
            error_x = self.target_position['x'] - current_x
            error_y = self.target_position['y'] - current_y
            error_z = self.target_position['z'] - current_z

            # Calculate orientation errors (shortest path)
            error_roll = self.angle_diff(self.target_position['roll'], current_roll)
            error_pitch = self.angle_diff(self.target_position['pitch'], current_pitch)
            error_yaw = self.angle_diff(self.target_position['yaw'], current_yaw)

            # Apply P-control to compute velocities
            vel_x = self.p_gain_pos * error_x
            vel_y = self.p_gain_pos * error_y
            vel_z = self.p_gain_pos * error_z
            
            vel_roll = self.p_gain_ori * error_roll
            vel_pitch = self.p_gain_ori * error_pitch
            vel_yaw = self.p_gain_ori * error_yaw

            # Apply velocity limits
            vel_linear = np.array([vel_x, vel_y, vel_z])
            vel_linear_norm = np.linalg.norm(vel_linear)
            if vel_linear_norm > self.max_linear_vel:
                vel_linear = vel_linear * (self.max_linear_vel / vel_linear_norm)
                vel_x, vel_y, vel_z = vel_linear

            vel_angular = np.array([vel_roll, vel_pitch, vel_yaw])
            vel_angular_norm = np.linalg.norm(vel_angular)
            if vel_angular_norm > self.max_angular_vel:
                vel_angular = vel_angular * (self.max_angular_vel / vel_angular_norm)
                vel_roll, vel_pitch, vel_yaw = vel_angular

            # # Check if we're close enough to target
            # position_error = np.linalg.norm([error_x, error_y, error_z])
            # if position_error < self.position_tolerance:
            #     # Close to target, reduce velocities
            #     vel_x *= 0.5
            #     vel_y *= 0.5
            #     vel_z *= 0.5
            #     vel_roll *= 0.5
            #     vel_pitch *= 0.5
            #     vel_yaw *= 0.5

            # Send velocity command
            velocities = [vel_x, vel_y, vel_z, vel_roll, vel_pitch, vel_yaw]
            code = self.arm.vc_set_cartesian_velocity(velocities)
            
            if code != 0:
                self.get_logger().warn(f'Velocity command failed with code: {code}')

        except Exception as e:
            self.get_logger().error(f'Error in control loop: {e}')

    def angle_diff(self, target, current):
        """
        Calculate the shortest angular difference between two angles.
        """
        diff = target - current
        while diff > math.pi:
            diff -= 2 * math.pi
        while diff < -math.pi:
            diff += 2 * math.pi
        return diff

    def gripper_callback(self, gripper_msg: Float32):
        """
        Callback for receiving gripper position commands.
        """
        if self.is_paused or self.is_resetting:
            return
            
        try:
            # Set gripper position
            self.gripper = gripper_msg.data
            
            # Convert normalized value [0-1] to XArm gripper value [0-850]
            grasp = 850 - 850 * self.gripper
            code = self.arm.set_gripper_position(grasp, wait=False)
            
            if code != 0:
                self.get_logger().warn(f'Gripper command failed with code: {code}')
                
        except Exception as e:
            self.get_logger().error(f'Error in gripper callback: {e}')

    def reset_callback(self, msg: Bool):
        """
        Reset the XArm to a predefined position.
        """
        if msg.data:
            self.get_logger().info('Resetting XArm position...')
            self.is_resetting = True
            self.target_position = None
            
            try:
                # Switch to position mode for reset
                self.switch_to_position_mode()
                
                # Reset position (in mm and degrees)
                code = self.arm.set_position(
                    x=166.9, y=1.9, z=230.8, 
                    roll=179.1, pitch=0, yaw=1.2, 
                    speed=100, is_radian=False, wait=True
                )
                
                if code != 0:
                    self.get_logger().error(f'Reset position failed with code: {code}')
                else:
                    # Reset gripper
                    code = self.arm.set_gripper_position(850, wait=True)  # Fully open
                    if code != 0:
                        self.get_logger().error(f'Reset gripper failed with code: {code}')
                    else:
                        self.get_logger().info('XArm reset complete')
                
            except Exception as e:
                self.get_logger().error(f'Error during reset: {e}')
            finally:
                self.is_resetting = False

    def pause_callback(self, msg: Bool):
        """
        Pause or resume XArm motion.
        """
        self.is_paused = msg.data
        if self.is_paused:
            self.get_logger().info('XArm motion paused')
            # Stop velocity immediately
            if self.velocity_control_active:
                try:
                    self.arm.vc_set_cartesian_velocity([0, 0, 0, 0, 0, 0])
                except:
                    pass
        else:
            self.get_logger().info('XArm motion resumed')
            # Resume velocity control if we have a target
            if self.target_position is not None:
                self.switch_to_velocity_mode()

    def destroy_node(self):
        """
        Cleanly disconnect from XArm before shutting down.
        """
        self.get_logger().info('Disconnecting from XArm...')
        try:
            # Stop any motion
            if self.velocity_control_active:
                self.arm.vc_set_cartesian_velocity([0, 0, 0, 0, 0, 0])
                time.sleep(0.1)
            self.arm.disconnect()
        except:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    # Create velocity controller node
    node = XArmVelocityController()
    
    try:
        node.get_logger().info('Running XArm Velocity Controller...')
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()