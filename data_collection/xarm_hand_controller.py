#!/usr/bin/env python3
"""
XArm Hand Controller Node

Subscribes to hand tracking data and controls the xArm robot:
- Receives hand/pose and hand/gripper_cmd from MediaPipe tracker
- Transforms hand position to robot workspace with safety bounds
- Uses P-loop velocity control for smooth motion
- Publishes robot_action/pose and robot_action/gripper for data collection
- Handles reset and pause commands

This node uses position-only control with fixed orientation.
"""

import time
import math
import collections
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32, Bool
from xarm.wrapper import XArmAPI
from scipy.spatial.transform import Rotation as R


class XArmHandController(Node):
    """
    ROS2 node that controls XArm based on hand tracking input.
    """
    def __init__(self):
        super().__init__('xarm_hand_controller')

        # Declare parameters
        self.ip = self.declare_parameter('xarm_ip', '192.168.1.219').value
        self.control_rate = self.declare_parameter('control_rate', 30.0).value
        self.p_gain_pos = self.declare_parameter('p_gain_position', 5.0).value
        self.max_linear_vel = self.declare_parameter('max_linear_velocity', 100.0).value  # mm/s
        self.latency_offset = self.declare_parameter('latency_offset', 0.0).value  # seconds

        # Workspace bounds (meters) - safety limits
        self.x_min = self.declare_parameter('workspace_x_min', 0.1).value
        self.x_max = self.declare_parameter('workspace_x_max', 0.5).value
        self.y_min = self.declare_parameter('workspace_y_min', -0.3).value
        self.y_max = self.declare_parameter('workspace_y_max', 0.3).value
        self.z_min = self.declare_parameter('workspace_z_min', 0.05).value
        self.z_max = self.declare_parameter('workspace_z_max', 0.4).value

        # Fixed orientation (pointing down): roll=180deg, pitch=0, yaw=0
        self.fixed_roll = self.declare_parameter('fixed_roll_deg', 180.0).value
        self.fixed_pitch = self.declare_parameter('fixed_pitch_deg', 0.0).value
        self.fixed_yaw = self.declare_parameter('fixed_yaw_deg', 0.0).value

        self.get_logger().info(f'Connecting to xArm at IP: {self.ip}')

        # Initialize XArm
        self.arm = XArmAPI(self.ip)
        self.setup_xarm()

        # Control state
        self.target_position = None  # Target XYZ in meters
        self.gripper_cmd = 0.0  # 0 = open, 1 = closed
        self.is_paused = False
        self.is_resetting = False
        self.velocity_control_active = False
        self.hand_tracking_active = False

        # Latency delay buffer: stores (timestamp_sec, position, gripper) tuples
        # Cap at 300 entries (~10s at 30Hz) to prevent unbounded growth
        self._delay_buffer = collections.deque(maxlen=300)

        # Subscribers for hand tracking input
        self.hand_pose_sub = self.create_subscription(
            PoseStamped, 'hand/pose', self.hand_pose_callback, 10)
        self.hand_gripper_sub = self.create_subscription(
            Float32, 'hand/gripper_cmd', self.gripper_callback, 10)
        self.tracking_status_sub = self.create_subscription(
            Bool, 'hand/tracking_active', self.tracking_status_callback, 10)

        # Subscribers for control signals
        self.reset_sub = self.create_subscription(
            Bool, '/reset_xarm', self.reset_callback, 10)
        self.pause_sub = self.create_subscription(
            Bool, '/pause_xarm', self.pause_callback, 10)

        # Publishers for data collection (actions)
        self.action_pose_pub = self.create_publisher(PoseStamped, 'robot_action/pose', 10)
        self.action_gripper_pub = self.create_publisher(Float32, 'robot_action/gripper', 10)

        # Control loop timer
        self.control_timer = self.create_timer(1.0 / self.control_rate, self.control_loop)

        # Initialize gripper
        self.initialize_gripper()

        self.get_logger().info('XArm Hand Controller initialized')
        self.get_logger().info(f'Workspace bounds: X[{self.x_min}, {self.x_max}], '
                              f'Y[{self.y_min}, {self.y_max}], Z[{self.z_min}, {self.z_max}]')
        if self.latency_offset > 0.0:
            self.get_logger().info(f'Latency delay: {self.latency_offset:.2f}s (buffer max 300 entries)')

    def setup_xarm(self):
        """Initialize XArm for control with safety limits."""
        self.arm.clean_error()
        self.arm.clean_warn()
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)  # Start in position mode
        self.arm.set_state(state=0)
        time.sleep(1)

        # Hardware-level safety boundaries (mm)
        # Format: [x_max, x_min, y_max, y_min, z_max, z_min]
        self.arm.set_reduced_tcp_boundary([
            self.x_max * 1000, self.x_min * 1000,
            self.y_max * 1000, self.y_min * 1000,
            self.z_max * 1000, self.z_min * 1000,
        ])
        self.arm.set_fense_mode(True)

        # Conservative acceleration limits for teleoperation
        self.arm.set_tcp_maxacc(5000)           # mm/s^2
        self.arm.set_joint_maxacc(10)           # rad/s^2
        self.arm.set_reduced_max_tcp_speed(200) # mm/s
        self.arm.set_reduced_max_joint_speed(60) # deg/s
        self.arm.set_reduced_mode(True)

        self.get_logger().info('XArm initialized with safety limits')

    def initialize_gripper(self):
        """Initialize gripper settings."""
        self.arm.set_gripper_mode(0)      # Location mode
        self.arm.set_gripper_enable(True)
        self.arm.set_gripper_speed(5000)
        self.arm.clean_gripper_error()
        self.get_logger().info('Gripper initialized')

    def switch_to_velocity_mode(self):
        """Switch XArm to velocity control mode."""
        if not self.velocity_control_active:
            self.arm.set_mode(5)  # Cartesian velocity control
            self.arm.set_state(0)
            time.sleep(0.1)
            self.velocity_control_active = True
            self.get_logger().debug('Switched to velocity control mode')

    def switch_to_position_mode(self):
        """Switch XArm to position control mode."""
        if self.velocity_control_active:
            # Stop any ongoing velocity
            self.arm.vc_set_cartesian_velocity([0, 0, 0, 0, 0, 0])
            time.sleep(0.1)
            self.arm.set_mode(0)
            self.arm.set_state(0)
            time.sleep(0.1)
            self.velocity_control_active = False
            self.get_logger().debug('Switched to position control mode')

    def clip_to_workspace(self, position: np.ndarray) -> np.ndarray:
        """
        Clip position to workspace bounds.

        Args:
            position: XYZ position in meters

        Returns:
            Clipped position
        """
        clipped = np.array([
            np.clip(position[0], self.x_min, self.x_max),
            np.clip(position[1], self.y_min, self.y_max),
            np.clip(position[2], self.z_min, self.z_max),
        ])
        return clipped

    def tracking_status_callback(self, msg: Bool):
        """Update hand tracking status."""
        was_active = self.hand_tracking_active
        self.hand_tracking_active = msg.data

        if was_active and not self.hand_tracking_active:
            # Hand tracking lost - stop robot and clear delay buffer (safety)
            self.target_position = None
            self._delay_buffer.clear()
            if self.velocity_control_active:
                try:
                    self.arm.vc_set_cartesian_velocity([0, 0, 0, 0, 0, 0])
                except:
                    pass

    def hand_pose_callback(self, msg: PoseStamped):
        """
        Receive hand pose and update target position.
        When latency_offset > 0, buffers the entry for delayed consumption.
        """
        if self.is_resetting or self.is_paused:
            return

        try:
            # Extract position (already in meters from hand tracker)
            position = np.array([
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z
            ])

            # Apply workspace bounds
            clipped = self.clip_to_workspace(position)

            if self.latency_offset > 0.0:
                # Buffer for delayed consumption
                now = self.get_clock().now().nanoseconds / 1e9
                self._delay_buffer.append((now, clipped, None))
            else:
                # Zero latency: apply immediately (original path)
                self.target_position = clipped
                self.publish_action_pose(self.target_position, msg.header.stamp)

            # Ensure we're in velocity mode
            if not self.is_paused and self.hand_tracking_active:
                self.switch_to_velocity_mode()

        except Exception as e:
            self.get_logger().error(f'Error in hand pose callback: {e}')

    def gripper_callback(self, msg: Float32):
        """Receive gripper command from hand tracker.
        When latency_offset > 0, buffers the entry for delayed consumption."""
        if self.is_resetting or self.is_paused:
            return

        try:
            if self.latency_offset > 0.0:
                # Buffer for delayed consumption
                now = self.get_clock().now().nanoseconds / 1e9
                self._delay_buffer.append((now, None, msg.data))
            else:
                # Zero latency: apply immediately (original path)
                self.gripper_cmd = msg.data

                # Convert normalized [0-1] to XArm gripper [850-0]
                # 0 = open (850), 1 = closed (0)
                grasp = 850 - 850 * self.gripper_cmd
                self.arm.set_gripper_position(int(grasp), wait=False)

                # Publish action for data collection
                gripper_msg = Float32()
                gripper_msg.data = self.gripper_cmd
                self.action_gripper_pub.publish(gripper_msg)

        except Exception as e:
            self.get_logger().error(f'Error in gripper callback: {e}')

    def publish_action_pose(self, position: np.ndarray, stamp):
        """Publish target pose as action for data collection."""
        msg = PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = "base_link"

        msg.pose.position.x = float(position[0])
        msg.pose.position.y = float(position[1])
        msg.pose.position.z = float(position[2])

        # Fixed orientation quaternion
        quat = R.from_euler('xyz', [
            math.radians(self.fixed_roll),
            math.radians(self.fixed_pitch),
            math.radians(self.fixed_yaw)
        ]).as_quat()

        msg.pose.orientation.x = float(quat[0])
        msg.pose.orientation.y = float(quat[1])
        msg.pose.orientation.z = float(quat[2])
        msg.pose.orientation.w = float(quat[3])

        self.action_pose_pub.publish(msg)

    def control_loop(self):
        """
        Main control loop - computes and sends velocity commands.
        Runs at control_rate Hz.

        When latency_offset > 0, drains buffered entries whose age >= latency_offset.
        """
        # Drain delay buffer if latency is active
        if self.latency_offset > 0.0 and len(self._delay_buffer) > 0:
            now = self.get_clock().now().nanoseconds / 1e9
            cutoff = now - self.latency_offset

            while len(self._delay_buffer) > 0 and self._delay_buffer[0][0] <= cutoff:
                ts, position, gripper = self._delay_buffer.popleft()

                if position is not None:
                    self.target_position = position
                    # Publish action pose at consumption time
                    stamp = self.get_clock().now().to_msg()
                    self.publish_action_pose(self.target_position, stamp)

                if gripper is not None:
                    self.gripper_cmd = gripper
                    grasp = 850 - 850 * self.gripper_cmd
                    self.arm.set_gripper_position(int(grasp), wait=False)

                    gripper_msg = Float32()
                    gripper_msg.data = self.gripper_cmd
                    self.action_gripper_pub.publish(gripper_msg)

        if (self.is_paused or self.is_resetting or
            self.target_position is None or not self.velocity_control_active):
            return

        try:
            # Get current position
            code, current_pos = self.arm.get_position(is_radian=True)
            if code != 0:
                self.get_logger().warn(f'Failed to get position, code: {code}')
                return

            # Current position in mm
            current_x_mm, current_y_mm, current_z_mm = current_pos[0:3]
            current_roll, current_pitch, current_yaw = current_pos[3:6]

            # Target position in mm
            target_x_mm = self.target_position[0] * 1000.0
            target_y_mm = self.target_position[1] * 1000.0
            target_z_mm = self.target_position[2] * 1000.0

            # Target orientation in radians
            target_roll = math.radians(self.fixed_roll)
            target_pitch = math.radians(self.fixed_pitch)
            target_yaw = math.radians(self.fixed_yaw)

            # Calculate position errors
            error_x = target_x_mm - current_x_mm
            error_y = target_y_mm - current_y_mm
            error_z = target_z_mm - current_z_mm

            # Calculate orientation errors
            error_roll = self.angle_diff(target_roll, current_roll)
            error_pitch = self.angle_diff(target_pitch, current_pitch)
            error_yaw = self.angle_diff(target_yaw, current_yaw)

            # P-control for velocity
            vel_x = self.p_gain_pos * error_x
            vel_y = self.p_gain_pos * error_y
            vel_z = self.p_gain_pos * error_z
            vel_roll = 1.0 * error_roll
            vel_pitch = 1.0 * error_pitch
            vel_yaw = 1.0 * error_yaw

            # Apply velocity limits
            vel_linear = np.array([vel_x, vel_y, vel_z])
            vel_linear_norm = np.linalg.norm(vel_linear)
            if vel_linear_norm > self.max_linear_vel:
                vel_linear = vel_linear * (self.max_linear_vel / vel_linear_norm)
                vel_x, vel_y, vel_z = vel_linear

            # Limit angular velocity
            max_angular_vel = 1.0
            vel_angular = np.array([vel_roll, vel_pitch, vel_yaw])
            vel_angular_norm = np.linalg.norm(vel_angular)
            if vel_angular_norm > max_angular_vel:
                vel_angular = vel_angular * (max_angular_vel / vel_angular_norm)
                vel_roll, vel_pitch, vel_yaw = vel_angular

            # Send velocity command
            velocities = [vel_x, vel_y, vel_z, vel_roll, vel_pitch, vel_yaw]
            self.arm.vc_set_cartesian_velocity(velocities)

        except Exception as e:
            self.get_logger().error(f'Error in control loop: {e}')

    def angle_diff(self, target: float, current: float) -> float:
        """Calculate shortest angular difference."""
        diff = target - current
        while diff > math.pi:
            diff -= 2 * math.pi
        while diff < -math.pi:
            diff += 2 * math.pi
        return diff

    def reset_callback(self, msg: Bool):
        """Reset robot to home position."""
        if msg.data:
            self.get_logger().info('Resetting XArm to home position...')
            self.is_resetting = True
            self.target_position = None

            try:
                self.switch_to_position_mode()

                # Home position (in mm and degrees)
                code = self.arm.set_position(
                    x=200.0, y=0.0, z=250.0,
                    roll=180.0, pitch=0.0, yaw=0.0,
                    speed=100, is_radian=False, wait=True
                )

                if code != 0:
                    self.get_logger().error(f'Reset position failed: {code}')
                else:
                    # Open gripper
                    self.arm.set_gripper_position(850, wait=True)
                    self.get_logger().info('XArm reset complete')

            except Exception as e:
                self.get_logger().error(f'Error during reset: {e}')
            finally:
                self.is_resetting = False

    def pause_callback(self, msg: Bool):
        """Pause or resume robot motion."""
        self.is_paused = msg.data

        if self.is_paused:
            self.get_logger().info('XArm motion paused')
            if self.velocity_control_active:
                try:
                    self.arm.vc_set_cartesian_velocity([0, 0, 0, 0, 0, 0])
                except:
                    pass
        else:
            self.get_logger().info('XArm motion resumed')
            if self.target_position is not None and self.hand_tracking_active:
                self.switch_to_velocity_mode()

    def destroy_node(self):
        """Clean shutdown."""
        self.get_logger().info('Disconnecting from XArm...')
        try:
            if self.velocity_control_active:
                self.arm.vc_set_cartesian_velocity([0, 0, 0, 0, 0, 0])
                time.sleep(0.1)
            self.arm.disconnect()
        except:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    node = XArmHandController()

    try:
        node.get_logger().info('Running XArm Hand Controller...')
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
