#!/usr/bin/env python3
"""
xArm Simulator Node

Simulates xArm robot behavior for testing hand tracking control without hardware.
Subscribes to the same topics as the real controller and publishes simulated robot state.

Features:
- Realistic motion dynamics (velocity-limited movement)
- Workspace bounds enforcement
- Reset to home position
- Pause functionality
- Publishes same topics as real robot for seamless transition
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32, Bool

from utils.safety_limits import SafetyLimits
from utils.object_relative_transform import ObjectRelativeTransform, ObjectFrame


class XArmSimulator(Node):
    """
    Simulates xArm robot for testing control code without hardware.
    """

    # Home position (meters)
    HOME_POSITION = np.array([0.3, 0.0, 0.2])
    HOME_GRIPPER = 0.0  # Open

    # Fixed end-effector orientation (pointing down)
    # Quaternion: w, x, y, z for 180 degree rotation around X axis
    FIXED_ORIENTATION = np.array([0.0, 1.0, 0.0, 0.0])  # wxyz

    def __init__(self):
        super().__init__('xarm_simulator')

        # Parameters
        self.declare_parameter('max_velocity', 0.15)  # m/s
        self.declare_parameter('max_gripper_velocity', 1.0)  # units/s
        self.declare_parameter('control_rate', 30.0)  # Hz

        self._max_velocity = self.get_parameter('max_velocity').value
        self._max_gripper_velocity = self.get_parameter('max_gripper_velocity').value
        self._control_rate = self.get_parameter('control_rate').value
        self._dt = 1.0 / self._control_rate

        # Object-relative frame parameters
        self.declare_parameter('human_object_x', 0.30)
        self.declare_parameter('human_object_y', 0.00)
        self.declare_parameter('human_object_z', 0.10)
        self.declare_parameter('robot_object_x', 0.35)
        self.declare_parameter('robot_object_y', -0.15)
        self.declare_parameter('robot_object_z', 0.10)
        self.declare_parameter('use_object_relative', True)

        self._use_object_relative = self.get_parameter('use_object_relative').value
        self._object_transform = ObjectRelativeTransform(
            human_object_frame=ObjectFrame.from_position(
                self.get_parameter('human_object_x').value,
                self.get_parameter('human_object_y').value,
                self.get_parameter('human_object_z').value,
            ),
            robot_object_frame=ObjectFrame.from_position(
                self.get_parameter('robot_object_x').value,
                self.get_parameter('robot_object_y').value,
                self.get_parameter('robot_object_z').value,
            ),
        )

        # Safety limits
        self._safety = SafetyLimits()

        # Current simulated state
        self._current_position = self.HOME_POSITION.copy()
        self._current_gripper = self.HOME_GRIPPER

        # Target from hand tracking
        self._target_position = self.HOME_POSITION.copy()
        self._target_gripper = self.HOME_GRIPPER

        # Control state
        self._is_paused = False
        self._tracking_active = False

        # Subscribers
        self.hand_pose_sub = self.create_subscription(
            PoseStamped, 'hand/pose', self.hand_pose_callback, 10)
        self.gripper_cmd_sub = self.create_subscription(
            Float32, 'hand/gripper_cmd', self.gripper_cmd_callback, 10)
        self.tracking_active_sub = self.create_subscription(
            Bool, 'hand/tracking_active', self.tracking_active_callback, 10)
        self.reset_sub = self.create_subscription(
            Bool, '/reset_xarm', self.reset_callback, 10)
        self.pause_sub = self.create_subscription(
            Bool, '/pause_xarm', self.pause_callback, 10)

        # Publishers - same as real robot
        self.robot_obs_pose_pub = self.create_publisher(PoseStamped, 'robot_obs/pose', 10)
        self.robot_obs_gripper_pub = self.create_publisher(Float32, 'robot_obs/gripper', 10)
        self.robot_action_pose_pub = self.create_publisher(PoseStamped, 'robot_action/pose', 10)
        self.robot_action_gripper_pub = self.create_publisher(Float32, 'robot_action/gripper', 10)

        # Control loop timer
        self.control_timer = self.create_timer(self._dt, self.control_loop)

        self.get_logger().info('xArm Simulator initialized')
        self.get_logger().info(f'Max velocity: {self._max_velocity} m/s')
        self.get_logger().info(f'Workspace bounds: {self._safety.get_bounds_str()}')
        self.get_logger().info(f'Home position: {self.HOME_POSITION}')
        if self._use_object_relative:
            self.get_logger().info(
                f'Object-relative mode: human={self._object_transform.human_object_frame.position}, '
                f'robot={self._object_transform.robot_object_frame.position}'
            )
        else:
            self.get_logger().info('Object-relative mode: DISABLED (direct mapping)')

    def hand_pose_callback(self, msg: PoseStamped):
        """Update target position from hand tracking."""
        if self._is_paused:
            return

        # Extract raw hand position
        raw_hand = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])

        # Apply object-relative transform if enabled
        if self._use_object_relative:
            raw_target = self._object_transform.transform(raw_hand)
        else:
            raw_target = raw_hand

        # Apply safety limits
        self._target_position = self._safety.clamp_position(raw_target)

    def gripper_cmd_callback(self, msg: Float32):
        """Update target gripper from hand tracking."""
        if self._is_paused:
            return

        # Clamp gripper to [0, 1]
        self._target_gripper = np.clip(msg.data, 0.0, 1.0)

    def tracking_active_callback(self, msg: Bool):
        """Track whether hand is being detected."""
        self._tracking_active = msg.data

    def reset_callback(self, msg: Bool):
        """Reset robot to home position."""
        if msg.data:
            self.get_logger().info('Resetting to home position')
            self._target_position = self.HOME_POSITION.copy()
            self._target_gripper = self.HOME_GRIPPER
            self._is_paused = False

    def pause_callback(self, msg: Bool):
        """Pause/unpause robot movement."""
        self._is_paused = msg.data
        if self._is_paused:
            self.get_logger().info('Simulator PAUSED')
            # When paused, target stays at current position
            self._target_position = self._current_position.copy()
        else:
            self.get_logger().info('Simulator UNPAUSED')

    def control_loop(self):
        """
        Main control loop - move simulated robot toward target with velocity limits.
        """
        # Calculate position error
        pos_error = self._target_position - self._current_position
        pos_error_norm = np.linalg.norm(pos_error)

        # Apply velocity limit
        max_step = self._max_velocity * self._dt

        if pos_error_norm > max_step:
            # Move at max velocity toward target
            direction = pos_error / pos_error_norm
            self._current_position += direction * max_step
        else:
            # Can reach target this step
            self._current_position = self._target_position.copy()

        # Gripper movement with velocity limit
        gripper_error = self._target_gripper - self._current_gripper
        max_gripper_step = self._max_gripper_velocity * self._dt

        if abs(gripper_error) > max_gripper_step:
            self._current_gripper += np.sign(gripper_error) * max_gripper_step
        else:
            self._current_gripper = self._target_gripper

        # Ensure position is within bounds (safety check)
        self._current_position = self._safety.clamp_position(self._current_position)

        # Publish robot observation (current state)
        self.publish_robot_state()

        # Publish robot action (target command)
        self.publish_robot_action()

    def publish_robot_state(self):
        """Publish current simulated robot state."""
        now = self.get_clock().now().to_msg()

        # Pose
        pose_msg = PoseStamped()
        pose_msg.header.stamp = now
        pose_msg.header.frame_id = "base_link"
        pose_msg.pose.position.x = float(self._current_position[0])
        pose_msg.pose.position.y = float(self._current_position[1])
        pose_msg.pose.position.z = float(self._current_position[2])
        pose_msg.pose.orientation.w = float(self.FIXED_ORIENTATION[0])
        pose_msg.pose.orientation.x = float(self.FIXED_ORIENTATION[1])
        pose_msg.pose.orientation.y = float(self.FIXED_ORIENTATION[2])
        pose_msg.pose.orientation.z = float(self.FIXED_ORIENTATION[3])
        self.robot_obs_pose_pub.publish(pose_msg)

        # Gripper
        gripper_msg = Float32()
        gripper_msg.data = float(self._current_gripper)
        self.robot_obs_gripper_pub.publish(gripper_msg)

    def publish_robot_action(self):
        """Publish target command (action)."""
        now = self.get_clock().now().to_msg()

        # Pose
        pose_msg = PoseStamped()
        pose_msg.header.stamp = now
        pose_msg.header.frame_id = "base_link"
        pose_msg.pose.position.x = float(self._target_position[0])
        pose_msg.pose.position.y = float(self._target_position[1])
        pose_msg.pose.position.z = float(self._target_position[2])
        pose_msg.pose.orientation.w = float(self.FIXED_ORIENTATION[0])
        pose_msg.pose.orientation.x = float(self.FIXED_ORIENTATION[1])
        pose_msg.pose.orientation.y = float(self.FIXED_ORIENTATION[2])
        pose_msg.pose.orientation.z = float(self.FIXED_ORIENTATION[3])
        self.robot_action_pose_pub.publish(pose_msg)

        # Gripper
        gripper_msg = Float32()
        gripper_msg.data = float(self._target_gripper)
        self.robot_action_gripper_pub.publish(gripper_msg)


def main(args=None):
    rclpy.init(args=args)

    node = XArmSimulator()

    try:
        node.get_logger().info('Running xArm Simulator...')
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
