#!/usr/bin/env python3
"""
XArm Hand Controller Node

Subscribes to hand tracking data and controls the xArm robot:
- Receives hand/pose and hand/gripper_cmd from MediaPipe tracker
- Transforms hand position to robot workspace with safety bounds
- Uses P-loop velocity control for smooth motion
- Publishes robot_action/pose and robot_action/gripper for data collection
- Handles reset and pause commands

This node uses mode 7 (Cartesian online trajectory planning) for smooth motion.
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
    # Software clip margin inside the hardware fence (meters)
    # Keeps targets 10mm away from the fence to avoid triggering error 35
    BOUNDARY_MARGIN = 0.01

    def __init__(self):
        super().__init__('xarm_hand_controller')

        # Declare parameters
        self.ip = self.declare_parameter('xarm_ip', '192.168.1.219').value
        self.control_rate = self.declare_parameter('control_rate', 30.0).value
        self.max_tcp_speed = self.declare_parameter('max_tcp_speed', 100.0).value  # mm/s
        self.latency_offset = self.declare_parameter('latency_offset', 0.0).value  # seconds

        # Workspace bounds (meters) - safety limits
        self.x_min = self.declare_parameter('workspace_x_min', 0.1).value
        self.x_max = self.declare_parameter('workspace_x_max', 0.7).value
        self.y_min = self.declare_parameter('workspace_y_min', -0.3).value
        self.y_max = self.declare_parameter('workspace_y_max', 0.3).value
        self.z_min = self.declare_parameter('workspace_z_min', 0.05).value
        self.z_max = self.declare_parameter('workspace_z_max', 0.4).value

        # Fixed orientation (pointing down): roll=180deg, pitch=0, yaw=0
        self.fixed_roll = self.declare_parameter('fixed_roll_deg', 180.0).value
        self.fixed_pitch = self.declare_parameter('fixed_pitch_deg', 0.0).value
        self.fixed_yaw = self.declare_parameter('fixed_yaw_deg', 0.0).value

        # Yaw limits (applied to hand-tracked yaw + fixed_yaw offset)
        self.yaw_min_deg = self.declare_parameter('yaw_min_deg', -120.0).value
        self.yaw_max_deg = self.declare_parameter('yaw_max_deg', 120.0).value

        # TCP offset: vertical offset from flange to gripper contact point (mm)
        self.tcp_offset_z = self.declare_parameter('tcp_offset_z', 0.0).value

        self.get_logger().info(f'Connecting to xArm at IP: {self.ip}')

        # Initialize XArm
        self.arm = XArmAPI(self.ip)
        self.setup_xarm()

        # Control state
        self.target_position = None  # Target XYZ in meters
        self.target_yaw = math.radians(self.fixed_yaw)  # Target yaw in radians
        self.gripper_cmd = 0.0  # 0 = open, 1 = closed
        self.is_paused = False
        self.is_resetting = False
        self.servo_active = False  # True when in mode 7
        self.hand_tracking_active = False

        # Latency delay buffer: stores (timestamp_sec, position, gripper, yaw) tuples
        # Cap at 300 entries (~10s at 30Hz) to prevent unbounded growth
        self._delay_buffer = collections.deque(maxlen=300)

        # Subscribers for hand tracking input
        # Subscribe to eef/pose_target (object-relative transformed position)
        self.hand_pose_sub = self.create_subscription(
            PoseStamped, 'eef/pose_target', self.hand_pose_callback, 10)
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

        # Set TCP offset so positions reference the gripper tip, not the flange
        self.arm.set_tcp_offset([0, 0, self.tcp_offset_z, 0, 0, 0])
        time.sleep(0.1)

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

    def switch_to_online_trajectory_mode(self):
        """Switch XArm to mode 7 (Cartesian online trajectory planning)."""
        if not self.servo_active:
            self.arm.set_mode(7)
            self.arm.set_state(0)
            time.sleep(0.1)
            self.servo_active = True
            self.get_logger().debug('Switched to online trajectory mode (mode 7)')

    def switch_to_position_mode(self):
        """Switch XArm to position control mode (mode 0)."""
        if self.servo_active:
            self.arm.set_mode(0)
            self.arm.set_state(0)
            time.sleep(0.1)
            self.servo_active = False
            self.get_logger().debug('Switched to position control mode')

    def recover_from_error(self):
        """Attempt to recover from xArm error state. Returns True if recovery was needed."""
        error_code = self.arm.error_code
        if error_code == 0:
            return False

        self.get_logger().warn(f'xArm error {error_code}, recovering...')
        self.arm.clean_error()
        self.arm.clean_warn()
        self.arm.motion_enable(enable=True)
        # Restore whichever mode was active
        if self.servo_active:
            self.arm.set_mode(7)
        else:
            self.arm.set_mode(0)
        self.arm.set_state(0)
        time.sleep(0.1)
        self.get_logger().info(f'Recovered from error {error_code}')
        return True

    def clip_to_workspace(self, position: np.ndarray) -> np.ndarray:
        """
        Clip position to workspace bounds.

        Args:
            position: XYZ position in meters

        Returns:
            Clipped position
        """
        m = self.BOUNDARY_MARGIN
        clipped = np.array([
            np.clip(position[0], self.x_min + m, self.x_max - m),
            np.clip(position[1], self.y_min + m, self.y_max - m),
            np.clip(position[2], self.z_min + m, self.z_max - m),
        ])
        return clipped

    def tracking_status_callback(self, msg: Bool):
        """Update hand tracking status."""
        was_active = self.hand_tracking_active
        self.hand_tracking_active = msg.data

        if was_active and not self.hand_tracking_active:
            # Hand tracking lost - stop robot and clear delay buffer (safety)
            self.target_position = None
            self.target_yaw = math.radians(self.fixed_yaw)
            self._delay_buffer.clear()

    def hand_pose_callback(self, msg: PoseStamped):
        """
        Receive hand pose and update target position and yaw.
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

            # Extract hand yaw from Z-rotation quaternion
            o = msg.pose.orientation
            hand_yaw = math.atan2(2.0 * (o.w * o.z + o.x * o.y),
                                  1.0 - 2.0 * (o.y * o.y + o.z * o.z))
            # Add fixed_yaw offset and clamp to limits
            yaw = math.radians(self.fixed_yaw) + hand_yaw
            yaw = max(math.radians(self.yaw_min_deg),
                      min(math.radians(self.yaw_max_deg), yaw))

            # Apply workspace bounds
            clipped = self.clip_to_workspace(position)

            if self.latency_offset > 0.0:
                # Buffer for delayed consumption
                now = self.get_clock().now().nanoseconds / 1e9
                self._delay_buffer.append((now, clipped, None, yaw))
            else:
                # Zero latency: apply immediately (original path)
                self.target_position = clipped
                self.target_yaw = yaw
                self.publish_action_pose(self.target_position, msg.header.stamp)

            # Ensure we're in online trajectory mode
            if not self.is_paused and self.hand_tracking_active:
                self.switch_to_online_trajectory_mode()

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
                self._delay_buffer.append((now, None, msg.data, None))
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

        # Orientation quaternion (fixed roll/pitch, tracked yaw)
        quat = R.from_euler('xyz', [
            math.radians(self.fixed_roll),
            math.radians(self.fixed_pitch),
            self.target_yaw
        ]).as_quat()

        msg.pose.orientation.x = float(quat[0])
        msg.pose.orientation.y = float(quat[1])
        msg.pose.orientation.z = float(quat[2])
        msg.pose.orientation.w = float(quat[3])

        self.action_pose_pub.publish(msg)

    def control_loop(self):
        """
        Main control loop - sends target position to robot via mode 7.
        Runs at control_rate Hz.

        When latency_offset > 0, drains buffered entries whose age >= latency_offset.
        """
        # Drain delay buffer if latency is active
        if self.latency_offset > 0.0 and len(self._delay_buffer) > 0:
            now = self.get_clock().now().nanoseconds / 1e9
            cutoff = now - self.latency_offset

            while len(self._delay_buffer) > 0 and self._delay_buffer[0][0] <= cutoff:
                ts, position, gripper, yaw = self._delay_buffer.popleft()

                if position is not None:
                    self.target_position = position
                if yaw is not None:
                    self.target_yaw = yaw

                if position is not None or yaw is not None:
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
            self.target_position is None or not self.servo_active):
            return

        # Recover from arm errors (e.g. safety boundary) before sending commands
        if self.arm.error_code != 0:
            self.recover_from_error()
            return  # Skip this tick, resume next cycle

        try:
            # Target position in mm
            target_x_mm = self.target_position[0] * 1000.0
            target_y_mm = self.target_position[1] * 1000.0
            target_z_mm = self.target_position[2] * 1000.0

            # Target orientation in degrees (roll/pitch fixed, yaw tracked)
            target_roll_deg = self.fixed_roll
            target_pitch_deg = self.fixed_pitch
            target_yaw_deg = math.degrees(self.target_yaw)

            # Send target pose — mode 7 handles trajectory planning
            self.arm.set_position(
                x=target_x_mm, y=target_y_mm, z=target_z_mm,
                roll=target_roll_deg, pitch=target_pitch_deg, yaw=target_yaw_deg,
                speed=self.max_tcp_speed, is_radian=False, wait=False
            )

        except Exception as e:
            self.get_logger().error(f'Error in control loop: {e}')

    def reset_callback(self, msg: Bool):
        """Reset robot to home position."""
        if msg.data:
            self.get_logger().info('Resetting XArm to home position...')
            self.is_resetting = True
            self.target_position = None

            try:
                # Clear any existing errors before reset
                self.arm.clean_error()
                self.arm.clean_warn()
                self.arm.motion_enable(enable=True)
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
        else:
            self.get_logger().info('XArm motion resumed')
            if self.target_position is not None and self.hand_tracking_active:
                self.switch_to_online_trajectory_mode()

    def destroy_node(self):
        """Clean shutdown."""
        self.get_logger().info('Disconnecting from XArm...')
        try:
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
