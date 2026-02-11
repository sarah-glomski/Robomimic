#!/usr/bin/env python3
"""
XArm Hand Controller Node

Subscribes to hand tracking data and controls the xArm robot:
- Receives hand/pose and hand/gripper_cmd from MediaPipe tracker
- Transforms hand position to robot workspace with safety bounds
- Uses P-loop velocity control for smooth motion
- Publishes robot_action/pose and robot_action/gripper for data collection
- Handles reset and pause commands

Safety Features:
- Workspace bounds enforcement
- Velocity limits (configurable max linear/angular velocity)
- Position delta limits (max movement per control step)
- Soft-start mode (reduced speed for initial testing)

This node uses position-only control with fixed orientation.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import math
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32, Bool
from xarm.wrapper import XArmAPI
from scipy.spatial.transform import Rotation as R

from utils.safety_limits import SafetyLimits, WorkspaceBounds, VelocityLimits, DeltaLimits
from utils.object_relative_transform import ObjectRelativeTransform, ObjectFrame


class XArmHandController(Node):
    """
    ROS2 node that controls XArm based on hand tracking input.
    """
    def __init__(self):
        super().__init__('xarm_hand_controller')

        # Declare parameters
        self.ip = self.declare_parameter('xarm_ip', '192.168.1.153').value
        self.control_rate = self.declare_parameter('control_rate', 30.0).value
        self.p_gain_pos = self.declare_parameter('p_gain_position', 5.0).value
        self.max_linear_vel = self.declare_parameter('max_linear_velocity', 250.0).value  # mm/s

        # Soft start mode - reduce speed for initial testing (0.0-1.0)
        self.soft_start_factor = self.declare_parameter('soft_start_factor', 1.0).value

        # Neutral orientation (pointing down): roll=180deg, pitch=0, yaw=0
        self.neutral_roll = self.declare_parameter('neutral_roll_deg', 180.0).value
        self.neutral_pitch = self.declare_parameter('neutral_pitch_deg', 0.0).value
        self.neutral_yaw = self.declare_parameter('neutral_yaw_deg', 0.0).value

        # Orientation bounds (degrees from neutral)
        self.roll_limit = self.declare_parameter('roll_limit_deg', 30.0).value
        self.pitch_limit = self.declare_parameter('pitch_limit_deg', 45.0).value
        self.yaw_limit = self.declare_parameter('yaw_limit_deg', 120.0).value

        # Max orientation change per control step (degrees). At 30Hz, 3°/step = 90°/s.
        self.max_orientation_delta = self.declare_parameter('max_orientation_delta_deg', 3.0).value

        # Hardware-level joint safety limits (enforced by controller firmware)
        self.reduced_max_joint_speed = self.declare_parameter('reduced_max_joint_speed_deg', 180.0).value  # deg/s
        self.reduced_max_tcp_speed = self.declare_parameter('reduced_max_tcp_speed_mm', 250.0).value  # mm/s
        self.joint_max_acc = self.declare_parameter('joint_max_acc_deg', 500.0).value  # deg/s^2
        self.tcp_max_acc = self.declare_parameter('tcp_max_acc_mm', 2000.0).value  # mm/s^2

        # Collision sensitivity: 0=disabled, 1-5=increasing sensitivity.
        # During teleoperation, high sensitivity causes false collision triggers
        # from fast wrist motion, which halts the arm mid-trajectory. The next
        # command then arrives at a position far from where the arm stopped,
        # causing S23 (large motor position deviation).
        self.collision_sensitivity = self.declare_parameter('collision_sensitivity', 0).value

        # Object-relative frame parameters
        self.declare_parameter('human_object_x', 0.25)
        self.declare_parameter('human_object_y', -0.25)
        self.declare_parameter('human_object_z', 0.10)
        self.declare_parameter('robot_object_x', 0.25)
        self.declare_parameter('robot_object_y', 0.00)
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

        # Initialize safety limits
        self._safety = SafetyLimits(
            workspace=WorkspaceBounds(
                x_min=0.20, x_max=0.40,
                y_min=-0.15, y_max=0.15,
                z_min=0.10, z_max=0.30,
            ),
            velocity=VelocityLimits(
                max_linear_velocity=self.max_linear_vel / 1000.0,  # Convert mm/s to m/s
            ),
            delta=DeltaLimits(
                max_position_delta=0.015,  # 15mm max per control step
                max_gripper_delta=0.05,
            ),
            soft_start_factor=self.soft_start_factor
        )

        self.get_logger().info(f'Connecting to xArm at IP: {self.ip}')

        # Initialize XArm
        self.arm = XArmAPI(self.ip)
        self.setup_xarm()

        # Control state
        self.target_position = None  # Target XYZ in meters
        self.target_orientation = None  # Target roll, pitch, yaw in degrees
        self.gripper_cmd = 0.0  # 0 = open, 1 = closed
        self.gripper_enabled = False  # Set by initialize_gripper
        self.is_paused = False
        self.is_resetting = False
        self.position_control_active = False
        self.hand_tracking_active = False
        self._last_sent_orientation = None  # Last orientation sent to robot [roll, pitch, yaw] degrees

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

        # Publishers for robot observations (actual state)
        self.obs_pose_pub = self.create_publisher(PoseStamped, 'robot_obs/pose', 10)
        self.obs_gripper_pub = self.create_publisher(Float32, 'robot_obs/gripper', 10)

        # Control loop timer
        self.control_timer = self.create_timer(1.0 / self.control_rate, self.control_loop)

        # State publishing timer (same rate as control)
        self.state_timer = self.create_timer(1.0 / self.control_rate, self.publish_robot_state)

        # Initialize gripper
        self.initialize_gripper()

        self.get_logger().info('XArm Hand Controller initialized')
        self.get_logger().info(f'Workspace bounds: {self._safety.get_bounds_str()}')
        self.get_logger().info(f'Soft start factor: {self.soft_start_factor} (velocity = {self._safety.get_effective_max_velocity()*1000:.0f} mm/s)')
        self.get_logger().info(f'Max position delta: {self._safety.get_effective_max_delta()*1000:.1f} mm/step')
        self.get_logger().info(f'Orientation bounds: roll=±{self.roll_limit}°, pitch=±{self.pitch_limit}°, yaw=±{self.yaw_limit}°')

    def setup_xarm(self):
        """Initialize XArm for control."""
        # Clear any existing errors first
        self.arm.clean_error()
        self.arm.clean_gripper_error()

        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)  # Start in position mode
        self.arm.set_state(state=0)
        time.sleep(1)

        # Configure hardware-level safety limits enforced by the controller
        # firmware. These prevent C24 (joint speed exceeds limit) and S23
        # (large motor position deviation) errors that software-side delta
        # limiting alone cannot catch — Mode 7's internal trajectory planner
        # can still generate high joint velocities for some Cartesian moves.
        code = self.arm.set_reduced_max_joint_speed(self.reduced_max_joint_speed, is_radian=False)
        self.get_logger().info(f'Set reduced max joint speed: {self.reduced_max_joint_speed} deg/s (code={code})')

        code = self.arm.set_reduced_max_tcp_speed(self.reduced_max_tcp_speed)
        self.get_logger().info(f'Set reduced max TCP speed: {self.reduced_max_tcp_speed} mm/s (code={code})')

        code = self.arm.set_reduced_mode(True)
        self.get_logger().info(f'Enabled reduced mode (code={code})')

        code = self.arm.set_joint_maxacc(self.joint_max_acc, is_radian=False)
        self.get_logger().info(f'Set joint max acceleration: {self.joint_max_acc} deg/s^2 (code={code})')

        code = self.arm.set_tcp_maxacc(self.tcp_max_acc)
        self.get_logger().info(f'Set TCP max acceleration: {self.tcp_max_acc} mm/s^2 (code={code})')

        code = self.arm.set_collision_sensitivity(self.collision_sensitivity)
        self.get_logger().info(f'Set collision sensitivity: {self.collision_sensitivity} (code={code})')

        self.get_logger().info('XArm initialized with safety limits')

    def initialize_gripper(self):
        """Initialize Lite6 gripper."""
        try:
            # For Lite6, just open the gripper to start
            code = self.arm.open_lite6_gripper()
            self.get_logger().info(f'Open Lite6 gripper: code={code}')

            if code == 0:
                self.gripper_enabled = True
                self.get_logger().info('Lite6 gripper initialized successfully')
            else:
                self.gripper_enabled = False
                self.get_logger().warn(f'Lite6 gripper init failed (code={code})')

        except Exception as e:
            self.get_logger().error(f'Gripper initialization failed: {e}')
            self.gripper_enabled = False

    def switch_to_online_trajectory_mode(self):
        """Switch to cartesian online trajectory planning mode for real-time control."""
        if not self.position_control_active:
            # Mode 7: cartesian online trajectory planning
            # Running command is interrupted when next command arrives
            # Handles acceleration/deceleration smoothly with speed limits
            self.arm.set_mode(7)
            self.arm.set_state(0)
            time.sleep(0.1)
            self.position_control_active = True
            self.get_logger().info(f'Online trajectory mode active (speed={self.max_linear_vel} mm/s)')

    def switch_to_position_mode(self):
        """Switch to position mode for trajectory moves (e.g., reset)."""
        self.arm.set_mode(0)  # Position control mode
        self.arm.set_state(0)
        time.sleep(0.1)
        self.position_control_active = False  # Will need to re-enable servo mode

    def apply_safety_limits(self, position: np.ndarray, allow_reset: bool = False) -> np.ndarray:
        """
        Apply all safety limits to target position.

        Args:
            position: XYZ position in meters
            allow_reset: If True, skip delta limiting (for reset to home)

        Returns:
            Safe position after all limits applied
        """
        # First clamp to workspace
        safe_pos = self._safety.clamp_position(position)

        # Then apply delta limit (max movement per step)
        safe_pos = self._safety.limit_delta(safe_pos, allow_reset=allow_reset)

        return safe_pos

    def clamp_orientation(self, quat) -> np.ndarray:
        """
        Convert quaternion to euler angles and clamp within bounds.

        The hand tracker publishes orientation as a quaternion in robot base frame.
        We convert to the same euler convention the xArm uses (intrinsic xyz)
        and clamp within bounds around the neutral orientation.

        Args:
            quat: Quaternion [x, y, z, w] from hand tracker

        Returns:
            Clamped euler angles [roll, pitch, yaw] in degrees
        """
        # Convert quaternion to euler angles (degrees)
        # xArm uses RPY (extrinsic xyz): R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
        rotation = R.from_quat([quat.x, quat.y, quat.z, quat.w])
        euler = rotation.as_euler('xyz', degrees=True)  # [roll, pitch, yaw]

        roll, pitch, yaw = euler[0], euler[1], euler[2]

        # Normalize angles to be within ±180° of their neutral value before
        # clamping. as_euler returns roll in [-180, 180], so a neutral of 180°
        # sits right at the discontinuity: a tiny perturbation can flip the
        # reported roll from +180° to -180°, which would then clamp to the
        # lower bound instead of staying near neutral.
        while roll - self.neutral_roll > 180.0:
            roll -= 360.0
        while roll - self.neutral_roll < -180.0:
            roll += 360.0
        while pitch - self.neutral_pitch > 180.0:
            pitch -= 360.0
        while pitch - self.neutral_pitch < -180.0:
            pitch += 360.0
        while yaw - self.neutral_yaw > 180.0:
            yaw -= 360.0
        while yaw - self.neutral_yaw < -180.0:
            yaw += 360.0

        # Clamp within bounds around neutral orientation
        raw_roll, raw_pitch, raw_yaw = roll, pitch, yaw

        # Lock roll at neutral (180°) to reduce DOF for data collection
        roll = self.neutral_roll
        # Lock pitch at neutral (0°) so tool Z always points down
        pitch = self.neutral_pitch
        yaw = np.clip(yaw,
                      self.neutral_yaw - self.yaw_limit,
                      self.neutral_yaw + self.yaw_limit)

        # Only log yaw when it actually hits the clamp bounds
        yaw_lower = self.neutral_yaw - self.yaw_limit
        yaw_upper = self.neutral_yaw + self.yaw_limit
        if raw_yaw < yaw_lower or raw_yaw > yaw_upper:
            self.get_logger().info(f'CLIPPED yaw: {raw_yaw:.1f} -> {yaw:.1f} (bounds [{yaw_lower:.0f}, {yaw_upper:.0f}])')

        clamped = np.array([roll, pitch, yaw])

        # Limit per-step orientation change to prevent servo errors (S23).
        # Without this, a large orientation jump commands huge wrist joint
        # motion in one step, exceeding the motor's tracking capability.
        if self._last_sent_orientation is not None:
            delta = clamped - self._last_sent_orientation
            delta = np.clip(delta, -self.max_orientation_delta, self.max_orientation_delta)
            clamped = self._last_sent_orientation + delta

        self._last_sent_orientation = clamped
        return clamped

    def tracking_status_callback(self, msg: Bool):
        """Update hand tracking status."""
        was_active = self.hand_tracking_active
        self.hand_tracking_active = msg.data

        if was_active and not self.hand_tracking_active:
            # Hand tracking lost - stop robot
            self.target_position = None
            self.target_orientation = None
            self._last_sent_orientation = None
            # In position mode, robot stops when no new commands are sent

    def hand_pose_callback(self, msg: PoseStamped):
        """
        Receive hand pose and update target position and orientation.
        """
        if self.is_resetting or self.is_paused:
            return

        try:
            # Extract raw hand position (in meters from hand tracker)
            raw_hand = np.array([
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z
            ])

            # Apply object-relative transform if enabled
            if self._use_object_relative:
                position = self._object_transform.transform(raw_hand)
            else:
                position = raw_hand

            # Apply all safety limits (workspace bounds + delta limiting)
            self.target_position = self.apply_safety_limits(position)

            # Extract and clamp orientation
            # Check if orientation is valid (not identity quaternion)
            quat = msg.pose.orientation
            if quat.w != 1.0 or quat.x != 0.0 or quat.y != 0.0 or quat.z != 0.0:
                self.target_orientation = self.clamp_orientation(quat)
            else:
                # Use neutral orientation if no valid orientation from tracker
                self.target_orientation = np.array([self.neutral_roll, self.neutral_pitch, self.neutral_yaw])

            # Publish action for data collection
            self.publish_action_pose(self.target_position, msg.header.stamp)

            # Ensure we're in servo mode for real-time control
            if not self.is_paused and self.hand_tracking_active:
                self.switch_to_online_trajectory_mode()

        except Exception as e:
            self.get_logger().error(f'Error in hand pose callback: {e}')

    def gripper_callback(self, msg: Float32):
        """Receive gripper command from hand tracker (0=open, 1=closed)."""
        if self.is_resetting or self.is_paused:
            return

        try:
            prev_cmd = self.gripper_cmd
            self.gripper_cmd = msg.data

            # Send gripper command if enabled (Lite6 uses open/close, not position)
            if self.gripper_enabled:
                # Threshold 0.75 = 4cm pinch distance (2cm→1.0, 10cm→0.0)
                # Close when pinch < 4cm, open when pinch >= 4cm
                if self.gripper_cmd > 0.75 and prev_cmd <= 0.75:
                    # Close gripper
                    self.arm.close_lite6_gripper()
                elif self.gripper_cmd <= 0.75 and prev_cmd > 0.75:
                    # Open gripper
                    self.arm.open_lite6_gripper()

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

        # Orientation quaternion from target or neutral
        if self.target_orientation is not None:
            roll, pitch, yaw = self.target_orientation
        else:
            roll, pitch, yaw = self.neutral_roll, self.neutral_pitch, self.neutral_yaw

        quat = R.from_euler('xyz', [
            math.radians(roll),
            math.radians(pitch),
            math.radians(yaw)
        ]).as_quat()

        msg.pose.orientation.x = float(quat[0])
        msg.pose.orientation.y = float(quat[1])
        msg.pose.orientation.z = float(quat[2])
        msg.pose.orientation.w = float(quat[3])

        self.action_pose_pub.publish(msg)

    def publish_robot_state(self):
        """Publish current robot position for visualization."""
        try:
            code, current_pos = self.arm.get_position(is_radian=True)
            if code != 0:
                return

            # Current position in mm, convert to meters
            x_m = current_pos[0] / 1000.0
            y_m = current_pos[1] / 1000.0
            z_m = current_pos[2] / 1000.0
            roll, pitch, yaw = current_pos[3:6]

            # Publish pose
            msg = PoseStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "base_link"
            msg.pose.position.x = x_m
            msg.pose.position.y = y_m
            msg.pose.position.z = z_m

            # Convert euler to quaternion
            quat = R.from_euler('xyz', [roll, pitch, yaw]).as_quat()
            msg.pose.orientation.x = float(quat[0])
            msg.pose.orientation.y = float(quat[1])
            msg.pose.orientation.z = float(quat[2])
            msg.pose.orientation.w = float(quat[3])

            self.obs_pose_pub.publish(msg)

            # Publish gripper state (disabled for now)
            gripper_msg = Float32()
            gripper_msg.data = self.gripper_cmd
            self.obs_gripper_pub.publish(gripper_msg)

        except Exception as e:
            pass  # Silently ignore errors in state publishing

    def recover_from_error(self):
        """
        Attempt to recover from a controller error (C24, S23, etc.).

        Clears the error, re-enables motion, and switches back to Mode 7.
        Returns True if recovery succeeded.
        """
        self.get_logger().warn(f'Recovering from error code: C{self.arm.error_code}')
        try:
            self.arm.clean_error()
            self.arm.clean_warn()
            self.arm.motion_enable(enable=True)
            self.arm.set_mode(0)
            self.arm.set_state(0)
            time.sleep(0.5)

            # Re-enter Mode 7 for real-time control
            self.position_control_active = False
            self.switch_to_online_trajectory_mode()

            if self.arm.error_code == 0:
                self.get_logger().info('Error recovery successful')
                return True
            else:
                self.get_logger().error(f'Recovery failed, error code still: C{self.arm.error_code}')
                return False
        except Exception as e:
            self.get_logger().error(f'Exception during recovery: {e}')
            return False

    def control_loop(self):
        """
        Main control loop - streams position commands using online trajectory mode.
        Runs at control_rate Hz. Each new command interrupts the previous one smoothly.
        """
        if (self.is_paused or self.is_resetting or
            self.target_position is None or not self.position_control_active):
            return

        # Check for controller errors (C24: joint speed exceeds limit, etc.)
        # and attempt auto-recovery so the session isn't bricked.
        if self.arm.error_code != 0:
            self.get_logger().warn(f'Controller error C{self.arm.error_code} detected, attempting recovery...')
            if not self.recover_from_error():
                return

        try:
            # Target position in mm
            target_x_mm = self.target_position[0] * 1000.0
            target_y_mm = self.target_position[1] * 1000.0
            target_z_mm = self.target_position[2] * 1000.0

            # Target orientation (use tracked or neutral)
            if self.target_orientation is not None:
                roll = self.target_orientation[0]
                pitch = self.target_orientation[1]
                yaw = self.target_orientation[2]
            else:
                roll = self.neutral_roll
                pitch = self.neutral_pitch
                yaw = self.neutral_yaw

            # Send position command (mode 7 interrupts previous motion smoothly)
            self.arm.set_position(
                x=target_x_mm,
                y=target_y_mm,
                z=target_z_mm,
                roll=roll,
                pitch=pitch,
                yaw=yaw,
                speed=self.max_linear_vel,
                is_radian=False,
                wait=False
            )

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
            self.target_orientation = None
            self._last_sent_orientation = None

            # Reset safety limit tracking
            self._safety.reset_tracking()

            try:
                self.switch_to_position_mode()

                # Home position (in mm and degrees) with neutral orientation
                code = self.arm.set_position(
                    x=200.0, y=0.0, z=250.0,
                    roll=self.neutral_roll, pitch=self.neutral_pitch, yaw=self.neutral_yaw,
                    speed=100, is_radian=False, wait=True
                )

                if code != 0:
                    self.get_logger().error(f'Reset position failed: {code}')
                else:
                    # Open gripper on reset (Lite6)
                    if self.gripper_enabled:
                        self.arm.open_lite6_gripper()
                        time.sleep(1)  # Wait for gripper to open
                        self.arm.stop_lite6_gripper()
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
            # In position mode, robot stops when control loop stops sending commands
        else:
            self.get_logger().info('XArm motion resumed')
            if self.target_position is not None and self.hand_tracking_active:
                self.switch_to_online_trajectory_mode()

    def destroy_node(self):
        """Clean shutdown."""
        self.get_logger().info('Disconnecting from XArm...')
        try:
            # Stop gripper motor before disconnecting
            if self.gripper_enabled:
                self.arm.stop_lite6_gripper()
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
