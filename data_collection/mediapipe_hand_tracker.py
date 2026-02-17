#!/usr/bin/env python3
"""
MediaPipe Hand Tracker Node

Subscribes to camera images, runs MediaPipe hand detection, and publishes:
- hand/pose: Hand position and orientation as PoseStamped
- hand/gripper_cmd: Gripper command from pinch gesture (Float32, 0-1)
- hand/tracking_active: Whether hand is currently detected (Bool)
- hand/landmarks: All 21 hand landmarks in robot frame (Float32MultiArray, 63 floats)

Uses one of the RealSense cameras for hand tracking.
"""

import numpy as np
import cv2
import mediapipe as mp

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32, Bool, Float32MultiArray, MultiArrayDimension
from cv_bridge import CvBridge

from utils.hand_to_action import HandToActionTransformer, get_hand_tracking_status


class MediaPipeHandTracker(Node):
    """
    ROS2 node that runs MediaPipe hand detection and publishes hand pose.
    """
    def __init__(self):
        super().__init__('mediapipe_hand_tracker')

        # Parameters
        self.declare_parameter('position_scale', 0.3)
        self.declare_parameter('position_offset_x', 0.3)
        self.declare_parameter('position_offset_y', 0.0)
        self.declare_parameter('position_offset_z', 0.2)
        self.declare_parameter('filter_alpha', 0.3)
        self.declare_parameter('detection_confidence', 0.7)
        self.declare_parameter('tracking_confidence', 0.5)

        position_scale = self.get_parameter('position_scale').value
        position_offset = np.array([
            self.get_parameter('position_offset_x').value,
            self.get_parameter('position_offset_y').value,
            self.get_parameter('position_offset_z').value,
        ])
        filter_alpha = self.get_parameter('filter_alpha').value
        detection_conf = self.get_parameter('detection_confidence').value
        tracking_conf = self.get_parameter('tracking_confidence').value

        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # Hand to action transformer
        self.transformer = HandToActionTransformer(
            position_scale=position_scale,
            position_offset=position_offset,
            filter_alpha=filter_alpha
        )

        # CV bridge for image conversion
        self._bridge = CvBridge()

        # QoS for camera streams
        sensor_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST
        )

        # Subscribe to head camera for hand tracking
        self.image_sub = self.create_subscription(
            CompressedImage,
            '/rs_head/rs_head/color/image_raw/compressed',
            self.image_callback,
            sensor_qos
        )

        # Publishers
        self.hand_pose_pub = self.create_publisher(PoseStamped, 'hand/pose', 10)
        self.gripper_cmd_pub = self.create_publisher(Float32, 'hand/gripper_cmd', 10)
        self.tracking_active_pub = self.create_publisher(Bool, 'hand/tracking_active', 10)
        self.landmarks_pub = self.create_publisher(Float32MultiArray, 'hand/landmarks', 10)

        # Store transformation parameters for landmark conversion
        self._position_scale = position_scale
        self._position_offset = position_offset

        # State
        self._last_tracking_active = False

        self.get_logger().info('MediaPipe Hand Tracker initialized')
        self.get_logger().info(f'Position scale: {position_scale}, offset: {position_offset}')

    def image_callback(self, msg: CompressedImage):
        """
        Process incoming camera image with MediaPipe hand detection.
        """
        try:
            # Decompress image
            bgr_image = self._bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Convert to RGB for MediaPipe
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

            # Run MediaPipe hand detection
            results = self.hands.process(rgb_image)

            # Check if hand detected
            tracking_active = (results.multi_hand_landmarks is not None and
                              len(results.multi_hand_landmarks) > 0)

            # Publish tracking status
            status_msg = Bool()
            status_msg.data = tracking_active
            self.tracking_active_pub.publish(status_msg)

            if tracking_active:
                # Get first hand landmarks
                landmarks = results.multi_hand_landmarks[0].landmark

                # Transform to robot action
                action = self.transformer.landmarks_to_action(landmarks)

                # Publish hand pose
                self.publish_hand_pose(action['position'], msg.header.stamp)

                # Publish gripper command
                gripper_msg = Float32()
                gripper_msg.data = action['gripper']
                self.gripper_cmd_pub.publish(gripper_msg)

                # Publish all 21 landmarks in robot frame
                self.publish_landmarks(landmarks)

                if not self._last_tracking_active:
                    self.get_logger().info('Hand tracking started')

            else:
                if self._last_tracking_active:
                    self.get_logger().info('Hand tracking lost')
                    # Reset filters when tracking is lost
                    self.transformer.reset()

            self._last_tracking_active = tracking_active

        except Exception as e:
            self.get_logger().error(f'Error in image callback: {e}')

    def publish_hand_pose(self, position: np.ndarray, stamp):
        """
        Publish hand pose as PoseStamped.

        Position is in robot frame (meters).
        Orientation is set to identity (not used for position-only control).
        """
        msg = PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = "base_link"

        msg.pose.position.x = float(position[0])
        msg.pose.position.y = float(position[1])
        msg.pose.position.z = float(position[2])

        # Identity quaternion (orientation not used for position-only control)
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = 0.0
        msg.pose.orientation.w = 1.0

        self.hand_pose_pub.publish(msg)

    def publish_landmarks(self, landmarks):
        """
        Publish all 21 hand landmarks transformed to robot frame.

        Output format: Float32MultiArray with 63 floats (21 landmarks × 3 coords)
        Landmarks are in robot frame (meters).
        """
        # Transform each landmark to robot frame
        robot_landmarks = []
        for lm in landmarks:
            # Get MediaPipe normalized coordinates
            mp_pos = np.array([lm.x, lm.y, lm.z])

            # Center around 0.5
            centered_x = mp_pos[0] - 0.5
            centered_y = mp_pos[1] - 0.5
            depth_z = mp_pos[2]

            # Transform to robot frame (same as in hand_to_action.py)
            robot_pos = np.array([
                -depth_z * self._position_scale * 2.0,
                -centered_x * self._position_scale * 2.0,
                -centered_y * self._position_scale * 2.0,
            ])
            robot_pos = robot_pos + self._position_offset

            robot_landmarks.extend([float(robot_pos[0]), float(robot_pos[1]), float(robot_pos[2])])

        # Create Float32MultiArray message
        msg = Float32MultiArray()
        msg.layout.dim = [
            MultiArrayDimension(label='landmarks', size=21, stride=63),
            MultiArrayDimension(label='xyz', size=3, stride=3)
        ]
        msg.layout.data_offset = 0
        msg.data = robot_landmarks

        self.landmarks_pub.publish(msg)

    def destroy_node(self):
        """Clean up MediaPipe resources."""
        self.hands.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    node = MediaPipeHandTracker()

    try:
        node.get_logger().info('Running MediaPipe Hand Tracker...')
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
