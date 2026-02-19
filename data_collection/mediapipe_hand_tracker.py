#!/usr/bin/env python3
"""
MediaPipe Hand Tracker Node

Subscribes to camera images, runs MediaPipe hand detection, and publishes:
- hand/pose: Hand position and orientation as PoseStamped
- hand/gripper_cmd: Gripper command from pinch gesture (Float32, 0-1)
- hand/tracking_active: Whether hand is currently detected (Bool)
- hand/landmarks: All 21 hand landmarks in robot frame (Float32MultiArray, 63 floats)

Supports two modes:
- RGB-D backprojection (use_depth=True): Uses aligned depth stream + camera intrinsics
  to produce accurate metric 3D hand positions in robot frame via known extrinsics.
- Fallback (use_depth=False): Uses MediaPipe's normalized landmark coordinates with
  heuristic scaling (original behavior).
"""

import os
import json
import math
import numpy as np
import cv2
import mediapipe as mp

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CompressedImage, Image, CameraInfo
from std_msgs.msg import Float32, Bool, Float32MultiArray, MultiArrayDimension
from cv_bridge import CvBridge
import message_filters

from utils.hand_to_action import HandToActionTransformer, get_hand_tracking_status

# Palm landmark indices (wrist + 4 MCP knuckles)
PALM_INDICES = [0, 5, 9, 13, 17]


class MediaPipeHandTracker(Node):
    """
    ROS2 node that runs MediaPipe hand detection and publishes hand pose.
    """
    def __init__(self):
        super().__init__('mediapipe_hand_tracker')

        # Parameters
        self.declare_parameter('position_scale', 0.3)
        self.declare_parameter('position_offset_x', 0.0)
        self.declare_parameter('position_offset_y', 0.0)
        self.declare_parameter('position_offset_z', 0.0)
        self.declare_parameter('filter_alpha', 0.3)
        self.declare_parameter('detection_confidence', 0.2)
        self.declare_parameter('tracking_confidence', 0.15)

        self.declare_parameter('hand_yaw_offset_deg', -90.0)

        # Depth parameters
        self.declare_parameter('use_depth', True)
        self.declare_parameter('max_valid_depth_m', 1.5)
        self.declare_parameter('min_valid_depth_m', 0.1)

        # Object-relative transform parameters
        self.declare_parameter('use_object_relative', True)
        self.declare_parameter('human_object_x', 0.365)
        self.declare_parameter('human_object_y', -0.36)
        self.declare_parameter('human_object_z', 0.0)
        self.declare_parameter('robot_object_x', 0.365)
        self.declare_parameter('robot_object_y', 0.0)
        self.declare_parameter('robot_object_z', 0.0)

        position_scale = self.get_parameter('position_scale').value
        position_offset = np.array([
            self.get_parameter('position_offset_x').value,
            self.get_parameter('position_offset_y').value,
            self.get_parameter('position_offset_z').value,
        ])
        filter_alpha = self.get_parameter('filter_alpha').value
        detection_conf = self.get_parameter('detection_confidence').value
        tracking_conf = self.get_parameter('tracking_confidence').value

        self._hand_yaw_offset = math.radians(self.get_parameter('hand_yaw_offset_deg').value)

        self._use_depth = self.get_parameter('use_depth').value
        self._max_valid_depth_m = self.get_parameter('max_valid_depth_m').value
        self._min_valid_depth_m = self.get_parameter('min_valid_depth_m').value

        # Object-relative transform
        self._use_object_relative = self.get_parameter('use_object_relative').value
        self._human_object_pos = np.array([
            self.get_parameter('human_object_x').value,
            self.get_parameter('human_object_y').value,
            self.get_parameter('human_object_z').value,
        ])
        self._robot_object_pos = np.array([
            self.get_parameter('robot_object_x').value,
            self.get_parameter('robot_object_y').value,
            self.get_parameter('robot_object_z').value,
        ])

        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # Hand to action transformer (used in fallback mode and for gripper detection)
        self.transformer = HandToActionTransformer(
            position_scale=position_scale,
            position_offset=position_offset,
            filter_alpha=filter_alpha
        )

        # CV bridge for image conversion
        self._bridge = CvBridge()

        # Camera intrinsics (populated from camera_info)
        self._fx = None
        self._fy = None
        self._cx = None
        self._cy = None
        self._intrinsics_received = False

        # Camera-to-robot extrinsics (loaded from calibration file or hardcoded fallback)
        self._cam_to_robot_rotation, self._cam_to_robot_translation = \
            self._load_head_camera_extrinsics()

        # Low-pass filters for depth-based position and yaw
        from utils.hand_to_action import LowPassFilter
        self._depth_position_filter = LowPassFilter(alpha=filter_alpha)
        self._yaw_filter = LowPassFilter(alpha=filter_alpha)

        # QoS for camera streams
        sensor_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST
        )

        # Depth mode state
        self._depth_mode_active = False  # True once synced frames start arriving
        self._fell_back_to_color = False  # True if auto-fallback triggered
        self._sync_count = 0

        if self._use_depth:
            # Subscribe to camera_info (one-shot for intrinsics)
            self.camera_info_sub = self.create_subscription(
                CameraInfo,
                '/rs_head/rs_head/color/camera_info',
                self.camera_info_callback,
                sensor_qos
            )

            # Synchronized color + depth subscription
            # Use raw Image (not compressed) because enable_sync mode
            # doesn't reliably publish the compressed transport variant
            self.color_sub = message_filters.Subscriber(
                self, Image,
                '/rs_head/rs_head/color/image_raw',
                qos_profile=sensor_qos
            )
            self.depth_sub = message_filters.Subscriber(
                self, Image,
                '/rs_head/rs_head/aligned_depth_to_color/image_raw',
                qos_profile=sensor_qos
            )
            self.time_sync = message_filters.ApproximateTimeSynchronizer(
                [self.color_sub, self.depth_sub],
                queue_size=5,
                slop=0.1,
            )
            self.time_sync.registerCallback(self.synced_image_callback)

            # Auto-fallback to color-only if sync doesn't work within 10s
            self._fallback_timer = self.create_timer(10.0, self._auto_fallback_callback)

            self.get_logger().info('Depth mode enabled - waiting for aligned depth + camera_info')
        else:
            # Fallback: subscribe to raw color only (original behavior)
            self.image_sub = self.create_subscription(
                Image,
                '/rs_head/rs_head/color/image_raw',
                self.image_callback,
                sensor_qos
            )
            self.get_logger().info('Depth mode disabled - using MediaPipe normalized coordinates')

        # Publishers
        self.hand_pose_pub = self.create_publisher(PoseStamped, 'hand/pose', 10)
        self.eef_target_pub = self.create_publisher(PoseStamped, 'eef/pose_target', 10)
        self.gripper_cmd_pub = self.create_publisher(Float32, 'hand/gripper_cmd', 10)
        self.tracking_active_pub = self.create_publisher(Bool, 'hand/tracking_active', 10)
        self.landmarks_pub = self.create_publisher(Float32MultiArray, 'hand/landmarks', 10)

        # Store transformation parameters for landmark conversion (fallback mode)
        self._position_scale = position_scale
        self._position_offset = position_offset

        # State
        self._last_tracking_active = False

        self.get_logger().info('MediaPipe Hand Tracker initialized')
        self.get_logger().info(f'Position scale: {position_scale}, offset: {position_offset}')
        if self._use_object_relative:
            self.get_logger().info(
                f'Object-relative transform: human={self._human_object_pos}, '
                f'robot={self._robot_object_pos}'
            )

    def _load_head_camera_extrinsics(self):
        """
        Load head camera extrinsics from camera_calibration.json if it exists.
        Falls back to hardcoded defaults if file not found.
        """
        default_rotation = np.array([
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
        ], dtype=np.float64)
        default_translation = np.array([0.37, 0.0, 1.02], dtype=np.float64)

        calib_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'camera_calibration.json'
        )

        if not os.path.exists(calib_path):
            self.get_logger().info('No camera_calibration.json — using hardcoded extrinsics')
            return default_rotation, default_translation

        try:
            with open(calib_path, 'r') as f:
                calib = json.load(f)

            head = calib.get('head_camera')
            if head is None:
                self.get_logger().warn('camera_calibration.json missing head_camera — using hardcoded')
                return default_rotation, default_translation

            rotation = np.array(head['rotation_cam_to_robot'], dtype=np.float64)
            translation = np.array(head['translation_cam_to_robot'], dtype=np.float64)

            self.get_logger().info(f'Loaded head camera extrinsics from {calib_path}')
            self.get_logger().info(
                f'  Translation: [{translation[0]:.4f}, {translation[1]:.4f}, {translation[2]:.4f}]'
            )
            return rotation, translation

        except Exception as e:
            self.get_logger().warn(f'Failed to load calibration: {e} — using hardcoded')
            return default_rotation, default_translation

    def _auto_fallback_callback(self):
        """Auto-fallback to color-only mode if sync never works."""
        self._fallback_timer.cancel()

        if self._sync_count > 0:
            # Depth mode working
            self._depth_mode_active = True
            return

        self.get_logger().warn(
            'Auto-fallback: switching to color-only mode (no depth sync after 10s). '
            'Hand tracking will use MediaPipe normalized coordinates.'
        )
        self._fell_back_to_color = True

        # Subscribe to color-only for the original image_callback
        sensor_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST
        )
        self.image_sub = self.create_subscription(
            Image,
            '/rs_head/rs_head/color/image_raw',
            self.image_callback,
            sensor_qos
        )

    def camera_info_callback(self, msg: CameraInfo):
        """Extract camera intrinsics from camera_info (one-shot)."""
        if self._intrinsics_received:
            return

        # K matrix is 3x3 row-major: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        self._fx = msg.k[0]
        self._fy = msg.k[4]
        self._cx = msg.k[2]
        self._cy = msg.k[5]
        self._intrinsics_received = True

        self.get_logger().info(
            f'Camera intrinsics received: fx={self._fx:.1f}, fy={self._fy:.1f}, '
            f'cx={self._cx:.1f}, cy={self._cy:.1f}'
        )

    def _sample_depth_patch(self, depth_image: np.ndarray, u: int, v: int) -> float:
        """
        Sample median depth from a 3x3 patch around (u, v), filtering invalid values.

        Args:
            depth_image: Depth image (uint16, values in millimeters)
            u: Pixel column
            v: Pixel row

        Returns:
            Depth in meters, or 0.0 if no valid depth found
        """
        h, w = depth_image.shape[:2]

        # Clamp patch bounds
        u_min = max(0, u - 1)
        u_max = min(w, u + 2)
        v_min = max(0, v - 1)
        v_max = min(h, v + 2)

        patch = depth_image[v_min:v_max, u_min:u_max].astype(np.float64)
        valid = patch[patch > 0]

        if len(valid) == 0:
            return 0.0

        depth_m = np.median(valid) / 1000.0  # mm -> meters

        if depth_m < self._min_valid_depth_m or depth_m > self._max_valid_depth_m:
            return 0.0

        return depth_m

    def _backproject_to_camera_frame(self, u: float, v: float, depth_m: float) -> np.ndarray:
        """
        Backproject a pixel (u, v) with depth to 3D point in camera frame.

        Args:
            u: Pixel column
            v: Pixel row
            depth_m: Depth in meters

        Returns:
            np.ndarray (3,): 3D point in camera frame [X, Y, Z]
        """
        x_cam = (u - self._cx) / self._fx * depth_m
        y_cam = (v - self._cy) / self._fy * depth_m
        z_cam = depth_m
        return np.array([x_cam, y_cam, z_cam])

    def _camera_to_robot_frame(self, point_cam: np.ndarray) -> np.ndarray:
        """
        Transform a 3D point from camera frame to robot frame.

        Args:
            point_cam: 3D point in camera frame

        Returns:
            np.ndarray (3,): 3D point in robot frame
        """
        return self._cam_to_robot_rotation @ point_cam + self._cam_to_robot_translation

    def get_backprojected_palm_position(
        self, landmarks, depth_image: np.ndarray, img_w: int, img_h: int
    ) -> np.ndarray:
        """
        Backproject 5 palm landmarks using depth, average valid ones.

        Args:
            landmarks: MediaPipe hand landmarks
            depth_image: Aligned depth image (uint16, mm)
            img_w: Image width
            img_h: Image height

        Returns:
            np.ndarray (3,) in robot frame, or None if all depths invalid
        """
        valid_robot_points = []

        for idx in PALM_INDICES:
            lm = landmarks[idx]
            u = int(lm.x * img_w)
            v = int(lm.y * img_h)

            # Clamp to image bounds
            u = max(0, min(u, img_w - 1))
            v = max(0, min(v, img_h - 1))

            depth_m = self._sample_depth_patch(depth_image, u, v)
            if depth_m > 0.0:
                point_cam = self._backproject_to_camera_frame(float(u), float(v), depth_m)
                point_robot = self._camera_to_robot_frame(point_cam)
                valid_robot_points.append(point_robot)

        if len(valid_robot_points) == 0:
            return None

        return np.mean(valid_robot_points, axis=0)

    def get_hand_yaw_depth(
        self, landmarks, depth_image: np.ndarray, img_w: int, img_h: int
    ):
        """
        Compute hand yaw angle from backprojected wrist and middle_mcp landmarks.

        Returns:
            Yaw angle in radians, or None if depth invalid for either landmark.
        """
        points = []
        for idx in [0, 9]:  # wrist, middle_mcp
            lm = landmarks[idx]
            u = max(0, min(int(lm.x * img_w), img_w - 1))
            v = max(0, min(int(lm.y * img_h), img_h - 1))
            depth_m = self._sample_depth_patch(depth_image, u, v)
            if depth_m <= 0.0:
                return None
            point_cam = self._backproject_to_camera_frame(float(u), float(v), depth_m)
            points.append(self._camera_to_robot_frame(point_cam))

        forward = points[1] - points[0]  # wrist → middle_mcp
        return math.atan2(forward[1], forward[0])

    def get_hand_yaw_pixels(self, landmarks):
        """
        Compute hand yaw angle from 2D pixel coordinates (fallback, no depth).

        Camera-to-robot mapping: cam X → robot Y, cam Y → robot X.
        """
        dx = landmarks[9].x - landmarks[0].x  # camera X delta
        dy = landmarks[9].y - landmarks[0].y  # camera Y delta
        # In robot frame: forward_x ~ dy, forward_y ~ dx
        return math.atan2(dx, dy)

    def publish_landmarks_3d(
        self, landmarks, depth_image: np.ndarray, img_w: int, img_h: int
    ):
        """
        Backproject all 21 hand landmarks to robot frame and publish.

        For landmarks with invalid depth, uses the nearest valid depth or skips.
        """
        robot_landmarks = []

        for lm in landmarks:
            u = int(lm.x * img_w)
            v = int(lm.y * img_h)
            u = max(0, min(u, img_w - 1))
            v = max(0, min(v, img_h - 1))

            depth_m = self._sample_depth_patch(depth_image, u, v)
            if depth_m > 0.0:
                point_cam = self._backproject_to_camera_frame(float(u), float(v), depth_m)
                point_robot = self._camera_to_robot_frame(point_cam)
                robot_landmarks.extend([float(point_robot[0]), float(point_robot[1]), float(point_robot[2])])
            else:
                # Fallback: use a default depth of 0.5m for visualization continuity
                point_cam = self._backproject_to_camera_frame(float(u), float(v), 0.5)
                point_robot = self._camera_to_robot_frame(point_cam)
                robot_landmarks.extend([float(point_robot[0]), float(point_robot[1]), float(point_robot[2])])

        msg = Float32MultiArray()
        msg.layout.dim = [
            MultiArrayDimension(label='landmarks', size=21, stride=63),
            MultiArrayDimension(label='xyz', size=3, stride=3)
        ]
        msg.layout.data_offset = 0
        msg.data = robot_landmarks

        self.landmarks_pub.publish(msg)

    def synced_image_callback(self, color_msg: Image, depth_msg: Image):
        """
        Synchronized color + depth callback. Runs MediaPipe on color,
        backprojects landmarks using depth, publishes in robot frame.
        """
        if not self._intrinsics_received:
            return

        self._sync_count += 1
        if self._sync_count == 1:
            self.get_logger().info('First synced color+depth frame received — depth mode active')

        try:
            # Convert raw color image (RGB8 from realsense) to BGR then RGB for MediaPipe
            rgb_image = self._bridge.imgmsg_to_cv2(color_msg, desired_encoding='rgb8')

            # Convert depth image (16UC1, values in mm)
            depth_image = self._bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

            img_h, img_w = rgb_image.shape[:2]

            # Run MediaPipe hand detection
            results = self.hands.process(rgb_image)

            tracking_active = (results.multi_hand_landmarks is not None and
                              len(results.multi_hand_landmarks) > 0)

            # Publish tracking status
            status_msg = Bool()
            status_msg.data = tracking_active
            self.tracking_active_pub.publish(status_msg)

            if tracking_active:
                landmarks = results.multi_hand_landmarks[0].landmark

                # Compute hand yaw from depth
                raw_yaw = self.get_hand_yaw_depth(
                    landmarks, depth_image, img_w, img_h
                )
                if raw_yaw is not None:
                    yaw = self._yaw_filter.filter(np.array([raw_yaw]))[0]
                else:
                    # Keep last filtered value, or 0 if no previous
                    yaw = self._yaw_filter.value[0] if self._yaw_filter.value is not None else 0.0

                # Attempt backprojected palm position
                palm_robot = self.get_backprojected_palm_position(
                    landmarks, depth_image, img_w, img_h
                )

                if palm_robot is not None:
                    # Apply low-pass filter
                    palm_robot = self._depth_position_filter.filter(palm_robot)
                    self.publish_hand_pose(palm_robot, color_msg.header.stamp, yaw)
                else:
                    # Fallback to MediaPipe normalized coordinates
                    action = self.transformer.landmarks_to_action(landmarks)
                    self.publish_hand_pose(action['position'], color_msg.header.stamp, yaw)

                # Publish gripper from pinch gesture (always uses MediaPipe landmarks)
                gripper = self.transformer.get_gripper_from_pinch(landmarks)
                gripper_msg = Float32()
                gripper_msg.data = gripper
                self.gripper_cmd_pub.publish(gripper_msg)

                # Publish 3D landmarks
                self.publish_landmarks_3d(landmarks, depth_image, img_w, img_h)

                if not self._last_tracking_active:
                    self.get_logger().info('Hand tracking started (depth mode)')

            else:
                if self._last_tracking_active:
                    self.get_logger().info('Hand tracking lost')
                    self.transformer.reset()
                    self._depth_position_filter.reset()
                    self._yaw_filter.reset()

            self._last_tracking_active = tracking_active

        except Exception as e:
            self.get_logger().error(f'Error in synced image callback: {e}')

    def image_callback(self, msg: Image):
        """
        Process incoming camera image with MediaPipe hand detection.
        Fallback path when use_depth is False.
        """
        try:
            # Convert raw image to RGB for MediaPipe
            rgb_image = self._bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')

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

                # Compute hand yaw from pixel coordinates
                raw_yaw = self.get_hand_yaw_pixels(landmarks)
                yaw = self._yaw_filter.filter(np.array([raw_yaw]))[0]

                # Transform to robot action
                action = self.transformer.landmarks_to_action(landmarks)

                # Publish hand pose with yaw
                self.publish_hand_pose(action['position'], msg.header.stamp, yaw)

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
                    self._yaw_filter.reset()

            self._last_tracking_active = tracking_active

        except Exception as e:
            self.get_logger().error(f'Error in image callback: {e}')

    def publish_hand_pose(self, position: np.ndarray, stamp, yaw: float = 0.0):
        """
        Publish hand pose as PoseStamped (actual hand position in robot frame).
        Also publishes the EEF target (object-relative transformed position).

        Position is in robot frame (meters).
        Orientation encodes hand yaw as a Z-rotation quaternion.
        """
        # Apply hand-to-robot yaw offset, then encode as Z-rotation quaternion
        yaw = yaw - self._hand_yaw_offset
        qx, qy = 0.0, 0.0
        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0)

        msg = PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = "base_link"

        msg.pose.position.x = float(position[0])
        msg.pose.position.y = float(position[1])
        msg.pose.position.z = float(position[2])

        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw

        self.hand_pose_pub.publish(msg)

        # Compute and publish EEF target (object-relative transform)
        if self._use_object_relative:
            delta = position - self._human_object_pos
            target = self._robot_object_pos + delta
        else:
            target = position

        target_msg = PoseStamped()
        target_msg.header.stamp = stamp
        target_msg.header.frame_id = "base_link"
        target_msg.pose.position.x = float(target[0])
        target_msg.pose.position.y = float(target[1])
        target_msg.pose.position.z = float(target[2])
        target_msg.pose.orientation.x = qx
        target_msg.pose.orientation.y = qy
        target_msg.pose.orientation.z = qz
        target_msg.pose.orientation.w = qw

        self.eef_target_pub.publish(target_msg)

    def publish_landmarks(self, landmarks):
        """
        Publish all 21 hand landmarks transformed to robot frame.
        Fallback path using MediaPipe normalized coordinates.

        Output format: Float32MultiArray with 63 floats (21 landmarks x 3 coords)
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
