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

Optionally fuses hand tracking from both head and front cameras (use_front_camera=True).
"""

import os
import json
import math
import time
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

# Max age (seconds) for a camera result to be considered valid for fusion
FUSION_STALENESS_S = 0.15


class CameraState:
    """Per-camera state for intrinsics, extrinsics, filters, and latest result."""
    def __init__(self, name: str, rotation: np.ndarray, translation: np.ndarray, filter_alpha: float):
        self.name = name
        # Intrinsics (populated from camera_info)
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.intrinsics_received = False
        # Extrinsics
        self.cam_to_robot_rotation = rotation
        self.cam_to_robot_translation = translation
        # Sync state
        self.sync_count = 0
        # Per-camera low-pass filters
        from utils.hand_to_action import LowPassFilter
        self.position_filter = LowPassFilter(alpha=filter_alpha)
        self.yaw_filter = LowPassFilter(alpha=filter_alpha)
        # Latest detection result
        self.latest_position = None  # np.ndarray (3,) in robot frame
        self.latest_yaw = None       # float, radians
        self.latest_time = None      # float, time.monotonic()
        self.latest_landmarks = None # MediaPipe landmarks (for gripper/landmarks publishing)


class MediaPipeHandTracker(Node):
    """
    ROS2 node that runs MediaPipe hand detection and publishes hand pose.
    """
    def __init__(self):
        super().__init__('mediapipe_hand_tracker')

        # Parameters
        self.declare_parameter('position_scale', 0.3)
        self.declare_parameter('position_offset_x', 0.0)
        self.declare_parameter('position_offset_y', 0.3)
        self.declare_parameter('position_offset_z', 0.0)
        self.declare_parameter('filter_alpha', 0.3)
        self.declare_parameter('detection_confidence', 0.2)
        self.declare_parameter('tracking_confidence', 0.15)

        self.declare_parameter('hand_yaw_offset_deg', -90.0)

        # Handedness: 'right' uses the defaults above; 'left' swaps in overrides.
        # Only yaw and fine_offset_y differ between hands (x/z are the same).
        self.declare_parameter('handedness', 'right')
        self.declare_parameter('left_hand_yaw_offset_deg', -60.0)
        self.declare_parameter('left_fine_offset_y', -0.05)

        # Depth parameters
        self.declare_parameter('use_depth', True)
        self.declare_parameter('max_valid_depth_m', 1.5)
        self.declare_parameter('min_valid_depth_m', 0.1)

        # Front camera fusion (disabled by default)
        self.declare_parameter('use_front_camera', True)

        # Object-relative transform parameters
        self.declare_parameter('use_object_relative', True)
        self.declare_parameter('human_object_x', 0.365)
        self.declare_parameter('human_object_y', -0.36)
        self.declare_parameter('human_object_z', 0.0)
        self.declare_parameter('robot_object_x', 0.365)
        self.declare_parameter('robot_object_y', 0.0)
        self.declare_parameter('robot_object_z', 0.0)

        # Final position offset for fine-tuning hand-to-gripper alignment (meters)
        self.declare_parameter('fine_offset_x', 0.0)
        self.declare_parameter('fine_offset_y', 0.02)
        # use 0 for full gripper height, -0.092 for grasping from top
        self.declare_parameter('fine_offset_z', -0.092)

        position_scale = self.get_parameter('position_scale').value
        position_offset = np.array([
            self.get_parameter('position_offset_x').value,
            self.get_parameter('position_offset_y').value,
            self.get_parameter('position_offset_z').value,
        ])
        filter_alpha = self.get_parameter('filter_alpha').value
        detection_conf = self.get_parameter('detection_confidence').value
        tracking_conf = self.get_parameter('tracking_confidence').value

        self._handedness = self.get_parameter('handedness').value
        if self._handedness == 'left':
            self._hand_yaw_offset = math.radians(
                self.get_parameter('left_hand_yaw_offset_deg').value)
        else:
            self._hand_yaw_offset = math.radians(
                self.get_parameter('hand_yaw_offset_deg').value)

        self._use_depth = self.get_parameter('use_depth').value
        self._max_valid_depth_m = self.get_parameter('max_valid_depth_m').value
        self._min_valid_depth_m = self.get_parameter('min_valid_depth_m').value
        self._use_front_camera = self.get_parameter('use_front_camera').value

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
        if self._handedness == 'left':
            fine_offset_y = self.get_parameter('left_fine_offset_y').value
        else:
            fine_offset_y = self.get_parameter('fine_offset_y').value

        self._fine_offset = np.array([
            self.get_parameter('fine_offset_x').value,
            fine_offset_y,
            self.get_parameter('fine_offset_z').value,
        ])

        self.get_logger().info(f'Handedness: {self._handedness} | '
            f'yaw_offset={math.degrees(self._hand_yaw_offset):.1f}deg | '
            f'fine_offset={self._fine_offset}')

        # Initialize MediaPipe Hands (one per camera stream for independent tracking state)
        self.mp_hands = mp.solutions.hands
        self._hands_head = self.mp_hands.Hands(
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

        # Camera states
        head_rot, head_trans = self._load_camera_extrinsics('head_camera')
        self._head = CameraState('head', head_rot, head_trans, filter_alpha)

        self._front = None
        if self._use_front_camera:
            front_rot, front_trans = self._load_camera_extrinsics('front_camera')
            self._front = CameraState('front', front_rot, front_trans, filter_alpha)
            self._hands_front = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=detection_conf,
                min_tracking_confidence=tracking_conf
            )

        # Fused position filter (applied after averaging camera results)
        from utils.hand_to_action import LowPassFilter
        self._fused_position_filter = LowPassFilter(alpha=filter_alpha)
        self._fused_yaw_filter = LowPassFilter(alpha=filter_alpha)

        # QoS for camera streams
        sensor_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST
        )

        # Depth mode state
        self._fell_back_to_color = False

        if self._use_depth:
            # Head camera subscriptions
            self._head_info_sub = self.create_subscription(
                CameraInfo,
                '/rs_head/rs_head/color/camera_info',
                lambda msg: self._camera_info_callback(msg, self._head),
                sensor_qos
            )
            self._head_color_sub = message_filters.Subscriber(
                self, Image,
                '/rs_head/rs_head/color/image_raw',
                qos_profile=sensor_qos
            )
            self._head_depth_sub = message_filters.Subscriber(
                self, Image,
                '/rs_head/rs_head/aligned_depth_to_color/image_raw',
                qos_profile=sensor_qos
            )
            self._head_sync = message_filters.ApproximateTimeSynchronizer(
                [self._head_color_sub, self._head_depth_sub],
                queue_size=30,
                slop=0.1,
            )
            self._head_sync.registerCallback(
                lambda color, depth: self._synced_callback(color, depth, self._head, self._hands_head)
            )
            self._head_fallback_timer = self.create_timer(
                10.0, lambda: self._auto_fallback_callback(self._head)
            )

            # Front camera subscriptions (only if enabled)
            if self._use_front_camera:
                self._front_info_sub = self.create_subscription(
                    CameraInfo,
                    '/rs_front/rs_front/color/camera_info',
                    lambda msg: self._camera_info_callback(msg, self._front),
                    sensor_qos
                )
                self._front_color_sub = message_filters.Subscriber(
                    self, Image,
                    '/rs_front/rs_front/color/image_raw',
                    qos_profile=sensor_qos
                )
                self._front_depth_sub = message_filters.Subscriber(
                    self, Image,
                    '/rs_front/rs_front/aligned_depth_to_color/image_raw',
                    qos_profile=sensor_qos
                )
                self._front_sync = message_filters.ApproximateTimeSynchronizer(
                    [self._front_color_sub, self._front_depth_sub],
                    queue_size=30,
                    slop=0.1,
                )
                self._front_sync.registerCallback(
                    lambda color, depth: self._synced_callback(color, depth, self._front, self._hands_front)
                )
                self._front_fallback_timer = self.create_timer(
                    10.0, lambda: self._auto_fallback_callback(self._front)
                )

            self.get_logger().info('Depth mode enabled - waiting for aligned depth + camera_info')
            if self._use_front_camera:
                self.get_logger().info('Front camera fusion enabled')
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
        if np.any(self._fine_offset != 0):
            self.get_logger().info(f'Fine offset: {self._fine_offset}')


    def _load_camera_extrinsics(self, camera_key: str):
        """
        Load camera extrinsics from camera_calibration.json.
        Falls back to hardcoded defaults if file not found.

        Args:
            camera_key: 'head_camera' or 'front_camera'
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
            self.get_logger().info(f'No camera_calibration.json — using hardcoded extrinsics for {camera_key}')
            return default_rotation, default_translation

        try:
            with open(calib_path, 'r') as f:
                calib = json.load(f)

            cam = calib.get(camera_key)
            if cam is None:
                self.get_logger().warn(f'camera_calibration.json missing {camera_key} — using hardcoded')
                return default_rotation, default_translation

            rotation = np.array(cam['rotation_cam_to_robot'], dtype=np.float64)
            translation = np.array(cam['translation_cam_to_robot'], dtype=np.float64)

            self.get_logger().info(f'Loaded {camera_key} extrinsics from {calib_path}')
            self.get_logger().info(
                f'  Translation: [{translation[0]:.4f}, {translation[1]:.4f}, {translation[2]:.4f}]'
            )
            return rotation, translation

        except Exception as e:
            self.get_logger().warn(f'Failed to load {camera_key} calibration: {e} — using hardcoded')
            return default_rotation, default_translation

    def _auto_fallback_callback(self, cam: CameraState):
        """Auto-fallback to color-only mode if sync never works for a camera."""
        # Cancel timer (only fires once per camera)
        if cam.name == 'head':
            self._head_fallback_timer.cancel()
        elif cam.name == 'front':
            self._front_fallback_timer.cancel()

        if cam.sync_count > 0:
            return

        self.get_logger().warn(
            f'Auto-fallback: {cam.name} camera — no depth sync after 10s.'
        )

        if cam.name == 'head':
            # Head camera falls back to color-only
            self._fell_back_to_color = True
            sensor_qos = QoSProfile(
                depth=10,
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST
            )
            self.image_sub = self.create_subscription(
                Image,
                '/rs_head/rs_head/color/image_raw',
                self.image_callback,
                sensor_qos
            )

    def _camera_info_callback(self, msg: CameraInfo, cam: CameraState):
        """Extract camera intrinsics from camera_info (one-shot per camera)."""
        if cam.intrinsics_received:
            return

        cam.fx = msg.k[0]
        cam.fy = msg.k[4]
        cam.cx = msg.k[2]
        cam.cy = msg.k[5]
        cam.intrinsics_received = True

        self.get_logger().info(
            f'{cam.name} camera intrinsics: fx={cam.fx:.1f}, fy={cam.fy:.1f}, '
            f'cx={cam.cx:.1f}, cy={cam.cy:.1f}'
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

    def _backproject_to_camera_frame(self, u: float, v: float, depth_m: float,
                                      cam: CameraState) -> np.ndarray:
        """Backproject a pixel (u, v) with depth to 3D point in camera frame."""
        x_cam = (u - cam.cx) / cam.fx * depth_m
        y_cam = (v - cam.cy) / cam.fy * depth_m
        z_cam = depth_m
        return np.array([x_cam, y_cam, z_cam])

    def _camera_to_robot_frame(self, point_cam: np.ndarray, cam: CameraState) -> np.ndarray:
        """Transform a 3D point from camera frame to robot frame."""
        return cam.cam_to_robot_rotation @ point_cam + cam.cam_to_robot_translation

    def get_backprojected_palm_position(
        self, landmarks, depth_image: np.ndarray, img_w: int, img_h: int,
        cam: CameraState
    ) -> np.ndarray:
        """
        Backproject 5 palm landmarks using depth, average valid ones.

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
                point_cam = self._backproject_to_camera_frame(float(u), float(v), depth_m, cam)
                point_robot = self._camera_to_robot_frame(point_cam, cam)
                valid_robot_points.append(point_robot)

        if len(valid_robot_points) == 0:
            return None

        return np.mean(valid_robot_points, axis=0)

    def get_hand_yaw_depth(
        self, landmarks, depth_image: np.ndarray, img_w: int, img_h: int,
        cam: CameraState
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
            point_cam = self._backproject_to_camera_frame(float(u), float(v), depth_m, cam)
            points.append(self._camera_to_robot_frame(point_cam, cam))

        forward = points[1] - points[0]  # wrist → middle_mcp
        return math.atan2(forward[1], forward[0])

    def _get_first_hand(self, results):
        """Return landmarks for the first detected hand, or None."""
        if (results.multi_hand_landmarks is not None and
            len(results.multi_hand_landmarks) > 0):
            return results.multi_hand_landmarks[0].landmark
        return None

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
        self, landmarks, depth_image: np.ndarray, img_w: int, img_h: int,
        cam: CameraState
    ):
        """Backproject all 21 hand landmarks to robot frame and publish."""
        robot_landmarks = []

        for lm in landmarks:
            u = int(lm.x * img_w)
            v = int(lm.y * img_h)
            u = max(0, min(u, img_w - 1))
            v = max(0, min(v, img_h - 1))

            depth_m = self._sample_depth_patch(depth_image, u, v)
            if depth_m > 0.0:
                point_cam = self._backproject_to_camera_frame(float(u), float(v), depth_m, cam)
                point_robot = self._camera_to_robot_frame(point_cam, cam)
                robot_landmarks.extend([float(point_robot[0]), float(point_robot[1]), float(point_robot[2])])
            else:
                point_cam = self._backproject_to_camera_frame(float(u), float(v), 0.5, cam)
                point_robot = self._camera_to_robot_frame(point_cam, cam)
                robot_landmarks.extend([float(point_robot[0]), float(point_robot[1]), float(point_robot[2])])

        msg = Float32MultiArray()
        msg.layout.dim = [
            MultiArrayDimension(label='landmarks', size=21, stride=63),
            MultiArrayDimension(label='xyz', size=3, stride=3)
        ]
        msg.layout.data_offset = 0
        msg.data = robot_landmarks

        self.landmarks_pub.publish(msg)

    def _synced_callback(self, color_msg: Image, depth_msg: Image,
                          cam: CameraState, hands_detector):
        """
        Synchronized color + depth callback for a single camera.
        Runs MediaPipe, backprojects, stores result in cam state, then fuses.
        """
        if not cam.intrinsics_received:
            return

        cam.sync_count += 1
        if cam.sync_count == 1:
            self.get_logger().info(f'{cam.name} camera: first synced color+depth frame — depth mode active')

        try:
            rgb_image = self._bridge.imgmsg_to_cv2(color_msg, desired_encoding='rgb8')
            depth_image = self._bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

            img_h, img_w = rgb_image.shape[:2]

            results = hands_detector.process(rgb_image)

            landmarks = self._get_first_hand(results)
            detected = landmarks is not None

            if detected:

                # Compute yaw
                raw_yaw = self.get_hand_yaw_depth(landmarks, depth_image, img_w, img_h, cam)
                if raw_yaw is not None:
                    yaw = cam.yaw_filter.filter(np.array([raw_yaw]))[0]
                else:
                    yaw = cam.yaw_filter.value[0] if cam.yaw_filter.value is not None else 0.0

                # Compute position
                palm_robot = self.get_backprojected_palm_position(
                    landmarks, depth_image, img_w, img_h, cam
                )

                if palm_robot is not None:
                    palm_robot = cam.position_filter.filter(palm_robot)
                    cam.latest_position = palm_robot
                    cam.latest_yaw = yaw
                    cam.latest_time = time.monotonic()
                    cam.latest_landmarks = landmarks
                else:
                    # Depth fallback — use MediaPipe normalized coords
                    action = self.transformer.landmarks_to_action(landmarks)
                    cam.latest_position = action['position']
                    cam.latest_yaw = yaw
                    cam.latest_time = time.monotonic()
                    cam.latest_landmarks = landmarks
            else:
                # No hand detected by this camera
                cam.latest_position = None
                cam.latest_yaw = None
                cam.latest_landmarks = None

            # Fuse results from all cameras and publish
            self._fuse_and_publish(color_msg.header.stamp, depth_image, img_w, img_h)

        except Exception as e:
            self.get_logger().error(f'Error in {cam.name} synced callback: {e}')

    def _fuse_and_publish(self, stamp, depth_image=None, img_w=0, img_h=0):
        """Fuse results from head (and optionally front) cameras and publish."""
        now = time.monotonic()

        # Collect valid (non-stale) results
        positions = []
        yaws = []
        primary_cam = None  # Camera used for gripper/landmarks

        for cam in [self._head, self._front]:
            if cam is None:
                continue
            if (cam.latest_position is not None and
                cam.latest_time is not None and
                (now - cam.latest_time) < FUSION_STALENESS_S):
                positions.append(cam.latest_position)
                yaws.append(cam.latest_yaw)
                if primary_cam is None:
                    primary_cam = cam  # Head is checked first, so it's preferred

        tracking_active = len(positions) > 0

        # Publish tracking status
        status_msg = Bool()
        status_msg.data = tracking_active
        self.tracking_active_pub.publish(status_msg)

        if tracking_active:
            # Average positions and yaw from available cameras
            fused_position = np.mean(positions, axis=0)
            fused_yaw = np.mean(yaws)

            # Apply fused filter
            fused_position = self._fused_position_filter.filter(fused_position)
            fused_yaw = self._fused_yaw_filter.filter(np.array([fused_yaw]))[0]

            self.publish_hand_pose(fused_position, stamp, fused_yaw)

            # Publish gripper from primary camera's landmarks
            if primary_cam is not None and primary_cam.latest_landmarks is not None:
                gripper = self.transformer.get_gripper_from_pinch(primary_cam.latest_landmarks)
                gripper_msg = Float32()
                gripper_msg.data = gripper
                self.gripper_cmd_pub.publish(gripper_msg)

            # Publish 3D landmarks from head camera only
            if (self._head.latest_landmarks is not None and depth_image is not None and
                self._head.latest_time is not None and
                (now - self._head.latest_time) < FUSION_STALENESS_S):
                self.publish_landmarks_3d(
                    self._head.latest_landmarks, depth_image, img_w, img_h, self._head
                )

            if not self._last_tracking_active:
                sources = [c.name for c in [self._head, self._front]
                          if c is not None and c.latest_position is not None
                          and c.latest_time is not None
                          and (now - c.latest_time) < FUSION_STALENESS_S]
                self.get_logger().info(f'Hand tracking started (depth, cameras: {sources})')

        else:
            if self._last_tracking_active:
                self.get_logger().info('Hand tracking lost')
                self.transformer.reset()
                self._fused_position_filter.reset()
                self._fused_yaw_filter.reset()
                self._head.position_filter.reset()
                self._head.yaw_filter.reset()
                if self._front is not None:
                    self._front.position_filter.reset()
                    self._front.yaw_filter.reset()

        self._last_tracking_active = tracking_active

    def image_callback(self, msg: Image):
        """
        Process incoming camera image with MediaPipe hand detection.
        Fallback path when use_depth is False.
        """
        try:
            # Convert raw image to RGB for MediaPipe
            rgb_image = self._bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')

            # Run MediaPipe hand detection
            results = self._hands_head.process(rgb_image)

            # Check if right hand detected
            landmarks = self._get_first_hand(results)
            tracking_active = landmarks is not None

            # Publish tracking status
            status_msg = Bool()
            status_msg.data = tracking_active
            self.tracking_active_pub.publish(status_msg)

            if tracking_active:

                # Compute hand yaw from pixel coordinates
                raw_yaw = self.get_hand_yaw_pixels(landmarks)
                yaw = self._fused_yaw_filter.filter(np.array([raw_yaw]))[0]

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
                    self.transformer.reset()
                    self._fused_yaw_filter.reset()

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
        yaw = yaw + self._hand_yaw_offset
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
        target = target + self._fine_offset

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
        self._hands_head.close()
        if self._use_front_camera:
            self._hands_front.close()
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
