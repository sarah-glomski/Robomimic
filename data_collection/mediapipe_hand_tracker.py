#!/usr/bin/env python3
"""
MediaPipe Hand Tracker Node with RealSense Depth

Subscribes to camera images and depth data, runs MediaPipe hand detection, and publishes:
- hand/pose: Hand position and orientation as PoseStamped (3D from depth)
- hand/gripper_cmd: Gripper command from pinch gesture (Float32, 0-1)
- hand/tracking_active: Whether hand is currently detected (Bool)
- hand/landmarks: All 21 hand landmarks in robot frame (Float32MultiArray, 63 floats)

Uses RealSense color + aligned depth for true 3D hand tracking.
"""

import sys
import os
# Add parent directory to path for imports when run standalone
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from collections import deque
import numpy as np
import cv2
import mediapipe as mp
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32, Bool, Float32MultiArray, MultiArrayDimension
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer

# Note: hand_to_action.py no longer used - 3D tracking done with depth directly


class MediaPipeHandTracker(Node):
    """
    ROS2 node that runs MediaPipe hand detection with RealSense depth for 3D tracking.
    """
    def __init__(self):
        super().__init__('mediapipe_hand_tracker')

        # Parameters
        self.declare_parameter('filter_alpha', 0.3)
        self.declare_parameter('orientation_filter_alpha', 0.12)
        self.declare_parameter('detection_confidence', 0.7)
        self.declare_parameter('tracking_confidence', 0.5)
        self.declare_parameter('depth_scale', 0.001)  # RealSense depth is in mm by default
        self.declare_parameter('max_depth', 1.5)  # Max valid depth in meters
        self.declare_parameter('min_depth', 0.1)  # Min valid depth in meters

        # Camera position relative to robot base (meters)
        # Default: camera 0.8m in front of robot, 0.5m high, centered
        self.declare_parameter('camera_x', 0.8)   # Forward from robot base
        self.declare_parameter('camera_y', -0.125)   # Left/right (0 = centered)
        self.declare_parameter('camera_z', 0.0)   # Height above robot base

        # Camera orientation: which way is the camera pointing?
        # 'towards_robot' = camera faces the robot (default for front-mounted)
        # 'away_from_robot' = camera faces away from robot
        # 'down' = camera looks straight down (top-down view)
        self.declare_parameter('camera_orientation', 'towards_robot')

        # Which hand to track: 'left' or 'right'
        self.declare_parameter('track_hand', 'right')

        filter_alpha = self.get_parameter('filter_alpha').value
        detection_conf = self.get_parameter('detection_confidence').value
        tracking_conf = self.get_parameter('tracking_confidence').value
        self._depth_scale = self.get_parameter('depth_scale').value
        self._max_depth = self.get_parameter('max_depth').value
        self._min_depth = self.get_parameter('min_depth').value

        # Camera extrinsics (position in robot frame)
        self._camera_position = np.array([
            self.get_parameter('camera_x').value,
            self.get_parameter('camera_y').value,
            self.get_parameter('camera_z').value,
        ])
        self._camera_orientation = self.get_parameter('camera_orientation').value
        self._track_hand = self.get_parameter('track_hand').value.lower()  # 'left' or 'right'

        # Build rotation matrix based on camera orientation
        self._build_camera_rotation()

        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # CV bridge for image conversion
        self._bridge = CvBridge()

        # Position filter for smoothing
        self._filter_alpha = filter_alpha
        self._orientation_filter_alpha = self.get_parameter('orientation_filter_alpha').value
        self._filtered_position = None
        self._position_window = deque(maxlen=5)  # Sliding window for median filtering

        # Camera intrinsics (will be set from CameraInfo)
        self._camera_info = None
        self._fx = None
        self._fy = None
        self._cx = None
        self._cy = None

        # QoS for camera streams - match RealSense QoS settings
        sensor_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        # Subscribe to camera info for intrinsics (uses VOLATILE durability)
        camera_info_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.VOLATILE
        )
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/rs_front/rs_front/aligned_depth_to_color/camera_info',
            self.camera_info_callback,
            camera_info_qos
        )

        # Synchronized subscribers for color and aligned depth
        self.color_sub = Subscriber(
            self, Image, '/rs_front/rs_front/color/image_raw', qos_profile=sensor_qos)
        self.depth_sub = Subscriber(
            self, Image, '/rs_front/rs_front/aligned_depth_to_color/image_raw', qos_profile=sensor_qos)

        # Approximate time synchronizer
        self.sync = ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub],
            queue_size=10,
            slop=0.05
        )
        self.sync.registerCallback(self.synced_callback)

        # Publishers
        self.hand_pose_pub = self.create_publisher(PoseStamped, 'hand/pose', 10)
        self.gripper_cmd_pub = self.create_publisher(Float32, 'hand/gripper_cmd', 10)
        self.tracking_active_pub = self.create_publisher(Bool, 'hand/tracking_active', 10)
        self.landmarks_pub = self.create_publisher(Float32MultiArray, 'hand/landmarks', 10)

        # State
        self._last_tracking_active = False
        self._prev_gripper = 0.0
        self._filtered_orientation = None  # Quaternion [x, y, z, w]

        self.get_logger().info('MediaPipe Hand Tracker initialized (with depth)')
        self.get_logger().info(f'Tracking: {self._track_hand.upper()} hand only')
        self.get_logger().info(f'Depth range: {self._min_depth}-{self._max_depth}m')
        self.get_logger().info(f'Camera position: {self._camera_position}')
        self.get_logger().info(f'Camera orientation: {self._camera_orientation}')

    def _build_camera_rotation(self):
        """
        Build the rotation matrix for camera-to-robot frame transformation.

        Camera optical frame convention (RealSense/OpenCV):
          X = right (from camera's view), Y = down, Z = forward (optical axis)

        Robot frame convention:
          X = forward, Y = left, Z = up

        The rotation matrix R transforms a point from camera frame to robot frame:
          point_robot = R @ point_camera
        Each row of R tells us which camera axis contributes to each robot axis.
        """
        if self._camera_orientation == 'towards_robot':
            # Camera faces towards robot (mounted in front of workspace, looking back at robot)
            # Camera is "upright" (RealSense logo on top)
            #
            # Camera Z (forward, into scene) points towards -X robot (towards robot)
            # Camera X (right from camera view) points towards +Y robot (robot's left, since we're facing it)
            # Camera Y (down) points towards -Z robot (down in robot frame too)
            #
            # R @ [cam_x, cam_y, cam_z]^T = [robot_x, robot_y, robot_z]^T
            # robot_x = -cam_z (depth towards robot is negative robot X)
            # robot_y = +cam_x (camera right is robot left)
            # robot_z = -cam_y (camera down is robot down)
            self._cam_to_robot_rotation = np.array([
                [0,  0, -1],   # Robot X = -Camera Z
                [1,  0,  0],   # Robot Y = +Camera X
                [0, -1,  0],   # Robot Z = -Camera Y
            ], dtype=np.float64)
        elif self._camera_orientation == 'away_from_robot':
            # Camera faces away from robot (mounted behind robot, looking forward)
            # Camera Z points towards +X robot
            # Camera X (right) points towards -Y robot
            # Camera Y (down) points towards -Z robot
            self._cam_to_robot_rotation = np.array([
                [0,  0,  1],   # Robot X = +Camera Z
                [-1, 0,  0],   # Robot Y = -Camera X
                [0, -1,  0],   # Robot Z = -Camera Y
            ], dtype=np.float64)
        elif self._camera_orientation == 'down':
            # Camera looks straight down (top-down view)
            # Assume camera X points towards +X robot (forward)
            # Camera Z points towards -Z robot (down)
            # Camera Y points towards +Y robot (left)
            self._cam_to_robot_rotation = np.array([
                [1,  0,  0],   # Robot X = +Camera X
                [0,  1,  0],   # Robot Y = +Camera Y
                [0,  0, -1],   # Robot Z = -Camera Z
            ], dtype=np.float64)
        else:
            self.get_logger().warn(f'Unknown camera orientation: {self._camera_orientation}, using towards_robot')
            self._cam_to_robot_rotation = np.array([
                [0,  0, -1],
                [1,  0,  0],
                [0, -1,  0],
            ], dtype=np.float64)

    def camera_info_callback(self, msg: CameraInfo):
        """Store camera intrinsics from CameraInfo message."""
        if self._camera_info is None:
            self._camera_info = msg
            # Extract intrinsics from K matrix [fx, 0, cx, 0, fy, cy, 0, 0, 1]
            self._fx = msg.k[0]
            self._fy = msg.k[4]
            self._cx = msg.k[2]
            self._cy = msg.k[5]
            self.get_logger().info(f'Camera intrinsics: fx={self._fx:.1f}, fy={self._fy:.1f}, cx={self._cx:.1f}, cy={self._cy:.1f}')

    def synced_callback(self, color_msg: Image, depth_msg: Image):
        """
        Process synchronized color and depth images with MediaPipe hand detection.
        """
        # Wait for camera intrinsics
        if self._camera_info is None:
            return

        try:
            # Convert ROS Images to OpenCV
            bgr_image = self._bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
            depth_image = self._bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

            # Convert to RGB for MediaPipe
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            img_h, img_w = rgb_image.shape[:2]

            # Run MediaPipe hand detection
            results = self.hands.process(rgb_image)

            # Check if hand detected and matches desired hand (left/right)
            tracking_active = False
            hand_index = None

            if results.multi_hand_landmarks and results.multi_handedness:
                # MediaPipe labels from camera's view (mirrored from user's perspective)
                # User's left hand = MediaPipe "Right", user's right = MediaPipe "Left"
                mp_label = 'right' if self._track_hand == 'left' else 'left'
                for idx, handedness in enumerate(results.multi_handedness):
                    label = handedness.classification[0].label.lower()
                    if label == mp_label:
                        tracking_active = True
                        hand_index = idx
                        break

            # Publish tracking status
            status_msg = Bool()
            status_msg.data = tracking_active
            self.tracking_active_pub.publish(status_msg)

            if tracking_active and hand_index is not None:
                # Get matching hand landmarks (image coords for position/depth)
                landmarks = results.multi_hand_landmarks[hand_index].landmark
                # World landmarks (metric 3D coords for orientation)
                world_landmarks = results.multi_hand_world_landmarks[hand_index].landmark

                # Get palm center in 3D using depth
                position_3d = self.get_palm_position_3d(landmarks, depth_image, img_w, img_h)

                if position_3d is not None:
                    # Apply low-pass filter
                    if self._filtered_position is None:
                        self._filtered_position = position_3d.copy()
                    else:
                        self._filtered_position = (self._filter_alpha * position_3d +
                                                   (1.0 - self._filter_alpha) * self._filtered_position)

                    # Get hand orientation from world landmarks (metric 3D)
                    orientation = self.get_hand_orientation(world_landmarks)

                    # Publish hand pose with orientation
                    self.publish_hand_pose(self._filtered_position, orientation, color_msg.header.stamp)

                    # Get gripper from pinch gesture
                    gripper = self.get_gripper_from_pinch(landmarks)
                    gripper_msg = Float32()
                    gripper_msg.data = gripper
                    self.gripper_cmd_pub.publish(gripper_msg)

                    # Publish all 21 landmarks in robot frame (with depth)
                    self.publish_landmarks_3d(landmarks, depth_image, img_w, img_h)

                    if not self._last_tracking_active:
                        self.get_logger().info('Hand tracking started')
                else:
                    # Invalid depth, treat as tracking lost
                    if self._last_tracking_active:
                        self.get_logger().info('Hand tracking lost (invalid depth)')
                    tracking_active = False

            else:
                if self._last_tracking_active:
                    self.get_logger().info('Hand tracking lost')
                    # Reset filters when tracking is lost
                    self._filtered_position = None
                    self._filtered_orientation = None
                    self._position_window.clear()

            self._last_tracking_active = tracking_active

        except Exception as e:
            self.get_logger().error(f'Error in synced callback: {e}')

    def get_palm_position_3d(self, landmarks, depth_image, img_w, img_h) -> np.ndarray:
        """
        Get palm center position in 3D camera frame using depth.

        Averages wrist + 4 MCP joints for a stable palm center estimate.
        A sliding window median filter rejects depth outliers before
        the EMA smoothing.

        Returns position in robot frame (meters) or None if depth invalid.
        """
        # Palm landmark indices: wrist + 4 MCP joints
        palm_indices = [0, 5, 9, 13, 17]

        pixel_coords = []
        depths = []

        for idx in palm_indices:
            lm = landmarks[idx]
            px = int(lm.x * img_w)
            py = int(lm.y * img_h)

            px = max(0, min(px, img_w - 1))
            py = max(0, min(py, img_h - 1))

            depth_val = self.sample_depth(depth_image, px, py, kernel_size=5)
            if depth_val is not None:
                pixel_coords.append((px, py))
                depths.append(depth_val)

        if len(depths) < 3:
            return None

        # Average the valid measurements
        avg_px = np.mean([p[0] for p in pixel_coords])
        avg_py = np.mean([p[1] for p in pixel_coords])
        avg_depth = np.mean(depths)

        # Convert to 3D camera coordinates using pinhole model
        # Camera frame: Z forward, X right, Y down
        cam_x = (avg_px - self._cx) * avg_depth / self._fx
        cam_y = (avg_py - self._cy) * avg_depth / self._fy
        cam_z = avg_depth

        # Point in camera frame
        point_cam = np.array([cam_x, cam_y, cam_z])

        # Transform to robot frame: rotate then translate
        point_robot = self._cam_to_robot_rotation @ point_cam + self._camera_position

        # Sliding window median filter to reject outlier frames
        self._position_window.append(point_robot)
        if len(self._position_window) >= 3:
            point_robot = np.median(self._position_window, axis=0)

        return point_robot

    def sample_depth(self, depth_image, px, py, kernel_size=3):
        """
        Sample depth at pixel location with a small kernel for robustness.
        Returns depth in meters or None if invalid.
        """
        h, w = depth_image.shape[:2]
        half_k = kernel_size // 2

        # Get region bounds
        x1 = max(0, px - half_k)
        x2 = min(w, px + half_k + 1)
        y1 = max(0, py - half_k)
        y2 = min(h, py + half_k + 1)

        region = depth_image[y1:y2, x1:x2]

        # Filter out zero/invalid depths
        valid_depths = region[region > 0]
        if len(valid_depths) == 0:
            return None

        # Use median for robustness
        depth_raw = np.median(valid_depths)
        depth_m = depth_raw * self._depth_scale

        # Check if within valid range
        if depth_m < self._min_depth or depth_m > self._max_depth:
            return None

        return depth_m

    def get_gripper_from_pinch(self, landmarks) -> float:
        """
        Get gripper command from pinch gesture (thumb to index finger distance).
        """
        thumb_tip = np.array([landmarks[4].x, landmarks[4].y, landmarks[4].z])
        index_tip = np.array([landmarks[8].x, landmarks[8].y, landmarks[8].z])

        distance = np.linalg.norm(thumb_tip - index_tip)

        # Thresholds for pinch detection (in normalized coordinates)
        pinch_close = 0.02  # 2cm - fully closed
        pinch_open = 0.10   # 10cm - fully open

        if distance < pinch_close:
            gripper = 1.0
        elif distance > pinch_open:
            gripper = 0.0
        else:
            gripper = 1.0 - (distance - pinch_close) / (pinch_open - pinch_close)

        # Simple smoothing
        gripper = 0.2 * gripper + 0.8 * self._prev_gripper
        self._prev_gripper = gripper

        return float(gripper)

    def get_hand_orientation(self, world_landmarks) -> np.ndarray:
        """
        Calculate hand orientation from MediaPipe's world landmarks.

        Computes two key directions from the hand:
        1. Palm normal (pointing out FRONT of palm, toward object to grasp)
           -> maps to tool Z axis (approach direction)
        2. Finger direction (wrist toward middle finger)
           -> maps to tool X axis

        Uses Rotation.align_vectors to find the rotation that maps
        the tool's identity frame axes to these observed hand directions.

        Returns quaternion [x, y, z, w] in robot frame, or None if invalid.
        """
        try:
            # Extract key landmarks in meters (MediaPipe camera frame)
            wrist = np.array([world_landmarks[0].x, world_landmarks[0].y, world_landmarks[0].z])
            index_mcp = np.array([world_landmarks[5].x, world_landmarks[5].y, world_landmarks[5].z])
            middle_mcp = np.array([world_landmarks[9].x, world_landmarks[9].y, world_landmarks[9].z])
            ring_mcp = np.array([world_landmarks[13].x, world_landmarks[13].y, world_landmarks[13].z])
            pinky_mcp = np.array([world_landmarks[17].x, world_landmarks[17].y, world_landmarks[17].z])

            # Vectors on the palm plane
            v_index = index_mcp - wrist
            v_pinky = pinky_mcp - wrist

            # Palm normal in camera frame
            palm_normal_cam = np.cross(v_index, v_pinky)
            norm = np.linalg.norm(palm_normal_cam)
            if norm < 1e-6:
                return None
            palm_normal_cam = palm_normal_cam / norm

            # The cross product direction depends on hand chirality due to the
            # spatial arrangement of index and pinky MCPs relative to the wrist:
            #   LEFT hand:  cross(v_index, v_pinky) -> DORSAL (back of hand)
            #   RIGHT hand: cross(v_index, v_pinky) -> PALMAR (front of palm)
            # We always want the palmar direction (front of palm = grasp approach).
            if self._track_hand == 'left':
                palm_normal_cam = -palm_normal_cam

            # Finger direction: average all 4 finger MCPs to reduce per-landmark
            # noise. A single MCP (e.g. middle) jitters by a few mm each frame;
            # averaging 4 points cuts that noise by ~half (sqrt(4)).
            avg_mcp = (index_mcp + middle_mcp + ring_mcp + pinky_mcp) / 4.0
            finger_dir_cam = avg_mcp - wrist
            finger_dir_cam = finger_dir_cam / (np.linalg.norm(finger_dir_cam) + 1e-8)

            # Transform both directions to robot frame
            palm_normal_robot = self._cam_to_robot_rotation @ palm_normal_cam
            finger_dir_robot = self._cam_to_robot_rotation @ finger_dir_cam

            # Use align_vectors to find rotation from tool identity frame to hand directions
            # At identity (roll=0, pitch=0, yaw=0): tool Z = [0,0,1], tool X = [1,0,0]
            # We want: tool Z -> palm_normal (approach direction)
            #          tool X -> finger_dir (reference direction)
            # Weight the palm normal 3x higher: it's geometrically stable (cross
            # product of two long vectors) while finger direction is noisier and
            # controls only the roll component.
            target_vectors = np.array([palm_normal_robot, finger_dir_robot])
            identity_axes = np.array([[0, 0, 1], [1, 0, 0]])  # Z and X of identity tool
            weights = np.array([3.0, 1.0])

            rotation, _ = R.align_vectors(target_vectors, identity_axes, weights=weights)

            # Apply yaw offset around local tool Z axis to align gripper Y
            # with the thumb-to-index finger closing direction
            offset_deg = -45 if self._track_hand == 'right' else 45
            yaw_offset = R.from_euler('z', offset_deg, degrees=True)
            rotation = rotation * yaw_offset

            quat = rotation.as_quat()  # [x, y, z, w]

            # Smoothing with quaternion continuity
            if self._filtered_orientation is None:
                self._filtered_orientation = quat
            else:
                if np.dot(self._filtered_orientation, quat) < 0:
                    quat = -quat
                alpha = self._orientation_filter_alpha
                self._filtered_orientation = alpha * quat + (1 - alpha) * self._filtered_orientation
                self._filtered_orientation = self._filtered_orientation / np.linalg.norm(self._filtered_orientation)

            return self._filtered_orientation

        except Exception as e:
            self.get_logger().warn(f'Orientation calculation failed: {e}')
            return None

    def publish_hand_pose(self, position: np.ndarray, orientation: np.ndarray, stamp):
        """
        Publish hand pose as PoseStamped.

        Position is in robot frame (meters).
        Orientation is quaternion [x, y, z, w] in robot frame.
        """
        msg = PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = "base_link"

        msg.pose.position.x = float(position[0])
        msg.pose.position.y = float(position[1])
        msg.pose.position.z = float(position[2])

        if orientation is not None:
            msg.pose.orientation.x = float(orientation[0])
            msg.pose.orientation.y = float(orientation[1])
            msg.pose.orientation.z = float(orientation[2])
            msg.pose.orientation.w = float(orientation[3])
        else:
            # Identity quaternion as fallback
            msg.pose.orientation.x = 0.0
            msg.pose.orientation.y = 0.0
            msg.pose.orientation.z = 0.0
            msg.pose.orientation.w = 1.0

        self.hand_pose_pub.publish(msg)

    def publish_landmarks_3d(self, landmarks, depth_image, img_w, img_h):
        """
        Publish all 21 hand landmarks in 3D robot frame using depth.

        Output format: Float32MultiArray with 63 floats (21 landmarks × 3 coords)
        """
        robot_landmarks = []

        for lm in landmarks:
            px = int(lm.x * img_w)
            py = int(lm.y * img_h)
            px = max(0, min(px, img_w - 1))
            py = max(0, min(py, img_h - 1))

            depth_m = self.sample_depth(depth_image, px, py)

            if depth_m is not None:
                # Convert to 3D camera coordinates
                cam_x = (px - self._cx) * depth_m / self._fx
                cam_y = (py - self._cy) * depth_m / self._fy
                cam_z = depth_m

                # Transform to robot frame
                point_cam = np.array([cam_x, cam_y, cam_z])
                point_robot = self._cam_to_robot_rotation @ point_cam + self._camera_position
                robot_x, robot_y, robot_z = point_robot
            else:
                # Fallback: use zero (will be filtered out by visualization)
                robot_x = robot_y = robot_z = 0.0

            robot_landmarks.extend([float(robot_x), float(robot_y), float(robot_z)])

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
