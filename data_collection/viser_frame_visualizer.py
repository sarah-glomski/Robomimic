#!/usr/bin/env python3
"""
Viser 3D Frame Visualizer Node

Web-based 3D visualization using viser library.
Displays:
- Robot base frame (origin)
- Robot EEF frame (end-effector)
- All 21 MediaPipe hand landmarks
- Hand skeleton connections
- Palm center frame
- Workspace bounds wireframe

Access visualization at: http://localhost:8080
"""

import threading
import numpy as np
import viser

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray, Bool, Float32


# MediaPipe hand landmark names
LANDMARK_NAMES = [
    'wrist',
    'thumb_cmc', 'thumb_mcp', 'thumb_ip', 'thumb_tip',
    'index_mcp', 'index_pip', 'index_dip', 'index_tip',
    'middle_mcp', 'middle_pip', 'middle_dip', 'middle_tip',
    'ring_mcp', 'ring_pip', 'ring_dip', 'ring_tip',
    'pinky_mcp', 'pinky_pip', 'pinky_dip', 'pinky_tip',
]

# Hand skeleton connections (pairs of landmark indices)
HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring finger
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky finger
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm connections
    (5, 9), (9, 13), (13, 17),
]


class ViserFrameVisualizer(Node):
    """
    ROS2 node that visualizes coordinate frames using viser.
    """
    def __init__(self):
        super().__init__('viser_frame_visualizer')

        # Parameters
        self.declare_parameter('viser_port', 8080)
        self.declare_parameter('workspace_x_min', 0.1)
        self.declare_parameter('workspace_x_max', 0.5)
        self.declare_parameter('workspace_y_min', -0.3)
        self.declare_parameter('workspace_y_max', 0.3)
        self.declare_parameter('workspace_z_min', 0.05)
        self.declare_parameter('workspace_z_max', 0.4)

        # Object frame parameters
        self.declare_parameter('human_object_x', 0.30)
        self.declare_parameter('human_object_y', 0.00)
        self.declare_parameter('human_object_z', 0.10)
        self.declare_parameter('robot_object_x', 0.35)
        self.declare_parameter('robot_object_y', -0.15)
        self.declare_parameter('robot_object_z', 0.10)

        # Camera position/orientation parameters
        self.declare_parameter('camera_x', 0.8)
        self.declare_parameter('camera_y', -0.125)
        self.declare_parameter('camera_z', 0.0)
        self.declare_parameter('camera_orientation', 'towards_robot')

        self.human_object_pos = (
            self.get_parameter('human_object_x').value,
            self.get_parameter('human_object_y').value,
            self.get_parameter('human_object_z').value,
        )
        self.robot_object_pos = (
            self.get_parameter('robot_object_x').value,
            self.get_parameter('robot_object_y').value,
            self.get_parameter('robot_object_z').value,
        )
        self.camera_pos = (
            self.get_parameter('camera_x').value,
            self.get_parameter('camera_y').value,
            self.get_parameter('camera_z').value,
        )
        self.camera_orientation = self.get_parameter('camera_orientation').value

        viser_port = self.get_parameter('viser_port').value
        self.workspace_bounds = {
            'x_min': self.get_parameter('workspace_x_min').value,
            'x_max': self.get_parameter('workspace_x_max').value,
            'y_min': self.get_parameter('workspace_y_min').value,
            'y_max': self.get_parameter('workspace_y_max').value,
            'z_min': self.get_parameter('workspace_z_min').value,
            'z_max': self.get_parameter('workspace_z_max').value,
        }

        # Initialize viser server
        self.get_logger().info(f'Starting viser server on port {viser_port}...')
        self.server = viser.ViserServer(port=viser_port)

        # Setup static scene elements
        self.setup_scene()

        # QoS for subscriptions
        sensor_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST
        )

        # Subscribers
        self.robot_pose_sub = self.create_subscription(
            PoseStamped, 'robot_obs/pose', self.robot_pose_callback, sensor_qos)
        self.robot_action_sub = self.create_subscription(
            PoseStamped, 'robot_action/pose', self.robot_action_callback, sensor_qos)
        self.robot_gripper_sub = self.create_subscription(
            Float32, 'robot_obs/gripper', self.gripper_callback, sensor_qos)
        self.hand_pose_sub = self.create_subscription(
            PoseStamped, 'hand/pose', self.hand_pose_callback, sensor_qos)
        self.landmarks_sub = self.create_subscription(
            Float32MultiArray, 'hand/landmarks', self.landmarks_callback, sensor_qos)
        self.tracking_sub = self.create_subscription(
            Bool, 'hand/tracking_active', self.tracking_callback, 10)

        # State
        self._hand_tracking_active = False
        self._landmarks = None
        self._robot_position = np.array([0.2, 0.0, 0.2])
        self._gripper_value = 0.0

        self.get_logger().info(f'Viser Frame Visualizer initialized')
        self.get_logger().info(f'Open browser to: http://localhost:{viser_port}')

    def setup_scene(self):
        """Setup static scene elements: robot base, workspace bounds."""
        # Robot base frame (yellow, at origin)
        self.robot_base_frame = self.server.scene.add_frame(
            "/robot_base",
            wxyz=(1.0, 0.0, 0.0, 0.0),
            position=(0.0, 0.0, 0.0),
            axes_length=0.1,
            axes_radius=0.005,
        )

        # Robot EEF frame (green)
        self.robot_eef_frame = self.server.scene.add_frame(
            "/robot_eef",
            wxyz=(1.0, 0.0, 0.0, 0.0),
            position=(0.2, 0.0, 0.2),
            axes_length=0.08,
            axes_radius=0.004,
            origin_color=(50, 200, 50),
        )

        # Hand palm frame (blue)
        self.hand_palm_frame = self.server.scene.add_frame(
            "/hand/palm",
            wxyz=(1.0, 0.0, 0.0, 0.0),
            position=(0.3, 0.0, 0.2),
            axes_length=0.05,
            axes_radius=0.003,
            origin_color=(50, 100, 255),
            visible=False,
        )

        # Robot target position marker (transparent red sphere)
        self.robot_target_marker = self.server.scene.add_icosphere(
            "/robot_target",
            radius=0.015,
            color=(255, 100, 100),
            position=(0.2, 0.0, 0.2),
        )

        # Simple robot arm visualization (line from base to EEF)
        self._arm_line_handle = self.server.scene.add_line_segments(
            "/robot_arm",
            points=np.array([[[0.0, 0.0, 0.0], [0.2, 0.0, 0.2]]], dtype=np.float32),
            colors=np.array([[[100, 180, 100], [100, 180, 100]]], dtype=np.uint8),
            line_width=4.0,
        )

        # Gripper visualization (two small boxes)
        self._gripper_left = self.server.scene.add_box(
            "/robot_gripper_left",
            dimensions=(0.01, 0.01, 0.03),
            color=(80, 80, 80),
            position=(0.2, 0.015, 0.2),
        )
        self._gripper_right = self.server.scene.add_box(
            "/robot_gripper_right",
            dimensions=(0.01, 0.01, 0.03),
            color=(80, 80, 80),
            position=(0.2, -0.015, 0.2),
        )

        # Workspace bounds wireframe
        self.add_workspace_bounds()

        # Add grid on XY plane at z=0
        self.server.scene.add_grid(
            "/grid",
            width=1.0,
            height=1.0,
            width_segments=10,
            height_segments=10,
            plane="xy",
            position=(0.3, 0.0, 0.0),
        )

        # Human object reference frame (orange)
        self.human_object_frame_viz = self.server.scene.add_frame(
            "/human_object_frame",
            wxyz=(1.0, 0.0, 0.0, 0.0),
            position=self.human_object_pos,
            axes_length=0.06,
            axes_radius=0.003,
            origin_color=(255, 165, 0),
        )
        self.server.scene.add_box(
            "/human_object_box",
            dimensions=(0.04, 0.04, 0.04),
            color=(255, 165, 0),
            position=self.human_object_pos,
        )
        self.server.scene.add_label(
            "/human_object_label",
            text="Human Object",
            position=(self.human_object_pos[0], self.human_object_pos[1],
                      self.human_object_pos[2] + 0.05),
        )

        # Robot object reference frame (cyan)
        self.robot_object_frame_viz = self.server.scene.add_frame(
            "/robot_object_frame",
            wxyz=(1.0, 0.0, 0.0, 0.0),
            position=self.robot_object_pos,
            axes_length=0.06,
            axes_radius=0.003,
            origin_color=(0, 200, 200),
        )
        self.server.scene.add_box(
            "/robot_object_box",
            dimensions=(0.04, 0.04, 0.04),
            color=(0, 200, 200),
            position=self.robot_object_pos,
        )
        self.server.scene.add_label(
            "/robot_object_label",
            text="Robot Object",
            position=(self.robot_object_pos[0], self.robot_object_pos[1],
                      self.robot_object_pos[2] + 0.05),
        )

        # Initialize hand landmarks point cloud (will be updated)
        self._hand_points_handle = None
        self._hand_skeleton_handle = None

        # Camera frame visualization (purple)
        self.add_camera_frame()

    def add_camera_frame(self):
        """Add camera position and orientation visualization."""
        # Get camera orientation quaternion
        camera_wxyz = self._get_camera_quaternion()

        # Camera frame (purple)
        self.camera_frame = self.server.scene.add_frame(
            "/camera_frame",
            wxyz=camera_wxyz,
            position=self.camera_pos,
            axes_length=0.08,
            axes_radius=0.004,
            origin_color=(150, 50, 200),
        )

        # Camera body (small box to represent camera)
        self.server.scene.add_box(
            "/camera_body",
            dimensions=(0.04, 0.08, 0.03),
            color=(100, 100, 100),
            position=self.camera_pos,
            wxyz=camera_wxyz,
        )

        # Camera lens (small cylinder facing forward)
        # We'll use a small sphere to indicate the lens direction
        lens_offset = self._get_camera_forward_vector() * 0.03
        lens_pos = (
            self.camera_pos[0] + lens_offset[0],
            self.camera_pos[1] + lens_offset[1],
            self.camera_pos[2] + lens_offset[2],
        )
        self.server.scene.add_icosphere(
            "/camera_lens",
            radius=0.01,
            color=(50, 50, 50),
            position=lens_pos,
        )

        # Camera label
        self.server.scene.add_label(
            "/camera_label",
            text="Camera",
            position=(self.camera_pos[0], self.camera_pos[1], self.camera_pos[2] + 0.08),
        )

        # Camera view direction line (show where camera is pointing)
        view_length = 0.3
        view_end = self._get_camera_forward_vector() * view_length
        view_points = np.array([[
            [self.camera_pos[0], self.camera_pos[1], self.camera_pos[2]],
            [self.camera_pos[0] + view_end[0], self.camera_pos[1] + view_end[1], self.camera_pos[2] + view_end[2]]
        ]], dtype=np.float32)
        view_colors = np.array([[[150, 50, 200], [150, 50, 200]]], dtype=np.uint8)

        self.server.scene.add_line_segments(
            "/camera_view_direction",
            points=view_points,
            colors=view_colors,
            line_width=2.0,
        )

    def _get_camera_quaternion(self):
        """Get quaternion (wxyz) for camera orientation."""
        if self.camera_orientation == 'towards_robot':
            # Camera faces -X direction (towards robot at origin)
            # Rotate 180 degrees around Z axis
            return (0.0, 0.0, 0.0, 1.0)  # 180 deg around Z
        elif self.camera_orientation == 'away_from_robot':
            # Camera faces +X direction
            return (1.0, 0.0, 0.0, 0.0)  # Identity
        elif self.camera_orientation == 'down':
            # Camera faces -Z direction (looking down)
            # Rotate -90 degrees around Y axis
            return (0.707, 0.0, -0.707, 0.0)
        else:
            return (1.0, 0.0, 0.0, 0.0)

    def _get_camera_forward_vector(self):
        """Get unit vector pointing in camera's forward direction."""
        if self.camera_orientation == 'towards_robot':
            return np.array([-1.0, 0.0, 0.0])
        elif self.camera_orientation == 'away_from_robot':
            return np.array([1.0, 0.0, 0.0])
        elif self.camera_orientation == 'down':
            return np.array([0.0, 0.0, -1.0])
        else:
            return np.array([-1.0, 0.0, 0.0])

    def add_workspace_bounds(self):
        """Add wireframe box showing workspace safety bounds."""
        b = self.workspace_bounds
        x_min, x_max = b['x_min'], b['x_max']
        y_min, y_max = b['y_min'], b['y_max']
        z_min, z_max = b['z_min'], b['z_max']

        # Define the 8 corners of the box
        corners = np.array([
            [x_min, y_min, z_min],
            [x_max, y_min, z_min],
            [x_max, y_max, z_min],
            [x_min, y_max, z_min],
            [x_min, y_min, z_max],
            [x_max, y_min, z_max],
            [x_max, y_max, z_max],
            [x_min, y_max, z_max],
        ])

        # Define the 12 edges of the box (pairs of corner indices)
        edges = [
            # Bottom face
            (0, 1), (1, 2), (2, 3), (3, 0),
            # Top face
            (4, 5), (5, 6), (6, 7), (7, 4),
            # Vertical edges
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]

        # Create line segments for the wireframe
        points = np.array([[corners[i], corners[j]] for i, j in edges])
        colors = np.array([[(100, 200, 200)] * 2 for _ in edges])

        self.server.scene.add_line_segments(
            "/workspace_bounds",
            points=points.astype(np.float32),
            colors=colors.astype(np.uint8),
            line_width=2.0,
        )

    def robot_pose_callback(self, msg: PoseStamped):
        """Update robot EEF frame position."""
        pos = msg.pose.position
        ori = msg.pose.orientation

        self._robot_position = np.array([pos.x, pos.y, pos.z])

        self.robot_eef_frame.position = (pos.x, pos.y, pos.z)
        self.robot_eef_frame.wxyz = (ori.w, ori.x, ori.y, ori.z)

        # Update arm line from base to EEF
        arm_points = np.array([[[0.0, 0.0, 0.0], [pos.x, pos.y, pos.z]]], dtype=np.float32)
        self._arm_line_handle.points = arm_points

        # Update gripper positions (attached to EEF)
        gripper_offset = 0.015 + 0.015 * (1.0 - self._gripper_value)  # Open/close
        self._gripper_left.position = (pos.x, pos.y + gripper_offset, pos.z)
        self._gripper_right.position = (pos.x, pos.y - gripper_offset, pos.z)

    def robot_action_callback(self, msg: PoseStamped):
        """Update robot target marker (commanded position)."""
        pos = msg.pose.position
        self.robot_target_marker.position = (pos.x, pos.y, pos.z)

    def gripper_callback(self, msg: Float32):
        """Update gripper value."""
        self._gripper_value = msg.data

    def hand_pose_callback(self, msg: PoseStamped):
        """Update hand palm frame position."""
        pos = msg.pose.position
        ori = msg.pose.orientation

        self.hand_palm_frame.position = (pos.x, pos.y, pos.z)
        self.hand_palm_frame.wxyz = (ori.w, ori.x, ori.y, ori.z)

    def tracking_callback(self, msg: Bool):
        """Update hand tracking status."""
        self._hand_tracking_active = msg.data
        self.hand_palm_frame.visible = msg.data

        if not msg.data:
            # Hide hand visualization when tracking lost
            if self._hand_points_handle is not None:
                self._hand_points_handle.visible = False
            if self._hand_skeleton_handle is not None:
                self._hand_skeleton_handle.visible = False

    def landmarks_callback(self, msg: Float32MultiArray):
        """Update hand landmarks visualization."""
        if not self._hand_tracking_active:
            return

        # Parse landmarks from flat array (21 landmarks × 3 coords = 63 floats)
        data = np.array(msg.data).reshape(21, 3)
        self._landmarks = data

        # Update point cloud
        colors = np.full((21, 3), [255, 100, 80], dtype=np.uint8)

        if self._hand_points_handle is None:
            self._hand_points_handle = self.server.scene.add_point_cloud(
                "/hand/points",
                points=data.astype(np.float32),
                colors=colors,
                point_size=0.012,
                point_shape="circle",
            )
        else:
            self._hand_points_handle.points = data.astype(np.float32)
            self._hand_points_handle.colors = colors
            self._hand_points_handle.visible = True

        # Update skeleton lines
        skeleton_points = []
        for i, j in HAND_CONNECTIONS:
            skeleton_points.append([data[i], data[j]])

        skeleton_points = np.array(skeleton_points, dtype=np.float32)
        skeleton_colors = np.full((len(HAND_CONNECTIONS), 2, 3), [180, 180, 180], dtype=np.uint8)

        if self._hand_skeleton_handle is None:
            self._hand_skeleton_handle = self.server.scene.add_line_segments(
                "/hand/skeleton",
                points=skeleton_points,
                colors=skeleton_colors,
                line_width=2.0,
            )
        else:
            self._hand_skeleton_handle.points = skeleton_points
            self._hand_skeleton_handle.colors = skeleton_colors
            self._hand_skeleton_handle.visible = True

    def destroy_node(self):
        """Clean up viser server."""
        self.get_logger().info('Shutting down viser server...')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    node = ViserFrameVisualizer()

    try:
        node.get_logger().info('Running Viser Frame Visualizer...')
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
