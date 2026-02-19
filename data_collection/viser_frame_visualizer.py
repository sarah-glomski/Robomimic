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

import os
import json
import collections
import threading
import numpy as np
import viser

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray, Bool


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
        self.declare_parameter('workspace_x_max', 0.7)
        self.declare_parameter('workspace_y_min', -0.3)
        self.declare_parameter('workspace_y_max', 0.3)
        self.declare_parameter('workspace_z_min', 0.05)
        self.declare_parameter('workspace_z_max', 0.4)
        self.declare_parameter('latency_offset', 0.0)  # For sim mode internal delay
        self.declare_parameter('human_object_x', 0.3)
        self.declare_parameter('human_object_y', -0.35)
        self.declare_parameter('human_object_z', 0.0)
        self.declare_parameter('robot_object_x', 0.3)
        self.declare_parameter('robot_object_y', 0.0)
        self.declare_parameter('robot_object_z', 0.0)

        viser_port = self.get_parameter('viser_port').value
        self._latency_offset = self.get_parameter('latency_offset').value
        self._human_object_pos = (
            self.get_parameter('human_object_x').value,
            self.get_parameter('human_object_y').value,
            self.get_parameter('human_object_z').value,
        )
        self._robot_object_pos = (
            self.get_parameter('robot_object_x').value,
            self.get_parameter('robot_object_y').value,
            self.get_parameter('robot_object_z').value,
        )
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
        self.hand_pose_sub = self.create_subscription(
            PoseStamped, 'hand/pose', self.hand_pose_callback, sensor_qos)
        self.landmarks_sub = self.create_subscription(
            Float32MultiArray, 'hand/landmarks', self.landmarks_callback, sensor_qos)
        self.tracking_sub = self.create_subscription(
            Bool, 'hand/tracking_active', self.tracking_callback, 10)

        # Subscribe to eef/pose_target for delayed target in sim mode
        self.eef_target_sub = self.create_subscription(
            PoseStamped, 'eef/pose_target', self.eef_target_callback, sensor_qos)

        # Subscribe to robot_action/pose (published by controller with latency applied)
        self.action_pose_sub = self.create_subscription(
            PoseStamped, 'robot_action/pose', self.action_pose_callback, sensor_qos)

        # State
        self._hand_tracking_active = False
        self._landmarks = None

        # Internal delay buffer for sim mode (when no controller is running)
        self._delay_buffer = collections.deque(maxlen=300)
        self._has_action_pose = False  # True once we receive robot_action/pose
        if self._latency_offset > 0.0:
            self._delay_timer = self.create_timer(1.0 / 30.0, self._drain_delay_buffer)
            self.get_logger().info(f'Sim mode delay: {self._latency_offset:.2f}s')

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

        # Delayed target frame (orange) - shows where robot is commanded to go
        self.delayed_target_frame = self.server.scene.add_frame(
            "/delayed_target",
            wxyz=(1.0, 0.0, 0.0, 0.0),
            position=(0.3, 0.0, 0.2),
            axes_length=0.06,
            axes_radius=0.004,
            origin_color=(255, 165, 0),
            visible=False,
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

        # Camera frustums
        self.add_camera_frustums()

        # Object markers (coffee mugs)
        self.add_object_markers()

        # Initialize hand landmarks point cloud (will be updated)
        self._hand_points_handle = None
        self._hand_skeleton_handle = None

    def _wireframe_box(self, name: str, bounds: dict, color: tuple, line_width: float = 2.0):
        """Add a wireframe box to the scene."""
        x_min, x_max = bounds['x_min'], bounds['x_max']
        y_min, y_max = bounds['y_min'], bounds['y_max']
        z_min, z_max = bounds['z_min'], bounds['z_max']

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

        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]

        points = np.array([[corners[i], corners[j]] for i, j in edges])
        colors = np.array([[color] * 2 for _ in edges])

        self.server.scene.add_line_segments(
            name,
            points=points.astype(np.float32),
            colors=colors.astype(np.uint8),
            line_width=line_width,
        )

    def add_workspace_bounds(self):
        """Add wireframe boxes for robot and human workspace bounds."""
        # Robot workspace (cyan)
        self._wireframe_box("/workspace_bounds", self.workspace_bounds, (100, 200, 200))

        # Human workspace: mirror robot bounds relative to human object position
        # robot_target = robot_obj + (hand - human_obj)
        # So hand = human_obj + (robot_target - robot_obj)
        # human bounds = human_obj + (robot_bounds - robot_obj)
        b = self.workspace_bounds
        ro = self._robot_object_pos
        ho = self._human_object_pos
        human_bounds = {
            'x_min': ho[0] + (b['x_min'] - ro[0]),
            'x_max': ho[0] + (b['x_max'] - ro[0]),
            'y_min': ho[1] + (b['y_min'] - ro[1]),
            'y_max': ho[1] + (b['y_max'] - ro[1]),
            'z_min': ho[2] + (b['z_min'] - ro[2]),
            'z_max': ho[2] + (b['z_max'] - ro[2]),
        }
        self._wireframe_box("/human_workspace_bounds", human_bounds, (160, 80, 200))
        self.server.scene.add_label("/human_workspace_bounds/label", text="Human Workspace")

    def _load_camera_poses(self) -> dict:
        """Load camera poses from camera_calibration.json, falling back to hardcoded."""
        defaults = {
            "head": {
                "position": (0.28, 0.0, 1.02),
                "wxyz": (0.0, 0.7071, 0.7071, 0.0),
            },
            "front": {
                "position": (1.125, 0.0, 0.7),
                "wxyz": (0.2706, -0.6533, -0.6533, 0.2706),
            },
        }

        calib_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'camera_calibration.json'
        )

        if not os.path.exists(calib_path):
            self.get_logger().info('No camera_calibration.json — using default camera poses')
            return defaults

        try:
            with open(calib_path, 'r') as f:
                calib = json.load(f)

            result = dict(defaults)
            for cam_name in ["head", "front"]:
                entry = calib.get(f"{cam_name}_camera")
                if entry is not None:
                    pos = entry.get("camera_position_robot_frame")
                    quat = entry.get("camera_quaternion_wxyz")
                    if pos is not None and quat is not None:
                        result[cam_name] = {
                            "position": tuple(pos),
                            "wxyz": tuple(quat),
                        }
                        self.get_logger().info(f'Loaded {cam_name} camera pose from calibration')

            return result

        except Exception as e:
            self.get_logger().warn(f'Failed to load calibration: {e} — using defaults')
            return defaults

    def add_camera_frustums(self):
        """Add camera frustum visualizations for head and front cameras."""
        cam_fov = 0.733   # ~42 deg vertical FOV (RealSense D435)
        cam_aspect = 640.0 / 360.0

        poses = self._load_camera_poses()

        self.server.scene.add_camera_frustum(
            "/cameras/head",
            fov=cam_fov,
            aspect=cam_aspect,
            scale=0.08,
            line_width=2.0,
            color=(100, 180, 255),
            wxyz=poses["head"]["wxyz"],
            position=poses["head"]["position"],
        )
        self.server.scene.add_label(
            "/cameras/head/label",
            text="Head Cam",
        )

        self.server.scene.add_camera_frustum(
            "/cameras/front",
            fov=cam_fov,
            aspect=cam_aspect,
            scale=0.08,
            line_width=2.0,
            color=(255, 180, 100),
            wxyz=poses["front"]["wxyz"],
            position=poses["front"]["position"],
        )
        self.server.scene.add_label(
            "/cameras/front/label",
            text="Front Cam",
        )

    def add_object_markers(self):
        """Add cylinder markers for human and robot objects (coffee mugs)."""
        mug_radius = 0.03
        mug_height = 0.08

        # Human object (purple cylinder)
        # Cylinder is centered at origin; offset position by half height so bottom sits on table
        self.server.scene.add_mesh_trimesh(
            "/objects/human_mug",
            mesh=self._make_cylinder_mesh(mug_radius, mug_height, color=(160, 80, 200)),
            position=(
                self._human_object_pos[0],
                self._human_object_pos[1],
                self._human_object_pos[2] + mug_height / 2.0,
            ),
        )
        self.server.scene.add_label(
            "/objects/human_mug/label",
            text="Human Mug",
        )

        # Robot object (cyan cylinder)
        self.server.scene.add_mesh_trimesh(
            "/objects/robot_mug",
            mesh=self._make_cylinder_mesh(mug_radius, mug_height, color=(80, 200, 200)),
            position=(
                self._robot_object_pos[0],
                self._robot_object_pos[1],
                self._robot_object_pos[2] + mug_height / 2.0,
            ),
        )
        self.server.scene.add_label(
            "/objects/robot_mug/label",
            text="Robot Mug",
        )

    @staticmethod
    def _make_cylinder_mesh(radius: float, height: float, color: tuple):
        """Create a trimesh cylinder with the given dimensions and color."""
        import trimesh
        mesh = trimesh.creation.cylinder(radius=radius, height=height, sections=16)
        mesh.visual.face_colors = [color[0], color[1], color[2], 255]
        return mesh

    @staticmethod
    def _yaw_to_eef_wxyz(ori):
        """Compose yaw-only quaternion with roll=180° for EEF display (Z pointing down).

        The hand tracker publishes a pure Z-rotation quaternion (yaw only).
        This composes it with a 180° X-rotation so frames display with Z down.
        Quaternion multiply: q_display = q_yaw * q_roll180
        where q_roll180 wxyz = (0, 1, 0, 0).
        """
        return (-ori.x, ori.w, ori.z, -ori.y)

    def robot_pose_callback(self, msg: PoseStamped):
        """Update robot EEF frame position."""
        pos = msg.pose.position
        ori = msg.pose.orientation

        self.robot_eef_frame.position = (pos.x, pos.y, pos.z)
        self.robot_eef_frame.wxyz = (ori.w, ori.x, ori.y, ori.z)

    def hand_pose_callback(self, msg: PoseStamped):
        """Update hand palm frame position (actual hand, blue frame)."""
        pos = msg.pose.position
        ori = msg.pose.orientation

        self.hand_palm_frame.position = (pos.x, pos.y, pos.z)
        self.hand_palm_frame.wxyz = self._yaw_to_eef_wxyz(ori)

    def eef_target_callback(self, msg: PoseStamped):
        """Update delayed target frame from eef/pose_target (sim mode, orange frame).

        Only used when no controller is running (no robot_action/pose received).
        The latency delay buffer is applied here independently of the
        object-relative transform (which was already applied upstream).
        """
        if self._has_action_pose:
            return

        pos = msg.pose.position
        ori = msg.pose.orientation
        wxyz = self._yaw_to_eef_wxyz(ori)

        if self._latency_offset > 0.0:
            # Buffer for delayed display
            now = self.get_clock().now().nanoseconds / 1e9
            self._delay_buffer.append((now, (pos.x, pos.y, pos.z), wxyz))
        else:
            # Zero latency: target follows eef target directly
            self.delayed_target_frame.position = (pos.x, pos.y, pos.z)
            self.delayed_target_frame.wxyz = wxyz
            self.delayed_target_frame.visible = self._hand_tracking_active

    def action_pose_callback(self, msg: PoseStamped):
        """Update delayed target frame from controller's robot_action/pose."""
        self._has_action_pose = True
        pos = msg.pose.position
        ori = msg.pose.orientation
        self.delayed_target_frame.position = (pos.x, pos.y, pos.z)
        # Controller quaternion already has full orientation (roll=180° + yaw)
        self.delayed_target_frame.wxyz = (ori.w, ori.x, ori.y, ori.z)
        self.delayed_target_frame.visible = self._hand_tracking_active

    def _drain_delay_buffer(self):
        """Drain sim-mode internal delay buffer for delayed target visualization."""
        if self._has_action_pose or len(self._delay_buffer) == 0:
            return

        now = self.get_clock().now().nanoseconds / 1e9
        cutoff = now - self._latency_offset

        last_entry = None
        while len(self._delay_buffer) > 0 and self._delay_buffer[0][0] <= cutoff:
            _, position, wxyz = self._delay_buffer.popleft()
            last_entry = (position, wxyz)

        if last_entry is not None:
            self.delayed_target_frame.position = last_entry[0]
            self.delayed_target_frame.wxyz = last_entry[1]
            self.delayed_target_frame.visible = self._hand_tracking_active

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
            # Hide delayed target and clear buffer
            self.delayed_target_frame.visible = False
            self._delay_buffer.clear()

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
