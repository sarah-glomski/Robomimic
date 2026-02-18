#!/usr/bin/env python3
"""
HDF5 Data Collector Node

Collects time-synchronized data from multiple sources and saves to HDF5:
- Robot actions (target pose, gripper command)
- Robot observations (current pose, gripper state)
- Hand tracking data
- Camera images (3 RealSense cameras: front, wrist, head)

Uses pygame for keyboard control:
- r: Reset robot to home position
- s: Start recording episode
- d: Done/end recording and save episode
- p: Pause recording and robot motion
- u: Unpause/resume

Runs pygame in main thread, ROS2 in background thread.
"""

import os
import threading
import numpy as np
import h5py
import cv2
import pygame

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import Bool, Float32
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from message_filters import Subscriber, ApproximateTimeSynchronizer
from cv_bridge import CvBridge


class HDF5DataCollector(Node):
    """
    ROS2 node for synchronized data collection with pygame keyboard control.
    """
    def __init__(self):
        super().__init__('hdf5_data_collector')

        self.get_logger().info('Initializing HDF5 Data Collector...')

        # Latency offset metadata (mirrors controller's parameter)
        self.declare_parameter('latency_offset', 0.0)
        self._latency_offset = self.get_parameter('latency_offset').value

        self._bridge = CvBridge()

        # QoS for camera streams
        sensor_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST
        )

        # Subscribers using message_filters for synchronization
        # Actions (commands to robot)
        self.action_pose_sub = Subscriber(
            self, PoseStamped, 'robot_action/pose', qos_profile=sensor_qos)
        self.action_gripper_sub = Subscriber(
            self, Float32, 'robot_action/gripper', qos_profile=sensor_qos)

        # Observations (state from robot)
        self.obs_pose_sub = Subscriber(
            self, PoseStamped, 'robot_obs/pose', qos_profile=sensor_qos)
        self.obs_gripper_sub = Subscriber(
            self, Float32, 'robot_obs/gripper', qos_profile=sensor_qos)

        # Hand tracking
        self.hand_pose_sub = Subscriber(
            self, PoseStamped, 'hand/pose', qos_profile=sensor_qos)

        # Camera images (raw Image for reliability with enable_sync mode)
        self.rs_front_sub = Subscriber(
            self, Image, '/rs_front/rs_front/color/image_raw',
            qos_profile=sensor_qos)
        self.rs_wrist_sub = Subscriber(
            self, Image, '/rs_wrist/rs_wrist/color/image_raw',
            qos_profile=sensor_qos)
        self.rs_head_sub = Subscriber(
            self, Image, '/rs_head/rs_head/color/image_raw',
            qos_profile=sensor_qos)

        # Approximate time synchronizer for all streams
        self.sync = ApproximateTimeSynchronizer(
            [
                self.action_pose_sub,
                self.action_gripper_sub,
                self.obs_pose_sub,
                self.obs_gripper_sub,
                self.hand_pose_sub,
                self.rs_front_sub,
                self.rs_wrist_sub,
                self.rs_head_sub,
            ],
            queue_size=100,
            slop=0.1,
            allow_headerless=True
        )
        self.sync.registerCallback(self.synced_callback)

        # Publishers for control signals
        self.reset_pub = self.create_publisher(Bool, '/reset_xarm', 10)
        self.pause_pub = self.create_publisher(Bool, '/pause_xarm', 10)

        # Collection state
        self.is_collecting = False
        self.is_paused = False
        self.demo_count = 0
        self.episode_start = None
        self.reset_buffers()

        # Thread lock for buffer access
        self._lock = threading.Lock()

        self.get_logger().info('HDF5 Data Collector initialized')

    def reset_buffers(self):
        """Clear all data buffers."""
        self.action_pose_buf = []
        self.action_gripper_buf = []
        self.obs_pose_buf = []
        self.obs_gripper_buf = []
        self.hand_pose_buf = []
        self.rs_front_buf = []
        self.rs_wrist_buf = []
        self.rs_head_buf = []

    def synced_callback(
        self,
        action_pose_msg: PoseStamped,
        action_gripper_msg: Float32,
        obs_pose_msg: PoseStamped,
        obs_gripper_msg: Float32,
        hand_pose_msg: PoseStamped,
        rs_front_msg: Image,
        rs_wrist_msg: Image,
        rs_head_msg: Image
    ):
        """
        Synchronized callback for all data streams.
        Only collects data when is_collecting is True and not paused.
        """
        if not self.is_collecting or self.is_paused:
            return

        with self._lock:
            # Action pose (target)
            p = action_pose_msg.pose.position
            o = action_pose_msg.pose.orientation
            self.action_pose_buf.append([p.x, p.y, p.z, o.x, o.y, o.z, o.w])

            # Action gripper
            self.action_gripper_buf.append(action_gripper_msg.data)

            # Observation pose (current)
            p = obs_pose_msg.pose.position
            o = obs_pose_msg.pose.orientation
            self.obs_pose_buf.append([p.x, p.y, p.z, o.x, o.y, o.z, o.w])

            # Observation gripper
            self.obs_gripper_buf.append(obs_gripper_msg.data)

            # Hand pose
            p = hand_pose_msg.pose.position
            o = hand_pose_msg.pose.orientation
            self.hand_pose_buf.append([p.x, p.y, p.z, o.x, o.y, o.z, o.w])

            # Camera images (CHW format)
            self.rs_front_buf.append(self.parse_color_image(rs_front_msg))
            self.rs_wrist_buf.append(self.parse_color_image(rs_wrist_msg))
            self.rs_head_buf.append(self.parse_color_image(rs_head_msg))

        frame_count = len(self.action_pose_buf)
        if frame_count % 30 == 0:
            self.get_logger().info(f'Collected {frame_count} frames')

    def parse_color_image(self, msg: Image) -> np.ndarray:
        """Convert raw image to CHW RGB numpy array."""
        rgb = self._bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        return rgb.transpose(2, 0, 1)  # (3, H, W)

    def start_collection(self):
        """Start collecting data for a new episode."""
        if not self.is_collecting:
            with self._lock:
                self.reset_buffers()
            self.is_collecting = True
            self.is_paused = False
            self.episode_start = self.get_clock().now()
            self.get_logger().info(f'Started recording episode {self.demo_count}')

    def end_collection(self):
        """End collection and save episode to HDF5."""
        if self.is_collecting:
            self.is_collecting = False
            self.save_episode()

            # Log stats
            dur = (self.get_clock().now() - self.episode_start).nanoseconds / 1e9
            n = len(self.action_pose_buf)
            hz = n / dur if dur > 0 else 0
            self.get_logger().info(f'Episode {self.demo_count} | {n} frames | {dur:.1f}s | {hz:.1f} Hz')

            self.demo_count += 1

    def pause_collection(self):
        """Pause data collection and robot motion."""
        if self.is_collecting and not self.is_paused:
            self.is_paused = True
            self.pause_pub.publish(Bool(data=True))
            self.get_logger().info('Paused recording')

    def unpause_collection(self):
        """Resume data collection and robot motion."""
        if self.is_collecting and self.is_paused:
            self.is_paused = False
            self.pause_pub.publish(Bool(data=False))
            self.get_logger().info('Resumed recording')

    def reset_robot(self):
        """Send reset command to robot."""
        self.get_logger().info('Sending reset command')
        self.reset_pub.publish(Bool(data=True))

    def save_episode(self):
        """Save collected data to HDF5 file."""
        with self._lock:
            if len(self.action_pose_buf) == 0:
                self.get_logger().warn('No data to save')
                return

            # Convert buffers to numpy arrays
            action_pose = np.array(self.action_pose_buf, dtype=np.float32)
            action_gripper = np.array(self.action_gripper_buf, dtype=np.float32)
            obs_pose = np.array(self.obs_pose_buf, dtype=np.float32)
            obs_gripper = np.array(self.obs_gripper_buf, dtype=np.float32)
            hand_pose = np.array(self.hand_pose_buf, dtype=np.float32)
            rs_front = np.array(self.rs_front_buf, dtype=np.uint8)
            rs_wrist = np.array(self.rs_wrist_buf, dtype=np.uint8)
            rs_head = np.array(self.rs_head_buf, dtype=np.uint8)

        # Create save directory
        save_dir = os.path.join(os.getcwd(), 'demo_data')
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, f'episode_{self.demo_count}.hdf5')

        # Save to HDF5
        with h5py.File(filename, 'w') as f:
            # Actions (commands to robot)
            action_grp = f.create_group('action')
            action_grp.create_dataset('pose', data=action_pose)
            action_grp.create_dataset('gripper', data=action_gripper)

            # Observations (state from robot)
            obs_grp = f.create_group('observation')
            obs_grp.create_dataset('pose', data=obs_pose)
            obs_grp.create_dataset('gripper', data=obs_gripper)

            # Hand tracking
            hand_grp = f.create_group('hand')
            hand_grp.create_dataset('pose', data=hand_pose)

            # Images with LZF compression
            images_grp = f.create_group('images')
            images_grp.create_dataset('rs_front', data=rs_front, compression='lzf')
            images_grp.create_dataset('rs_wrist', data=rs_wrist, compression='lzf')
            images_grp.create_dataset('rs_head', data=rs_head, compression='lzf')

            # Metadata
            f.attrs['num_frames'] = len(action_pose)
            f.attrs['collection_rate_hz'] = 30
            f.attrs['episode_index'] = self.demo_count
            f.attrs['latency_offset_sec'] = self._latency_offset

        self.get_logger().info(f'Saved episode to {filename}')


def run_pygame(node: HDF5DataCollector):
    """
    Main pygame loop for keyboard control.
    Runs in main thread.
    """
    pygame.init()
    screen = pygame.display.set_mode((500, 250))
    pygame.display.set_caption('Data Collection Control')
    font = pygame.font.Font(None, 32)
    small_font = pygame.font.Font(None, 24)

    clock = pygame.time.Clock()
    running = True

    while running and rclpy.ok():
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    node.reset_robot()
                elif event.key == pygame.K_s:
                    node.start_collection()
                elif event.key == pygame.K_d:
                    node.end_collection()
                elif event.key == pygame.K_p:
                    node.pause_collection()
                elif event.key == pygame.K_u:
                    node.unpause_collection()
                elif event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    running = False

        # Draw UI
        screen.fill((40, 44, 52))

        # Status
        if not node.is_collecting:
            status = "IDLE"
            status_color = (150, 150, 150)
        elif node.is_paused:
            status = "PAUSED"
            status_color = (255, 200, 50)
        else:
            status = "RECORDING"
            status_color = (80, 200, 80)

        # Title
        title = font.render(f'Status: {status}', True, status_color)
        screen.blit(title, (20, 20))

        # Frame count
        frame_count = len(node.action_pose_buf) if node.is_collecting else 0
        frames = font.render(f'Frames: {frame_count}', True, (200, 200, 200))
        screen.blit(frames, (20, 55))

        # Episode count
        episodes = font.render(f'Episodes saved: {node.demo_count}', True, (200, 200, 200))
        screen.blit(episodes, (20, 90))

        # Controls help
        controls = [
            "Controls:",
            "  R - Reset robot to home",
            "  S - Start recording",
            "  D - Done/end recording",
            "  P - Pause",
            "  U - Unpause",
            "  Q - Quit"
        ]
        y = 135
        for line in controls:
            text = small_font.render(line, True, (120, 130, 140))
            screen.blit(text, (20, y))
            y += 20

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


def main(args=None):
    rclpy.init(args=args)

    node = HDF5DataCollector()

    # Run ROS2 in background thread
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    ros_thread = threading.Thread(target=executor.spin, daemon=True)
    ros_thread.start()

    try:
        # Run pygame in main thread
        run_pygame(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info('Shutting down...')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
