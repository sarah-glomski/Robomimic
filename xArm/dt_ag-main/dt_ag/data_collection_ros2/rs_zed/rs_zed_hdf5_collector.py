#!/usr/bin/env python3

import os
import numpy as np
import h5py

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from std_msgs.msg import Bool, Float32
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, CompressedImage

from message_filters import Subscriber, ApproximateTimeSynchronizer
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import cv2  # for color conversion
from cv_bridge import CvBridge

class XArmDataCollection(Node):
    def __init__(self):
        super().__init__('xarm_data_collection_node')
        self.get_logger().info("Initializing data_collection_node with approximate sync.")

        sensor_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)

        # Subscribers for robot and cameras
        self.pose_sub = Subscriber(self, PoseStamped, 'robot_position_action', qos_profile=sensor_qos)
        self.gripper_sub = Subscriber(self, Float32, 'gripper_position', qos_profile=sensor_qos)
        self.rs_side_rgb_sub = Subscriber(self, CompressedImage, '/rs_side/rs_side/color/image_raw/compressed', qos_profile=sensor_qos)
        self.rs_wrist_rgb_sub = Subscriber(self, CompressedImage, '/rs_wrist/rs_wrist/color/image_raw/compressed', qos_profile=sensor_qos)
        self.rs_front_rgb_sub = Subscriber(self, CompressedImage, '/rs_front/rs_front/color/image_raw/compressed', qos_profile=sensor_qos)
        self.zed_rgb_sub = Subscriber(self, CompressedImage, 'zed_image/rgb/compressed', qos_profile=sensor_qos)
        self.zed_depth_sub = Subscriber(self, CompressedImage, 'zed_image/depth/compressed', qos_profile=sensor_qos)

        # ApproximateTimeSynchronizer
        self.sync = ApproximateTimeSynchronizer(
            [
                self.pose_sub,
                self.gripper_sub,
                self.rs_side_rgb_sub,
                self.rs_front_rgb_sub,
                self.rs_wrist_rgb_sub,
                self.zed_rgb_sub,
                self.zed_depth_sub
            ],
            queue_size=100,
            slop=0.1,
            allow_headerless=True
        )
        self.sync.registerCallback(self.synced_callback)

        # Start/End demo
        self.start_sub = self.create_subscription(Bool, 'start_demo', self.start_demo_callback, sensor_qos)
        self.end_sub = self.create_subscription(Bool, 'end_demo', self.end_demo_callback, sensor_qos)

        # State
        self.is_collecting = False
        self.demo_count = 0

        self.pose_data = []
        self.gripper_data = []
        self.rs_side_rgb_frames = []
        self.rs_wrist_rgb_frames = []
        self.rs_front_rgb_frames = []
        self.zed_rgb_frames = []
        self.zed_depth_frames = []

        self._bridge = CvBridge()

    def start_demo_callback(self, msg: Bool):
        if msg.data and not self.is_collecting:
            self.get_logger().info("Starting a new demonstration.")
            self.is_collecting = True

            # Clear buffers
            self.pose_data.clear()
            self.gripper_data.clear()
            self.rs_side_rgb_frames.clear()
            self.rs_front_rgb_frames.clear()
            self.rs_wrist_rgb_frames.clear()
            self.zed_rgb_frames.clear()
            self.zed_depth_frames.clear()

            self.episode_start_time = self.get_clock().now()

    def end_demo_callback(self, msg: Bool):
        if msg.data and self.is_collecting:
            self.get_logger().info(f"Ending episode {self.demo_count}")
            end_time = self.get_clock().now()
            duration = (end_time - self.episode_start_time).nanoseconds / 1e9 if self.episode_start_time else 0.0

            # Save data
            self.is_collecting = False
            self.save_demonstration()

            # Compute number of frames saved
            num_frames = len(self.pose_data)
            rate = num_frames / duration if duration > 0 else float('nan')

            # Log summary
            self.get_logger().info(f"Episode {self.demo_count} ended: Length = {duration:.2f}s, Num_frames = {num_frames}, Avg. Freq. = {rate:.2f}Hz")

            self.demo_count += 1

    def synced_callback(self, 
                        pose_msg: PoseStamped, 
                        grip_msg: Float32, 
                        rs_side_rgb_msg: CompressedImage, 
                        rs_front_rgb_msg: CompressedImage, 
                        rs_wrist_rgb_msg: CompressedImage, 
                        zed_rgb_msg: CompressedImage, 
                        zed_depth_msg: CompressedImage
                        ):  
        
        if not self.is_collecting:
            return

        # Robot pose & gripper
        p = pose_msg.pose.position
        o = pose_msg.pose.orientation
        self.pose_data.append([p.x, p.y, p.z, o.w, o.x, o.y, o.z]) # [x, y, z, qw, qx, qy, qz]
        self.gripper_data.append(grip_msg.data)

        # RealSense color
        rs_side_rgb_np = self.parse_color_image(rs_side_rgb_msg)
        self.rs_side_rgb_frames.append(rs_side_rgb_np)

        rs_front_rgb_np = self.parse_color_image(rs_front_rgb_msg)
        self.rs_front_rgb_frames.append(rs_front_rgb_np)

        rs_wrist_rgb_np = self.parse_color_image(rs_wrist_rgb_msg)
        self.rs_wrist_rgb_frames.append(rs_wrist_rgb_np)

        # ZED color
        zed_rgb_np = self.parse_color_image(zed_rgb_msg)
        self.zed_rgb_frames.append(zed_rgb_np)

        # ZED depth
        zed_depth_np = self.parse_depth_image(zed_depth_msg)
        self.zed_depth_frames.append(zed_depth_np)

        self.get_logger().info(f"Collected {len(self.pose_data)} frames")

    def save_demonstration(self):
        pose_array = np.array(self.pose_data, dtype=np.float32)
        pose_array[:, :3] /= 1000.0
        grip_array = np.array(self.gripper_data, dtype=np.float32)

        last_pose_array = np.roll(pose_array, shift=-1, axis=0)
        last_pose_array[0] = pose_array[0]

        rs_side_color_stack = np.stack(self.rs_side_rgb_frames,  axis=0) if self.rs_side_rgb_frames else []
        rs_front_color_stack = np.stack(self.rs_front_rgb_frames,  axis=0) if self.rs_front_rgb_frames else []
        rs_wrist_color_stack = np.stack(self.rs_wrist_rgb_frames,  axis=0) if self.rs_wrist_rgb_frames else []
        zed_color_stack = np.stack(self.zed_rgb_frames, axis=0) if self.zed_rgb_frames else []
        zed_depth_stack = np.stack(self.zed_depth_frames, axis=0) if self.zed_depth_frames else []

        save_dir = os.path.join(os.getcwd(), "demo_data")
        os.makedirs(save_dir, exist_ok=True)
        fn = os.path.join(save_dir, f"episode_{self.demo_count}.hdf5")
        with h5py.File(fn, "w") as f:
            f.create_dataset("pose", data=pose_array)
            f.create_dataset("gripper", data=grip_array)
            f.create_dataset("last_pose", data=last_pose_array)
            if len(rs_side_color_stack): f.create_dataset("rs_side_rgb", data=rs_side_color_stack, compression="lzf")
            if len(rs_front_color_stack): f.create_dataset("rs_front_rgb", data=rs_front_color_stack, compression="lzf")
            if len(rs_wrist_color_stack): f.create_dataset("rs_wrist_rgb", data=rs_wrist_color_stack, compression="lzf")
            if len(zed_color_stack): f.create_dataset("zed_rgb", data=zed_color_stack, compression="lzf")
            if len(zed_depth_stack): f.create_dataset("zed_depth", data=zed_depth_stack, compression="lzf")

        self.get_logger().info(f"Saved demonstration to {fn}")

    def parse_color_image(self, msg: CompressedImage) -> np.ndarray:
        """
        Decompress a sensor_msgs/CompressedImage and return an array (3,H,W) in RGB order.
        """
        # OpenCV returns BGR
        bgr = self._bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb.transpose(2, 0, 1)            # → CHW

    def parse_depth_image(self, msg: CompressedImage) -> np.ndarray:
        """
        Decode PNG depth (16-bit mm) → (H,W) uint16.  No scaling needed.
        """
        depth = self._bridge.compressed_imgmsg_to_cv2(msg)               # uint16 H×W
        if depth.dtype != np.uint16:
            depth = depth.astype(np.uint16)
        return depth


    def destroy_node(self) -> bool:
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = XArmDataCollection()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
