#!/usr/bin/env python3

import os
import numpy as np
import zarr
import cv2
import torch
from pathlib import Path
import time
from typing import Dict, List, Tuple

import rclpy
from rclpy.node import Node

# ROS messages
from std_msgs.msg import Bool, Float32
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image

# For time synchronization
from message_filters import Subscriber, ApproximateTimeSynchronizer

# Import Grounding DINO and SAM-2
from sam2.build_sam import build_sam2_video_predictor
from groundingdino.util.inference import load_model, predict, load_image
from torchvision.ops import box_convert

## Plug in Procedure: Realsense first, then ZED, then DenseTact Left (tape), then DenseTact Right (no tape)

class XArmDataCollection(Node):
    def __init__(self):
        super().__init__('Xarm_zarr_data_collection_node')
        self.get_logger().info("Initializing data_collection_node with approximate sync and zarr output.")

        # ==========================================================
        # Configuration
        # ==========================================================
        self.data_path = './demo_data_zarr'
        self.output_size = (84, 84)
        self.points_per_cloud = 2048
        
        # Camera intrinsics - ZED
        self.zed_fx = 1069.73
        self.zed_fy = 1069.73
        self.zed_cx = 1135.86
        self.zed_cy = 680.69
        
        # ZED to base transform
        self.T_base_zed = np.array([
            [-0.007, -1.000,  0.020,  0.580],
            [ 0.999, -0.007,  0.043,  0.020],
            [-0.043,  0.020,  0.999,  0.570],
            [ 0.000,  0.000,  0.000,  1.000],
        ], dtype=np.float32)
        
        # GDINO + SAM-2 configuration
        self.gdino_config = {
            "GDINO_CKPT": "/home/alex/Documents/Grounded-SAM-2/gdino_checkpoints/groundingdino_swint_ogc.pth",
            "GDINO_CFG": "/home/alex/Documents/Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            "BOX_THRESH": 0.30,
            "TEXT_THRESH": 0.20,
            "GDINO_QUERIES": ["red strawberry", "robot"],
            "SAM2_CKPT": "/home/alex/Documents/Grounded-SAM-2/checkpoints/sam2.1_hiera_base_plus.pt",
            "SAM2_CFG": "/home/alex/Documents/Grounded-SAM-2/sam2/configs/sam2.1/sam2.1_hiera_b+.yaml",
        }
        
        # Initialize models
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.init_models()
        
        # Create zarr root
        Path(self.data_path).mkdir(parents=True, exist_ok=True)
        self.zarr_root = zarr.open(self.data_path, mode='a')
        
        # Count episodes
        self.episode_count = len([k for k in self.zarr_root.keys() if k.startswith('episode_')])

        # ==========================================================
        # Add rate limiting parameters
        # ==========================================================
        self.desired_rate = 10.0  # Hz 
        self.last_processed_time = self.get_clock().now()
        self.frame_count = 0
        self.last_rate_report_time = self.get_clock().now()
        self.rate_report_interval = 2.0 

        # ==========================================================
        # Create message_filters Subscribers for RealSense + Robot
        # ==========================================================
        self.pose_sub = Subscriber(self, PoseStamped, 'robot_position_action')
        self.gripper_sub = Subscriber(self, Float32, 'gripper_position')
        self.rs_color_sub = Subscriber(self, Image, '/camera/camera/color/image_raw')
        self.rs_depth_sub = Subscriber(self, Image, '/camera/camera/depth/image_rect_raw')
        self.zed_color_sub = Subscriber(self, Image, 'zed_image/rgb')
        self.zed_depth_sub = Subscriber(self, Image, 'zed_image/depth')
        self.dt_left_sub = Subscriber(self, Image, 'RunCamera/image_raw_8')
        self.dt_right_sub = Subscriber(self, Image, 'RunCamera/image_raw_10')

        # ApproximateTimeSynchronizer for all five
        self.sync = ApproximateTimeSynchronizer(
            [
                self.pose_sub,
                self.gripper_sub,
                self.rs_color_sub,
                self.rs_depth_sub,
                self.dt_left_sub,
                self.dt_right_sub,
                self.zed_color_sub,
                self.zed_depth_sub
            ],
            queue_size=30,
            slop=0.1,
            allow_headerless=True  # needed because Float32 doesn't have a header
        )
        self.sync.registerCallback(self.synced_callback)

        # ==========================================================
        # Start/End Demo as separate subscriptions
        # ==========================================================

        self.start_sub = self.create_subscription(Bool, 'start_demo', self.start_demo_callback, 10)
        self.end_sub = self.create_subscription(Bool, 'end_demo', self.end_demo_callback, 10)

        # ==========================================================
        # State and storage
        # ==========================================================
        self.is_collecting = False

        # Robot data
        self.pose_data = []
        self.gripper_data = []

        # RealSense
        self.rs_color_frames = []
        self.rs_depth_frames = []

        # ZED
        self.zed_color_frames = []
        self.zed_depth_frames = []

        # DenseTacts
        self.dt_left_frames = []
        self.dt_right_frames = []

    def init_models(self):
        """Initialize GDINO and SAM-2 models"""
        self.get_logger().info(f"Initializing models on {self.device}...")
        
        # Grounding DINO
        self.gdino = load_model(
            model_config_path=self.gdino_config["GDINO_CFG"],
            model_checkpoint_path=self.gdino_config["GDINO_CKPT"],
            device=self.device,
        )
        
        # SAM-2
        self.video_predictor = build_sam2_video_predictor(
            self.gdino_config["SAM2_CFG"], 
            self.gdino_config["SAM2_CKPT"], 
            device=self.device
        )
        
        self.get_logger().info("Models ready.")

    ####################################################################
    # Start/End Demos
    ####################################################################
    def start_demo_callback(self, msg: Bool):
        if msg.data and not self.is_collecting:
            self.get_logger().info("Starting a new demonstration.")
            self.is_collecting = True

            # Clear buffers
            self.pose_data.clear()
            self.gripper_data.clear()
            self.rs_color_frames.clear()
            self.rs_depth_frames.clear()
            self.zed_color_frames.clear()
            self.zed_depth_frames.clear()
            self.dt_left_frames.clear()
            self.dt_right_frames.clear()

    def end_demo_callback(self, msg: Bool):
        if msg.data and self.is_collecting:
            self.get_logger().info("Ending demonstration and saving.")
            self.is_collecting = False
            self.save_demonstration()

    ####################################################################
    # Time sychronization callback
    ####################################################################
    def synced_callback(self,
                        pose_msg: PoseStamped,
                        grip_msg: Float32,
                        rs_color_msg: Image,
                        rs_depth_msg: Image,
                        dt_left_msg: Image,
                        dt_right_msg: Image,
                        zed_color_msg: Image,
                        zed_depth_msg: Image):
        """
        Called when all sensor data arrives (approximately) at the same time.
        """
        current_time = self.get_clock().now()
        
        # Calculate time since last report to periodically show the actual rate
        time_since_report = (current_time - self.last_rate_report_time).nanoseconds / 1e9
        if time_since_report >= self.rate_report_interval and self.frame_count > 0:
            actual_rate = self.frame_count / time_since_report
            self.frame_count = 0
            self.last_rate_report_time = current_time
        
        # Skip if not collecting
        if not self.is_collecting:
            return
        
        # Apply rate limiting - only process if enough time has passed
        elapsed = (current_time - self.last_processed_time).nanoseconds / 1e9  # seconds
        if elapsed < (1.0 / self.desired_rate):
            return
            
        # Update last processed time
        self.last_processed_time = current_time
        self.frame_count += 1

        # Robot pose/gripper
        p = pose_msg.pose.position
        o = pose_msg.pose.orientation
        self.pose_data.append([p.x, p.y, p.z, o.x, o.y, o.z, o.w])

        self.gripper_data.append(grip_msg.data)

        # Parse RealSense color
        rs_color_np_bgr = self.parse_color_image(rs_color_msg)  # shape (H, W, 3)
        rs_color_np_rgb = cv2.cvtColor(rs_color_np_bgr, cv2.COLOR_BGR2RGB)
        self.rs_color_frames.append(rs_color_np_rgb)

        # Parse RealSense depth
        rs_depth_np = self.parse_depth_image(rs_depth_msg)  # shape (H, W)
        self.rs_depth_frames.append(rs_depth_np)

        # Parse ZED color
        zed_np = self.parse_color_image(zed_color_msg)  # shape (H, W, 3)
        self.zed_color_frames.append(zed_np)

        # Parse ZED depth
        zed_depth_np = self.parse_depth_image(zed_depth_msg)  # shape (H, W)
        self.zed_depth_frames.append(zed_depth_np)

        # Capture from DenseTacts cameras
        dt_left_np = self.parse_dt_image(dt_left_msg)  # shape (H, W, 3)
        dt_right_np = self.parse_dt_image(dt_right_msg)  # shape (H, W, 3)
        self.dt_left_frames.append(dt_left_np)
        self.dt_right_frames.append(dt_right_np)

    ####################################################################
    # Save the demonstration to Zarr
    ####################################################################

    def save_demonstration(self):
        """Process the recorded episode using GDINO + SAM-2 and save to zarr"""
        self.get_logger().info("Processing episode with GDINO + SAM-2...")
        
        # Convert lists to numpy arrays
        pose_array = np.array(self.pose_data, dtype=np.float32)
        
        # convert mm to meters for pose
        pose_array[:, :3] /= 1000.0

        grip_array = np.array(self.gripper_data, dtype=np.float32)

        # Create last_pose by shifting pose_array by one index
        last_pose_array = np.roll(pose_array, shift=-1, axis=0)
        last_pose_array[0] = pose_array[0]
        
        # Stack data
        rs_color_stack = np.stack(self.rs_color_frames, axis=0) if len(self.rs_color_frames) > 0 else []
        rs_depth_stack = np.stack(self.rs_depth_frames, axis=0) if len(self.rs_depth_frames) > 0 else []
        zed_color_stack = np.stack(self.zed_color_frames, axis=0) if len(self.zed_color_frames) > 0 else []
        zed_depth_stack = np.stack(self.zed_depth_frames, axis=0) if len(self.zed_depth_frames) > 0 else []
        dt_left_stack = np.stack(self.dt_left_frames, axis=0) if len(self.dt_left_frames) > 0 else []
        dt_right_stack = np.stack(self.dt_right_frames, axis=0) if len(self.dt_right_frames) > 0 else []

        T = len(zed_color_stack)
        
        # Process with GDINO + SAM-2
        zed_pc_list = self.process_episode_with_gdino_sam(zed_color_stack, zed_depth_stack)
        
        # Resize images
        rs_color_resized = np.stack([cv2.resize(img, self.output_size) for img in rs_color_stack])
        rs_depth_resized = np.stack([cv2.resize(img, self.output_size) for img in rs_depth_stack])
        zed_color_resized = np.stack([cv2.resize(img, self.output_size) for img in zed_color_stack])
        zed_depth_resized = np.stack([cv2.resize(img, self.output_size) for img in zed_depth_stack])
        
        # Save to zarr
        episode_key = f"episode_{self.episode_count:04d}"
        grp = self.zarr_root.create_group(episode_key)
        
        # Stack camera data
        rgb_stacked = np.stack([rs_color_resized, zed_color_resized], axis=1)
        depth_stacked = np.stack([rs_depth_resized, zed_depth_resized], axis=1)
        
        grp.array("rgb", rgb_stacked, dtype=np.uint8, chunks=(1, 2, *self.output_size, 3))
        grp.array("depth", depth_stacked, dtype=np.float32, chunks=(1, 2, *self.output_size))
        
        # Point clouds (only ZED)
        empty_pc = np.zeros((T, self.points_per_cloud, 6), dtype=np.float32)
        zed_pc_array = np.stack(zed_pc_list)
        pc_stacked = np.stack([empty_pc, zed_pc_array], axis=1)
        
        grp.array("point_cloud", pc_stacked, dtype=np.float32, chunks=(1, 2, self.points_per_cloud, 6))
        
        # Save agent data
        grp.array("agent_pos", pose_array, dtype=np.float32)
        
        # Calculate actions (difference between consecutive poses)  
        actions = last_pose_array - pose_array
        grp.array("action", actions, dtype=np.float32)
        grp.attrs["length"] = T
        
        self.episode_count += 1
        self.get_logger().info(f"Saved episode {episode_key}")

    ####################################################################
    # GDINO + SAM-2 Processing
    ####################################################################
    
    def process_episode_with_gdino_sam(self, zed_rgb_seq, zed_depth_seq):
        """Process episode with GDINO + SAM-2"""
        T, H, W, _ = zed_rgb_seq.shape
        masks = {}
        
        # Create temp directory for SAM-2
        tmp_dir = Path("__tmp_gsam2_frames")
        tmp_dir.mkdir(exist_ok=True)
        
        # Save frames for SAM-2 processing
        for t in range(T):
            cv2.imwrite(str(tmp_dir / f"{t:04d}.jpg"), zed_rgb_seq[t])
        
        # Run GDINO on first frame
        frame0_path = str(tmp_dir / "0000.jpg")
        image_src, proc_img = load_image(frame0_path)
        boxes_all = []
        
        for caption in self.gdino_config["GDINO_QUERIES"]:
            b, _, _ = predict(
                model=self.gdino,
                image=proc_img,
                caption=caption,
                box_threshold=self.gdino_config["BOX_THRESH"],
                text_threshold=self.gdino_config["TEXT_THRESH"],
            )
            if b.numel():
                boxes_all.append(b)
        
        if not boxes_all:
            self.get_logger().warning("GDINO found no objects")
            empty_pc = np.zeros((self.points_per_cloud, 6), dtype=np.float32)
            return [empty_pc for _ in range(T)]
        
        boxes = torch.cat(boxes_all, dim=0)
        boxes = boxes * torch.tensor([W, H, W, H], device=boxes.device)
        boxes_xyxy = box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy").cpu().numpy()
        
        # Propagate masks with SAM-2
        state = self.video_predictor.init_state(video_path=str(tmp_dir))
        
        for obj_id, box in enumerate(boxes_xyxy, start=1):
            self.video_predictor.add_new_points_or_box(
                inference_state=state, frame_idx=0,
                obj_id=obj_id, box=box.astype(np.float32))
        
        for f_idx, obj_ids, mask_logits in self.video_predictor.propagate_in_video(state):
            masks_this = [(mask_logits[i] > 0).cpu().numpy() for i in range(len(obj_ids))]
            masks[f_idx] = self.combine_masks(masks_this)
        
        # Generate point clouds
        pc_list = []
        for t in range(T):
            mask = masks.get(t, np.zeros((H, W), dtype=np.uint8))
            pc = self.depth_to_point_cloud(mask, zed_depth_seq[t], zed_rgb_seq[t])
            pc_down = self.downsample_point_cloud(pc, self.points_per_cloud)
            pc_list.append(pc_down)
        
        # Cleanup
        for p in tmp_dir.glob("*.jpg"):
            p.unlink()
        tmp_dir.rmdir()
        
        return pc_list
    
    def combine_masks(self, masks):
        """Combine multiple masks into one"""
        if not masks:
            return np.zeros(0, dtype=np.uint8)
        merged = np.zeros_like(masks[0], dtype=np.uint8)
        for m in masks:
            merged |= m.astype(bool)
        return (merged * 255).astype(np.uint8)
    
    def depth_to_point_cloud(self, mask, depth, rgb_bgr):
        """Convert depth image to point cloud"""
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = np.squeeze(mask, axis=0)
        
        if mask.sum() == 0:
            return np.empty((0, 6), dtype=np.float32)
        
        v_idx, u_idx = np.where(mask > 0)
        Z = depth[v_idx, u_idx]
        
        valid = np.isfinite(Z) & (Z > 1e-6)
        if not np.any(valid):
            return np.empty((0, 6), dtype=np.float32)
        
        u, v, Z = u_idx[valid], v_idx[valid], Z[valid]
        X = (u - self.zed_cx) * Z / self.zed_fx
        Y = (v - self.zed_cy) * Z / self.zed_fy
        pts_cam = np.stack([X, Y, Z, np.ones_like(Z)], axis=1)
        
        # Transform to base frame
        pts_base = (self.T_base_zed @ pts_cam.T).T[:, :3]
        
        # Add color
        rgb = rgb_bgr[v, u][:, ::-1] / 255.0
        return np.concatenate([pts_base, rgb], axis=1).astype(np.float32)
    
    def downsample_point_cloud(self, pc, target_points):
        """Downsample point cloud to target number of points"""
        if pc.shape[0] <= target_points:
            padded = np.zeros((target_points, 6), dtype=np.float32)
            padded[:pc.shape[0]] = pc
            return padded
        else:
            indices = np.random.choice(pc.shape[0], target_points, replace=False)
            return pc[indices]

    ####################################################################
    # Helper Functions (unchanged)
    ####################################################################  
    def parse_dt_image(self, img_msg: Image) -> np.ndarray:
        """
        Parse DenseTact camera images, which publish in RGB but we want them in BGR.
        """
        height = img_msg.height
        width = img_msg.width
        step = img_msg.step
        data = img_msg.data

        # Convert raw bytes to 1D array of type uint8 and reshape
        np_data = np.frombuffer(data, dtype=np.uint8)
        np_data_2d = np_data.reshape((height, step))

        # Assume 3 channels: step == width * 3 (no row padding)
        channels = 3
        expected_bytes_per_row = width * channels
        np_data_2d_sliced = np_data_2d[:, :expected_bytes_per_row]

        # Reshape to (H, W, 3)
        color_img = np_data_2d_sliced.reshape((height, width, channels))

        # If the DT images are published in "rgb8", swap channels to get BGR
        if img_msg.encoding == "rgb8":
            color_img = color_img[..., ::-1]

        return color_img

    def parse_color_image(self, img_msg: Image) -> np.ndarray:
        """
        Convert sensor_msgs/Image (e.g., 'bgr8' or 'rgb8') into a NumPy array (H,W,3).
        Adjust for your specific encoding as needed.
        """
        height = img_msg.height
        width = img_msg.width
        step = img_msg.step  # bytes per row
        data = img_msg.data  # raw bytes

        # Convert to a 1D numpy array of dtype uint8
        np_data = np.frombuffer(data, dtype=np.uint8)
        
        # Reshape to (height, step)
        np_data_2d = np_data.reshape((height, step))

        # For color images, we expect step == width * 3 (if no row padding)
        channels = 3
        expected_bytes_per_row = width * channels
        np_data_2d_sliced = np_data_2d[:, :expected_bytes_per_row]

        # Finally reshape to (H, W, C)
        color_img = np_data_2d_sliced.reshape((height, width, channels))
        return color_img
    
    def parse_depth_image(self, img_msg: Image) -> np.ndarray:
        """
        Convert sensor_msgs/Image depth into a NumPy array (H, W).
        Handles both 32FC1 and 16UC1 encodings.
        """
        height = img_msg.height
        width = img_msg.width
        step = img_msg.step  # bytes per row
        data = img_msg.data
        
        if img_msg.encoding == "32FC1":
            # Each pixel is a 32-bit float => 4 bytes
            floats_per_row = step // 4
            np_data = np.frombuffer(data, dtype=np.float32)
            
            # Reshape to (height, floats_per_row)
            depth_2d = np_data.reshape((height, floats_per_row))
            
            # Slice out the valid columns
            depth_2d_sliced = depth_2d[:, :width]
            
        elif img_msg.encoding == "16UC1":
            # Each pixel is a 16-bit unsigned int => 2 bytes
            ints_per_row = step // 2
            np_data = np.frombuffer(data, dtype=np.uint16)
            
            # Reshape to (height, ints_per_row)
            depth_2d = np_data.reshape((height, ints_per_row))
            
            # Slice out the valid columns
            depth_2d_sliced = depth_2d[:, :width]
            
            # Convert uint16 millimeters to float32 meters
            depth_2d_sliced = depth_2d_sliced.astype(np.float32) / 1000.0
        else:
            self.get_logger().error(f"Unsupported depth encoding: {img_msg.encoding}")
            depth_2d_sliced = np.zeros((height, width), dtype=np.float32)
            
        return depth_2d_sliced


def main(args=None):
    rclpy.init(args=args)
    node = XArmDataCollection()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()