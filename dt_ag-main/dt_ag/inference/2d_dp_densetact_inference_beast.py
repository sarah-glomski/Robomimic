#!/usr/bin/env python3

import sys
import rclpy
from typing import Tuple    
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32, Bool
import pathlib
# import pygame
import numpy as np
import torch
from cv_bridge import CvBridge
import cv2
import time
import diffusers
import hydra
import dill
from pathlib import Path
# Multiprocessing imports
import multiprocessing
from multiprocessing import Process, Manager, Queue
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage
import message_filters
from rclpy.executors import MultiThreadedExecutor
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
DEBUG_DIR = SCRIPT_DIR / "2d_dp_debug_densetact_beast"
gendp_path = '/home/alex/Documents/DT-Diffusion-Policy/gendp/gendp'

if gendp_path not in sys.path:
    sys.path.append(gendp_path)

from gendp.model.common.rotation_transformer import RotationTransformer


class PolicyNode3D(Node):
    def __init__(self, shared_obs, action_queue, start_time, shape_meta: dict):
        super().__init__('Policy_Node')
        np.set_printoptions(suppress=True, precision=4)

        # --- QoS tuned for high-rate image streams ---
        sensor_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)

        # Extract observation keys from shape_meta to determine which cameras to use
        self.obs_keys = list(shape_meta['obs'].keys())
        self.has_zed_rgb = 'zed_rgb' in self.obs_keys
        self.has_zed_depth = 'zed_depth_images' in self.obs_keys
        self.has_rs_side_rgb = 'rs_side_rgb' in self.obs_keys
        self.has_dt_left = 'dt_left' in self.obs_keys
        self.has_dt_right = 'dt_right' in self.obs_keys
        
        self.get_logger().info(f"Policy observation keys: {self.obs_keys}")
        self.get_logger().info(f"Using ZED RGB camera: {self.has_zed_rgb}")
        self.get_logger().info(f"Using ZED Depth camera: {self.has_zed_depth}")
        self.get_logger().info(f"Using RealSense RGB camera: {self.has_rs_side_rgb}")
        self.get_logger().info(f"Using DT Left camera: {self.has_dt_left}")
        self.get_logger().info(f"Using DT Right camera: {self.has_dt_right}")

        # Build subscriber list dynamically
        subscribers = []
        
        # Always subscribe to pose and gripper
        self.pose_sub = message_filters.Subscriber(self, PoseStamped, '/robot_pose', qos_profile=sensor_qos)
        self.gripper_sub = message_filters.Subscriber(self, Float32, '/gripper_state', qos_profile=sensor_qos)
        subscribers.extend([self.pose_sub, self.gripper_sub])
        
        # Conditionally subscribe to cameras based on policy requirements
        if self.has_zed_rgb:
            # self.zed_rgb_sub = message_filters.Subscriber(self, Image, '/zed_image/rgb', qos_profile=sensor_qos)
            self.zed_rgb_compressed_sub = message_filters.Subscriber(self, CompressedImage, '/zed_image/rgb/compressed', qos_profile=sensor_qos)
            subscribers.append(self.zed_rgb_compressed_sub)

        if self.has_zed_depth:
            self.zed_depth_compressed_sub = message_filters.Subscriber(self, CompressedImage, '/zed_image/depth/compressed', qos_profile=sensor_qos)
            subscribers.append(self.zed_depth_compressed_sub)
            
        if self.has_rs_side_rgb:
            self.rs_color_compressed_sub = message_filters.Subscriber(self, CompressedImage, '/rs_side/rs_side/color/image_raw/compressed', qos_profile=sensor_qos)
            subscribers.append(self.rs_color_compressed_sub)

        if self.has_dt_left:
            self.dt_left_compressed_sub = message_filters.Subscriber(self, CompressedImage, '/RunCamera/image_raw_8/compressed', qos_profile=sensor_qos)
            subscribers.append(self.dt_left_compressed_sub)

        if self.has_dt_right:
            self.dt_right_compressed_sub = message_filters.Subscriber(self, CompressedImage, '/RunCamera/image_raw_10/compressed', qos_profile=sensor_qos)
            subscribers.append(self.dt_right_compressed_sub)

        # Create synchronizer with only the required subscribers
        self.sync = message_filters.ApproximateTimeSynchronizer(
            subscribers, 
            queue_size=100,         # how many "unmatched" msgs to keep
            slop=0.05,           
            allow_headerless=True)  # Allow messages without headers
        
        self.sync.registerCallback(self.synced_obs_callback)
        
        # Publishers
        self.gripper_pub = self.create_publisher(Float32, '/gripper_position', 1)
        self.reset_xarm_pub = self.create_publisher(Bool, '/reset_xarm', 1)
        self.pause_xarm_pub = self.create_publisher(Bool, '/pause_xarm', 1)
        self.pub_robot_pose = self.create_publisher(PoseStamped, '/xarm_position', 1)
        
        self._bridge = CvBridge()

        self.start_time = start_time
        self.shape_meta = shape_meta

        # Shared data and action queue
        self.shared_obs = shared_obs
        self.action_queue = action_queue
        
        # Timers
        self.create_timer(1/30.0, self.timer_callback)
        self.create_timer(1/30.0, self.update_observation)

        # Horizon for keeping recent observations
        self.observation_horizon = 2
        
        # Buffers for observations
        self.pose_buffer = []
        if self.has_zed_rgb:
            self.zed_rgb_buffer = []
        if self.has_zed_depth:
            self.zed_depth_buffer = []
        if self.has_rs_side_rgb:
            self.rs_color_buffer = []
        if self.has_dt_left:
            self.dt_left_buffer = []
        if self.has_dt_right:
            self.dt_right_buffer = []

        # Rotation transformer
        self.tf = RotationTransformer('rotation_6d', 'quaternion')

        # Control state
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.paused = False
        self.even = True
        self.pending_actions = []
        self.zed_crop = True
        self.rs_side_crop = False
        self.dt_left_crop = True
        self.dt_right_crop = True

        # Jiggle gripper to activate it
        msg = Float32()
        msg.data = 0.05
        self.gripper_pub.publish(msg)
        time.sleep(0.5)
        msg.data = 0.0
        self.gripper_pub.publish(msg)
        self.gripper_state = 0.0

        # Reset robot to home position
        self.reset_xarm_pub.publish(Bool(data=True))

        self.get_logger().info("2D Diffusion Policy Node for Baseline Gripper Initialized!")

    def reset_xarm(self):
        """Reset robot to home position"""
        self.get_logger().info("Reset xarm.")
        ee_pose = PoseStamped()
        ee_pose.header.stamp = self.get_clock().now().to_msg()
        ee_pose.pose.position.x = float(0.1669)
        ee_pose.pose.position.y = float(0.0019)
        ee_pose.pose.position.z = float(0.2308)
        quats = [0.9999, -0.00995, 0.00507, 0.00785]
        ee_pose.pose.orientation.x = float(quats[0])
        ee_pose.pose.orientation.y = float(quats[1])
        ee_pose.pose.orientation.z = float(quats[2])
        ee_pose.pose.orientation.w = float(quats[3])
        self.pub_robot_pose.publish(ee_pose)

        # Set gripper to open position
        gripper_value = float(0.0)
        msg = Float32()
        msg.data = gripper_value
        self.gripper_pub.publish(msg)

    def pause_policy(self):
        """Pause robot execution"""
        self.get_logger().info("Pause policy.")
        self.pause_xarm_pub.publish(Bool(data=True))
        self.paused = True
        self.shared_obs['paused'] = True          # tell inference
        self.pending_actions.clear()              # drop anything queued
        while not self.action_queue.empty():      # flush queue
            _ = self.action_queue.get_nowait()

    def resume_policy(self):
        """Resume robot execution"""
        self.get_logger().info("Resume policy.")
        self.pause_xarm_pub.publish(Bool(data=False))        
        self.paused = False
        self.shared_obs['paused'] = False         # let inference run

    def cleanup(self):
        """Clean up resources"""
        # pygame.quit()

    def update_observation(self):
        """Consolidate the latest *horizon* observations and push to `shared_obs`."""
        # Check minimum buffer lengths based on what cameras are available
        buffer_lengths = [len(self.pose_buffer)]
        if self.has_zed_rgb:
            buffer_lengths.append(len(self.zed_rgb_buffer))
        if self.has_rs_side_rgb:
            buffer_lengths.append(len(self.rs_color_buffer))
            
        min_len = min(buffer_lengths)
        if min_len < self.observation_horizon:
            return  # not enough data yet

        # Robot Pose
        pose_slice = self.pose_buffer[-self.observation_horizon:]
        pose_np = np.stack([p[0] for p in pose_slice])           # (T, 10)
        pose_tstamps = np.array([p[1] for p in pose_slice])           # (T,)
        pose_tensor = torch.from_numpy(pose_np).unsqueeze(0)         # (1, T, 10)

        # Build observation dict dynamically
        obs_dict = {
            'pose': pose_tensor,
            'pose_timestamps': pose_tstamps,
        }

        # Add ZED data if available
        if self.has_zed_rgb:
            zed_rgb_slice = self.zed_rgb_buffer[-self.observation_horizon:]
            zed_rgb_np = np.stack([r[0] for r in zed_rgb_slice])            # (T, 3, H, W)
            zed_rgb_tstamps = np.array([r[1] for r in zed_rgb_slice])            # (T,)
            zed_rgb_tensor = torch.from_numpy(zed_rgb_np).unsqueeze(0)          # (1, T, 3, H, W)
            obs_dict['zed_rgb'] = zed_rgb_tensor
            obs_dict['zed_rgb_timestamps'] = zed_rgb_tstamps

        # Add RealSense data if available
        if self.has_rs_side_rgb:
            rs_slice = self.rs_color_buffer[-self.observation_horizon:] 
            rs_np = np.stack([r[0] for r in rs_slice])             # (T, 3, H, W)
            rs_rgb_tstamps = np.array([r[1] for r in rs_slice])             # (T,)
            rs_rgb_tensor = torch.from_numpy(rs_np).unsqueeze(0)           # (1, T, 3, H, W)
            obs_dict['rs_side_rgb'] = rs_rgb_tensor
            obs_dict['rs_rgb_timestamps'] = rs_rgb_tstamps

        if self.has_dt_left:
            dt_left_slice = self.dt_left_buffer[-self.observation_horizon:]
            dt_left_np = np.stack([r[0] for r in dt_left_slice])
            dt_left_tstamps = np.array([r[1] for r in dt_left_slice])
            dt_left_tensor = torch.from_numpy(dt_left_np).unsqueeze(0)
            obs_dict['dt_left'] = dt_left_tensor
            obs_dict['dt_left_timestamps'] = dt_left_tstamps

        if self.has_dt_right:
            dt_right_slice = self.dt_right_buffer[-self.observation_horizon:]
            dt_right_np = np.stack([r[0] for r in dt_right_slice])
            dt_right_tstamps = np.array([r[1] for r in dt_right_slice])
            dt_right_tensor = torch.from_numpy(dt_right_np).unsqueeze(0)
            obs_dict['dt_right'] = dt_right_tensor
            obs_dict['dt_right_timestamps'] = dt_right_tstamps

        # Push to shared memory (IPC)
        self.shared_obs['obs'] = obs_dict

    def timer_callback(self):
        # Pull freshly queued actions into the local list
        # while not self.action_queue.empty():
        #     self.pending_actions.append(self.action_queue.get())

        while not self.action_queue.empty():
            self.pending_actions.append(self.action_queue.get()[0])

        if not self.pending_actions:
            self.shared_obs["exec_done"] = True
            return
        else:
            self.shared_obs["exec_done"] = False

        while self.pending_actions:
            action = self.pending_actions.pop(0)

            ee_pos, ee_rot6d, grip = action[:3], action[3:9], action[9]
            ee_quat = self.tf.forward(torch.tensor(ee_rot6d))
            ee_msg = PoseStamped()
            ee_msg.header.stamp = self.get_clock().now().to_msg()
            ee_msg.pose.position.x = float(ee_pos[0])
            ee_msg.pose.position.y = float(ee_pos[1])
            ee_msg.pose.position.z = float(ee_pos[2])
            ee_msg.pose.orientation.x = float(ee_quat[1])
            ee_msg.pose.orientation.y = float(ee_quat[2])
            ee_msg.pose.orientation.z = float(ee_quat[3])
            ee_msg.pose.orientation.w = float(ee_quat[0])

            grip_msg = Float32()
            grip_msg.data = float(grip)

            if not self.paused:
                self.get_logger().info(f"Publishing action: Position = {ee_pos.round(4)} | Gripper = {grip:.4f} | Timestamp = {time.monotonic() - self.start_time}")
                self.pub_robot_pose.publish(ee_msg)
                self.gripper_pub.publish(grip_msg)

            # time.sleep(0.01)
        
        self.shared_obs["exec_done"] = True

    def synced_obs_callback(self, *args):
        """Process synchronized observations and generate point cloud"""
                
        # Parse arguments based on what's available
        arg_idx = 0
        pose_msg = args[arg_idx]
        arg_idx += 1
        
        gripper_msg = args[arg_idx]
        arg_idx += 1
        
        zed_rgb_msg = None
        zed_depth_msg = None
        rs_side_msg = None
        dt_left_msg = None
        dt_right_msg = None
        
        if self.has_zed_rgb:
            zed_rgb_msg = args[arg_idx]
            arg_idx += 1
            
        if self.has_rs_side_rgb:
            rs_side_msg = args[arg_idx]
            arg_idx += 1

        if self.has_dt_left:
            dt_left_msg = args[arg_idx]
            arg_idx += 1

        if self.has_dt_right:
            dt_right_msg = args[arg_idx]
            arg_idx += 1
            
        # Update gripper state
        self.gripper_state = gripper_msg.data
        
        # Process pose message
        self.pose_callback(pose_msg)

        # Process camera messages based on availability
        if self.has_zed_rgb and zed_rgb_msg is not None:
            self.zed_rgb_compressed_callback(zed_rgb_msg)

        if self.has_zed_depth and zed_depth_msg is not None:
            self.zed_depth_compressed_callback(zed_depth_msg)

        if self.has_rs_side_rgb and rs_side_msg is not None:
            self.rs_rgb_compressed_callback(rs_side_msg)

        if self.has_dt_left and dt_left_msg is not None:
            self.dt_left_compressed_callback(dt_left_msg)

        if self.has_dt_right and dt_right_msg is not None:
            self.dt_right_compressed_callback(dt_right_msg)

        self.update_observation()

    def pose_callback(self, msg):
        """Process robot pose"""
        robot_pos = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z] 

        # tf.inverse expects [wxyz]
        robot_ori = [msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z] 
        
        # # Convert to 6D rotation representation
        robot_ori_tensor = torch.tensor(robot_ori, dtype=torch.float32)
        robot_ori_6d = self.tf.inverse(robot_ori_tensor)

        # Combine position, orientation
        robot_pose = np.concatenate([robot_pos, robot_ori_6d.numpy(), [self.gripper_state]], axis=None)
        self.pose_buffer.append((robot_pose, time.monotonic() - self.start_time))
        if len(self.pose_buffer) > self.observation_horizon:
            self.pose_buffer.pop(0)

    def zed_rgb_compressed_callback(self, msg):
        """Process zed compressed rgb message"""
        zed_rgb_img = self._bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        # Image is already in BGR format from compressed message, no conversion needed
        # swap bgr and rgb
        zed_rgb_img = zed_rgb_img[:, :, ::-1] # Shape is (H, W, 3)
        # self.get_logger().info(f"Zed rgb image shape: {zed_rgb_img.shape}")
        if self.zed_crop:
            zed_rgb_img = zed_rgb_img[:, :400, :]

        zed_rgb_img = self.resize_for_policy(zed_rgb_img, 'zed_rgb')

        self.zed_rgb_buffer.append((zed_rgb_img, time.monotonic() - self.start_time))
        if len(self.zed_rgb_buffer) > self.observation_horizon:
            self.zed_rgb_buffer.pop(0)

    def rs_rgb_compressed_callback(self, msg):
        """Process rs compressed rgb message"""
        rs_img = self._bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        # Image is already in BGR format from compressed message, no conversion needed
        # swap bgr and rgb
        rs_img = rs_img[:, :, ::-1] # Shape is (H, W, 3)
        # self.get_logger().info(f"Rs rgb image shape: {rs_img.shape}")
        if self.rs_side_crop:
            rs_img = rs_img[160:320, :640, :]
        rs_img = self.resize_for_policy(rs_img, 'rs_side_rgb')

        self.rs_color_buffer.append((rs_img, time.monotonic() - self.start_time))
        if len(self.rs_color_buffer) > self.observation_horizon:
            self.rs_color_buffer.pop(0)

    def zed_depth_compressed_callback(self, msg):
        """Process zed depth compressed message"""
        zed_depth_img = self._bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='32FC1')
        zed_depth_img = self.resize_for_policy(zed_depth_img, 'zed_depth_images')

        self.zed_depth_buffer.append((zed_depth_img, time.monotonic() - self.start_time))
        if len(self.zed_depth_buffer) > self.observation_horizon:
            self.zed_depth_buffer.pop(0)

    def dt_left_compressed_callback(self, msg):
        """Process dt left compressed message"""
        dt_left_img = self._bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8') 
        # self.get_logger().info(f"Dt left image shape: {dt_left_img.shape}")
        if self.dt_left_crop:
            dt_left_img = dt_left_img[:, 50:300, :]

        dt_left_img = self.resize_for_policy(dt_left_img, 'dt_left')

        self.dt_left_buffer.append((dt_left_img, time.monotonic() - self.start_time))
        if len(self.dt_left_buffer) > self.observation_horizon:
            self.dt_left_buffer.pop(0)

    def dt_right_compressed_callback(self, msg):
        """Process dt right compressed message"""
        dt_right_img = self._bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        # self.get_logger().info(f"Dt right image shape: {dt_right_img.shape}")
        if self.dt_right_crop:
            dt_right_img = dt_right_img[:, 50:300, :]

        dt_right_img = self.resize_for_policy(dt_right_img, 'dt_right') 

        self.dt_right_buffer.append((dt_right_img, time.monotonic() - self.start_time))
        if len(self.dt_right_buffer) > self.observation_horizon:
            self.dt_right_buffer.pop(0)

    def resize_for_policy(self, img: np.ndarray, cam_key: str) -> np.ndarray:
        """
        Resize *and* change layout from HWC-BGR (OpenCV) to CHW-RGB according to `shape_meta`.
        """
        C, H, W = self.shape_meta['obs'][cam_key]['shape']
        # self.get_logger().info(f"Resizing {cam_key} to {H}x{W}")
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))                  # HWC ➜ CHW
        assert img.shape == (C, H, W), \
            f"{cam_key} expected {(C,H,W)}, got {img.shape}"
        return img


def inference_loop(model_path, shared_obs, action_queue, action_horizon = 4, device = "cuda", start_time = 0):

    model = load_diffusion_policy(model_path)
    print("Inference process started.")

    # ─── Wait until first observation ───────────────────────────────
    while shared_obs.get("obs") is None:
        time.sleep(0.05)

    # Initialize previous timestamps based on available data
    prev_timestamps = {}
    obs_now = shared_obs["obs"]
    
    if "pose_timestamps" in obs_now:
        prev_timestamps["pose"] = obs_now["pose_timestamps"][-1]
    if "zed_rgb_timestamps" in obs_now:
        prev_timestamps["zed"] = obs_now["zed_rgb_timestamps"][-1]
    if "rs_rgb_timestamps" in obs_now:
        prev_timestamps["rs"] = obs_now["rs_rgb_timestamps"][-1]

    while True:
        loop_start_time = time.time()
        
        # If user paused with the keyboard, spin-wait cheaply
        if shared_obs.get("paused", False):
            time.sleep(0.05)
            continue

        wait_start_time = time.time()
        while True:
            obs_now = shared_obs["obs"]

            # Check if we have new data for all available sensors
            all_new = True
            
            if "pose_timestamps" in obs_now:
                pose_new = np.min(obs_now["pose_timestamps"]) > prev_timestamps.get("pose", -1)
                all_new = all_new and pose_new
                
            if "zed_rgb_timestamps" in obs_now:
                zed_new = np.min(obs_now["zed_rgb_timestamps"]) > prev_timestamps.get("zed", -1)
                all_new = all_new and zed_new
                
            if "rs_rgb_timestamps" in obs_now:
                rs_new = np.min(obs_now["rs_rgb_timestamps"]) > prev_timestamps.get("rs", -1)
                all_new = all_new and rs_new

            # if all_new and shared_obs['exec_done']:
            #     break

            if all_new:
                break
            time.sleep(0.001)
        
        wait_time = time.time() - wait_start_time

        # Save newest observation timestamps
        prep_start_time = time.time()
        if "pose_timestamps" in obs_now:
            prev_timestamps["pose"] = obs_now["pose_timestamps"][-1]
        if "zed_rgb_timestamps" in obs_now:
            prev_timestamps["zed"] = obs_now["zed_rgb_timestamps"][-1]
        if "rs_rgb_timestamps" in obs_now:
            prev_timestamps["rs"] = obs_now["rs_rgb_timestamps"][-1]

        # Grab new observations
        obs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in obs_now.items()} 

        # Extract policy observations - only include keys that the model expects
        model_obs = {}
        for k in ["pose", "zed_rgb", "rs_side_rgb", "dt_left", "dt_right"]:
            if k in obs:
                model_obs[k] = obs[k]
        
        prep_time = time.time() - prep_start_time

        # Save the most recent observation for debugging
        save_data(obs, debug_dir=DEBUG_DIR)

        # Predict an action-horizon batch
        inference_start_time = time.time()
        with torch.no_grad():
            actions = model.predict_action(model_obs)["action"][0].detach().cpu().numpy()
        inference_time = time.time() - inference_start_time

        # q_actions = actions[:action_horizon]          # shape (action_horizon, 10)
        # print actions size
        print(f"Actions size: {actions.shape}")
        q_actions = [actions[1]]

        # Print observation
        log_policy(obs, q_actions, start_time)

        # # Fill the queue
        queue_start_time = time.time()
        for act in q_actions:
            action_queue.put((act, time.monotonic() - start_time))
        queue_time = time.time() - queue_start_time

        time.sleep(.05)


        total_loop_time = time.time() - loop_start_time
        
        print(f"\n{'='*60}")
        print(f"INFERENCE LOOP TIMING (iteration at {time.monotonic() - start_time:.3f}s)")
        print(f"{'='*60}")
        print(f"Wait for new data:     {wait_time*1000:.2f} ms")
        print(f"Data preparation:      {prep_time*1000:.2f} ms") 
        print(f"Model inference:       {inference_time*1000:.2f} ms")
        print(f"Queue actions:         {queue_time*1000:.2f} ms")
        print(f"TOTAL LOOP TIME:       {total_loop_time*1000:.2f} ms")
        print(f"{'='*60}\n")

        # Sleep to allow for execution latency
        # time.sleep(5.0)


def load_diffusion_policy(model_path):
    """Load diffusion policy model from checkpoint"""
    # Load checkpoint
    path = pathlib.Path(model_path)
    payload = torch.load(path.open('rb'), pickle_module=dill, map_location='cpu')

    # Extract configuration
    cfg = payload['cfg']
    print(cfg)

    # Instantiate model
    model = hydra.utils.instantiate(cfg.policy)

    # Load weights
    model.load_state_dict(payload['state_dicts']['model'])
    
    # Move to device and set evaluation mode
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.reset()
    model.eval()
    model.num_inference_steps = 16 #32  # Number of diffusion steps
    
    # Set up noise scheduler
    noise_scheduler = diffusers.schedulers.scheduling_ddim.DDIMScheduler(
        num_train_timesteps=100,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon',
    )
    model.noise_scheduler = noise_scheduler
    
    return model

def load_shape_meta(model_path: str):
    """
    Return the `shape_meta` dict stored in the checkpoint.
    """
    path = pathlib.Path(model_path)
    payload = torch.load(path.open("rb"), pickle_module=dill, map_location="cpu")
    return payload["cfg"].policy.shape_meta      # <-- C,H,W per observation key

def log_policy(obs, actions, start_time, *, max_channels=6):
    def _shape(x):
        return tuple(x.shape) if hasattr(x, "shape") else f"(len={len(x)})"

    bar = "=" * 80
    print(f"\n{bar}\nCYCLE SUMMARY\n{bar}")
    for k, v in obs.items():
        kind = type(v).__name__
        if k.endswith("_timestamps"):
            print(f"{k:<20} {kind:<14} {_shape(v)}")
        elif isinstance(v, (torch.Tensor, np.ndarray)):
            print(f"{k:<20} {kind:<14} {_shape(v)}")
        else:
            print(f"{k:<20} {kind:<14} {v}")

    print(f"\nObservations Passed to Model\n{'-'*80}")

    # Print all observations
    if "pose" in obs and isinstance(obs["pose"], torch.Tensor):
        for ind, curr_obs in enumerate(obs["pose"][0]):
            pose = curr_obs.cpu().numpy()
            pose_timestamp = obs["pose_timestamps"][ind]
            pos, rot6d, grip = pose[:3], pose[3:9], pose[9]
            print(f"Pose {ind + 1}: Position = {pos.round(4)} | Rotation = {rot6d.round(4)} | Gripper = {grip:.4f} | Timestamp = {pose_timestamp}")

    if "zed_rgb" in obs and isinstance(obs["zed_rgb"], torch.Tensor):
        print()
        for ind, curr_obs in enumerate(obs["zed_rgb"][0]):
            zed_rgb_timestamp = obs["zed_rgb_timestamps"][ind]
            print(f"Zed RGB Image {ind + 1}: Shape = {curr_obs.shape} | Timestamp = {zed_rgb_timestamp}")

    if "rs_side_rgb" in obs and isinstance(obs["rs_side_rgb"], torch.Tensor):
        print()
        for ind, curr_obs in enumerate(obs["rs_side_rgb"][0]):
            rs_rgb_timestamp = obs["rs_rgb_timestamps"][ind]
            print(f"RealSense RGB Image {ind + 1}: Shape = {curr_obs.shape} | Timestamp = {rs_rgb_timestamp}")

    if "dt_left" in obs and isinstance(obs["dt_left"], torch.Tensor):
        print()
        for ind, curr_obs in enumerate(obs["dt_left"][0]):
            dt_left_timestamp = obs["dt_left_timestamps"][ind]
            print(f"DT Left Image {ind + 1}: Shape = {curr_obs.shape} | Timestamp = {dt_left_timestamp}")
            
    if "dt_right" in obs and isinstance(obs["dt_right"], torch.Tensor):
        print()
        for ind, curr_obs in enumerate(obs["dt_right"][0]):
            dt_right_timestamp = obs["dt_right_timestamps"][ind]
            print(f"DT Right Image {ind + 1}: Shape = {curr_obs.shape} | Timestamp = {dt_right_timestamp}")

    if actions is not None:
        print(f"\nActions to be published\n{'-'*80}")
        for ind, curr_act in enumerate(actions):
            print(f"Action {ind + 1}: Position = {curr_act[:3].round(4)} | Rotation = {curr_act[3:9].round(4)} | Gripper = {curr_act[9]:.4f} | Timestamp = {time.monotonic() - start_time}")

    print(bar + "\n")

# Create helper method to save images
def save_data(obs_dict=None, debug_dir=Path("../inference/2d_dp_debug")):
    """Save RGB and depth images for debugging"""
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    if obs_dict is None:
        return
        
    # Save the most recent observation (last in the horizon)
    if "rs_side_rgb" in obs_dict and obs_dict["rs_side_rgb"] is not None:
        rs_tensor = obs_dict["rs_side_rgb"]
        if isinstance(rs_tensor, torch.Tensor):
            # Get the most recent frame: [1, T, C, H, W] -> [C, H, W]
            rs_rgb = rs_tensor[0, -1].cpu().numpy()  # Last frame in horizon
            
            # Convert from CHW float32 normalized to HWC uint8 for OpenCV
            rs_img = (rs_rgb * 255).astype(np.uint8).transpose(1, 2, 0)
            # Convert RGB back to BGR for OpenCV
            rs_img = cv2.cvtColor(rs_img, cv2.COLOR_RGB2BGR)

            cv2.imwrite(str(debug_dir/f"rs_side_rgb.jpg"), rs_img)
    
    # Save ZED RGB if available
    if "zed_rgb" in obs_dict and obs_dict["zed_rgb"] is not None:
        zed_tensor = obs_dict["zed_rgb"]
        if isinstance(zed_tensor, torch.Tensor):
            # Get the most recent frame: [1, T, C, H, W] -> [C, H, W]
            zed_rgb = zed_tensor[0, -1].cpu().numpy()  # Last frame in horizon
            
            # Convert from CHW float32 normalized to HWC uint8 for OpenCV
            zed_img = (zed_rgb * 255).astype(np.uint8).transpose(1, 2, 0)
            # Convert RGB back to BGR for OpenCV
            zed_img = cv2.cvtColor(zed_img, cv2.COLOR_RGB2BGR)

            cv2.imwrite(str(debug_dir/f"zed_rgb.jpg"), zed_img)

    if "dt_left" in obs_dict and obs_dict["dt_left"] is not None:
        dt_left_tensor = obs_dict["dt_left"]
        if isinstance(dt_left_tensor, torch.Tensor):
            dt_left_img = dt_left_tensor[0, -1].cpu().numpy()  # (3, H, W)
            dt_left_hwc = np.transpose(dt_left_img, (1, 2, 0))  # now (H, W, 3)
            dt_left_img = (dt_left_hwc * 255).astype(np.uint8)
            cv2.imwrite(str(debug_dir/f"dt_left.jpg"), dt_left_img)

    if "dt_right" in obs_dict and obs_dict["dt_right"] is not None:
        dt_right_tensor = obs_dict["dt_right"]
        if isinstance(dt_right_tensor, torch.Tensor):
            dt_right_img = dt_right_tensor[0, -1].cpu().numpy()  # (3, H, W)
            dt_right_hwc = np.transpose(dt_right_img, (1, 2, 0))  # now (H, W, 3)
            dt_right_img = (dt_right_hwc * 255).astype(np.uint8)
            cv2.imwrite(str(debug_dir/f"dt_right.jpg"), dt_right_img)


def main(args=None):
    np.set_printoptions(suppress=True, precision=4)
    # Initialize multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    rclpy.init(args=args)
    
    # Create shared memory structures
    manager = Manager()
    shared_obs = manager.dict(obs=None, paused=False, exec_done=True)
    action_queue = Queue()
    start_time = time.monotonic()

    # Model path
    model_path = '/home/alex/Documents/3D-Diffusion-Policy/dt_ag/inference/models/vision_only_2000_epochs_bigger.ckpt'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    shape_meta = load_shape_meta(model_path)

    inference_action_horizon = 6

    # Start inference process
    inference_process = Process(
        target=inference_loop, 
        args=(model_path, shared_obs, action_queue, inference_action_horizon, device, start_time)
    )
    inference_process.daemon = True
    inference_process.start()

    # Create ROS node
    node = PolicyNode3D(shared_obs, action_queue, start_time, shape_meta=shape_meta)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Clean up resources
        node.cleanup()
        node.destroy_node()
        rclpy.shutdown()
        inference_process.terminate()


if __name__ == '__main__':
    main()