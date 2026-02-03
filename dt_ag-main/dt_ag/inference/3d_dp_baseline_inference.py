#!/usr/bin/env python3
"""
Enhanced Policy Node with Point Cloud Generation
===============================================

This script combines the 3D diffusion policy inference with
Grounded SAM-2 for point cloud generation from RGB-D images.
"""

import sys
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32, Bool
import threading
import pathlib
import pygame
from pathlib import Path
import numpy as np
import torch
from cv_bridge import CvBridge
import cv2
import time
import diffusers
import hydra
import dill

from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

# Custom gsam utilities
from custom_gsam_utils import GroundedSAM2, default_gsam_config

# Multiprocessing imports
import multiprocessing
from multiprocessing import Process, Manager, Queue
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import message_filters

from visualizer.pointcloud import Visualizer

gendp_path = '/home/alex/Documents/DT-Diffusion-Policy/gendp/gendp'

if gendp_path not in sys.path:
    sys.path.append(gendp_path)

from gendp.policy.diffusion_unet_hybrid_image_densetact_policy import DiffusionUnetHybridImageDenseTactPolicy as gendp
from gendp.model.common.rotation_transformer import RotationTransformer


class PolicyNode3D(Node):
    def __init__(self, shared_obs, action_queue, gsam_config, start_time):
        super().__init__('Policy_Node')

        # --- QoS tuned for high-rate image streams ---
        sensor_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)

        # Subscribers
        self.pose_sub = message_filters.Subscriber(self, PoseStamped, '/robot_pose', qos_profile=sensor_qos)
        self.zed_rgb_sub = message_filters.Subscriber(self, Image, '/zed_image/rgb', qos_profile=sensor_qos)
        self.zed_depth_sub = message_filters.Subscriber(self, Image, '/zed_image/depth', qos_profile=sensor_qos)
        self.gripper_sub = message_filters.Subscriber(self, Float32, '/gripper_state', qos_profile=sensor_qos)

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.pose_sub, self.zed_rgb_sub, self.zed_depth_sub, self.gripper_sub], 
            queue_size=30,         # how many "unmatched" msgs to keep
            slop=0.3,
            allow_headerless=True)  # Allow messages without headers
        
        self.sync.registerCallback(self.synced_obs_callback)
        
        # Publishers
        self.gripper_pub = self.create_publisher(Float32, '/gripper_position', 1)
        self.reset_xarm_pub = self.create_publisher(Bool, '/reset_xarm', 1)
        self.pause_xarm_pub = self.create_publisher(Bool, '/pause_xarm', 1)
        self.pub_robot_pose = self.create_publisher(PoseStamped, '/xarm_position', 1)
        
        self._bridge = CvBridge()

        # Shared data and action queue
        self.shared_obs = shared_obs
        self.action_queue = action_queue
        self.dt = 0.05  
        self.start_time = start_time
        
        # ─────── Visualization ────────
        self.visualizer = Visualizer(camera_position='isometric')
        t = threading.Thread(
            target=self.visualizer.run,
            kwargs={'host': '0.0.0.0', 'port': 8050}, 
            daemon=True
        )
        t.start()
        self.visualizer_update_counter = 0
        self.get_logger().info("PointCloud visualizer running on http://localhost:8050")

        # Timers
        self.create_timer(self.dt, self.timer_callback)
        self.create_timer(self.dt, self.update_observation)

        # Horizon for keeping recent observations
        self.observation_horizon = 2
        
        # Buffers for observations
        self.pose_buffer = []
        self.pcd_buffer = []
        
        # Rotation transformer
        self.tf = RotationTransformer('rotation_6d', 'quaternion')

        # Control state
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.paused = False
        self.even = True
        self.pending_actions = []
        
        # ------------  Build GSAM‑2 helper -----------------------
        self.gsam = GroundedSAM2(cfg=gsam_config)
        self.zed_cam_intrinsics = gsam_config["zed_intri"]
        self.T_base_cam = gsam_config["T_base_cam"]
        self.pts_in_pcd = gsam_config["pts_per_pcd"]
        self.gsam_init = False

        # Jiggle gripper to activate it
        msg = Float32()
        msg.data = 0.05
        self.gripper_pub.publish(msg)
        time.sleep(0.5)
        msg = Float32()
        msg.data = 0.0
        self.gripper_pub.publish(msg)
        self.gripper_state = 0.0

        # Reset robot to home position
        self.reset_xarm_pub.publish(Bool(data=True))
        
        self.get_logger().info("3D Diffusion Policy Node for Baseline Gripper Initialized!")

    def reset_xarm(self):
        """Reset robot to home position"""
        self.get_logger().info("Reset xarm.")
        ee_pose = PoseStamped()
        ee_pose.header.stamp = self.get_clock().now().to_msg()
        ee_pose.pose.position.x = float(0.1668)
        ee_pose.pose.position.y = float(0.0018)
        ee_pose.pose.position.z = float(0.2448)
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
        
        # Reset GSAM tracking
        self.gsam_init = False

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
        
        # Reset GSAM tracking to get fresh object detection
        self.gsam.reset()
        self.gsam_init = False
        
        self.paused = False
        self.shared_obs['paused'] = False         # let inference run

    def cleanup(self):
        """Clean up resources"""
        self.gsam.cleanup()
        pygame.quit()

    # Create helper method to save images
    def save_data(self, zed_rgb_msg, zed_depth_msg, debug_mask, boxes_np, debug_dir=Path("debug_3d_dp")):
        """Save data for debugging"""
        debug_dir.mkdir(parents=True, exist_ok=True)

        # 1. RGB
        if isinstance(zed_rgb_msg, np.ndarray):
            rgb_bgr = zed_rgb_msg
        else:
            rgb_bgr = self._bridge.imgmsg_to_cv2(zed_rgb_msg, desired_encoding='bgr8')
        cv2.imwrite(str(debug_dir / "zed_rgb.jpg"), rgb_bgr)

        # 2. Depth → colormap
        if zed_depth_msg is not None:
            if isinstance(zed_depth_msg, np.ndarray):
                depth = zed_depth_msg
            else:
                depth = self._bridge.imgmsg_to_cv2(zed_depth_msg, desired_encoding='32FC1')
            # clip to a reasonable range (e.g. 0–3 m) and normalize
            clipped = np.clip(depth, 0.0, 3.0)
            disp = (clipped / 3.0 * 255).astype(np.uint8)
            depth_color = cv2.applyColorMap(disp, cv2.COLORMAP_JET)
            cv2.imwrite(str(debug_dir / "zed_depth.jpg"), depth_color)

        # 3. Boxes annotated
        if boxes_np is not None and len(boxes_np):
            # assume boxes_np is Nx4 or Nx5 (x1,y1,x2,y2[,score])
            ann = rgb_bgr.copy()
            for bb in boxes_np:
                x1, y1, x2, y2 = bb[:4].astype(int)
                cv2.rectangle(ann, (x1, y1), (x2, y2), (0,255,0), 2)
                if bb.shape[-1] > 4:
                    score = bb[4]
                    cv2.putText(ann, f"{score:.2f}", (x1, y1-4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            cv2.imwrite(str(debug_dir / "boxes_annotated.jpg"), ann)

        # 4. Fused mask
        if debug_mask is not None:
            # assume mask is HxW float in [0,1] or uint8 0/255
            if debug_mask.dtype != np.uint8:
                m = (debug_mask * 255).astype(np.uint8)
            else:
                m = debug_mask
            # overlay mask onto gray background
            colored = cv2.applyColorMap(m, cv2.COLORMAP_HOT)
            cv2.imwrite(str(debug_dir / "debug_mask.jpg"), colored)

    def synced_obs_callback(self, pose_msg, rgb_msg, depth_msg, gripper_msg):
        """Process synchronized observations and generate point cloud."""

        # ─── 1 update scalar state ──────────────────────────────────────────────
        self.gripper_state = gripper_msg.data
        self.pose_callback(pose_msg)

        # ─── 2 convert ROS images -> numpy ──────────────────────────────────────
        rgb_img = self._bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
        depth_img = self._bridge.imgmsg_to_cv2(depth_msg, desired_encoding="32FC1")

        # ─── 3 Grounding-DINO detections ────────────────────────────────────────
        boxes_np, found, _ = self.gsam.detect_boxes(
            rgb_img, self.gsam.cfg["queries"],
            self.gsam.cfg["box_thresh"], self.gsam.cfg["text_thresh"]
        )
        if not found:
            self.get_logger().warning("No boxes found; skipping frame.")
            return

        # ─── 4 mask tracking  (init → weighted; else → weighted propagate) ─────
        if not self.gsam_init:
            masks_dict, _ = self.gsam.initialize_tracking_weighted(rgb_img, boxes_np)
            self.gsam_init = True
        else:
            masks_dict, _ = self.gsam.propagate_masks_weighted(rgb_img)

        # ─── 5 (optional) visual debug storage ─────────────────────────────────
        # debug_mask = self.gsam.fuse_masks(masks_dict)
        # self.save_data(rgb_img, depth_img, debug_mask, boxes_np)

        # ─── 6 build the weighted scene point-cloud ────────────────────────────
        pcd_np = self.gsam.weighted_pcd(rgb_img, depth_img, masks_dict, self.gsam.cfg["pts_per_pcd"])

        # ─── 7 buffer, visualise, ship to inference ────────────────────────────
        self.pcd_buffer.append((pcd_np, time.monotonic() - self.start_time))
        if len(self.pcd_buffer) > self.observation_horizon:
            self.pcd_buffer.pop(0)

        self.visualizer.update_pcd(pcd_np)

    def update_observation(self):
        """Update shared observation dictionary for inference process"""

        # Check if we have enough data
        min_length = min(len(self.pose_buffer), len(self.pcd_buffer))

        # self.get_logger().info(f"Pose buffer length: {len(self.pose_buffer)} | PCD buffer length: {len(self.pcd_buffer)}")

        if min_length >= self.observation_horizon:

            # Process pose data and extract timestamps
            pose_data = self.pose_buffer[-self.observation_horizon:]
            pose_np = np.stack([pose_obs_tuple[0] for pose_obs_tuple in pose_data])
            pose_timestamps = np.array([pose_obs_tuple[1] for pose_obs_tuple in pose_data])
            pose_tensor = torch.from_numpy(pose_np).unsqueeze(0)
            
            # Process point cloud data and extract timestamps
            pcd_data = self.pcd_buffer[-self.observation_horizon:]
            pcd_stack = np.stack([pcd_obs_tuple[0] for pcd_obs_tuple in pcd_data])
            pcd_timestamps = np.array([pcd_obs_tuple[1] for pcd_obs_tuple in pcd_data])
            pcd_transposed = pcd_stack.transpose(0, 2, 1)
            pcd_tensor = torch.from_numpy(pcd_transposed).unsqueeze(0)
            
            # Create observation dictionary with timestamps
            obs_dict = {
                "pose": pose_tensor,
                "d3fields": pcd_tensor,
                "pose_timestamps": pose_timestamps,
                "pcd_timestamps": pcd_timestamps
            }
            
            # Update shared observation
            self.shared_obs["obs"] = obs_dict

    def timer_callback(self):
        """
        Publish exactly ONE action per dt and keep the pending-action counter in sync.
        """
        # Pull freshly queued actions into the local list
        while not self.action_queue.empty():
            self.pending_actions.append(self.action_queue.get())

        if not self.pending_actions:
            self.shared_obs["exec_done"] = True
            return                         

        # Publish the oldest action
        action = self.pending_actions.pop(0)

        # Create action messages
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

    def pose_callback(self, msg):
        """Process robot pose"""
        robot_pos = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z] 

        # tf.inverse expects [wxyz]
        robot_ori = [msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z] 
        
        # Convert to 6D rotation representation
        robot_ori_tensor = torch.tensor(robot_ori, dtype=torch.float32)
        robot_ori_6d = self.tf.inverse(robot_ori_tensor)

        # Combine position, orientation
        robot_pose = np.concatenate([robot_pos, robot_ori_6d.numpy(), self.gripper_state], axis=None)
        self.pose_buffer.append((robot_pose, time.monotonic() - self.start_time))
        if len(self.pose_buffer) > self.observation_horizon:
            self.pose_buffer.pop(0)


def inference_loop(shared_obs, action_queue, model_path, action_horizon = 4, device = "cuda", start_time = 0):

    model = load_diffusion_policy(model_path)
    print("Inference process started.")

    # ─── Wait until first observation ───────────────────────────────
    while shared_obs.get("obs") is None:
        time.sleep(0.05)

    prev_pose_latest = shared_obs["obs"]["pose_timestamps"][-1]
    prev_pcd_latest  = shared_obs["obs"]["pcd_timestamps"][-1]

    while True:
        # If user paused with the keyboard, spin-wait cheaply
        if shared_obs.get("paused", False):
            time.sleep(0.05)
            continue

        # Busy‐wait until *all* timestamps have advanced
        while True:
            obs_now = shared_obs["obs"]
            pose_new = np.min(obs_now["pose_timestamps"]) > prev_pose_latest
            pcd_new = np.min(obs_now["pcd_timestamps"]) > prev_pcd_latest

            if pose_new and pcd_new and shared_obs["exec_done"]:
                break
            time.sleep(0.001)

        # Save the new arrays
        prev_pose_latest = obs_now["pose_timestamps"][-1]
        prev_pcd_latest = obs_now["pcd_timestamps"][-1]

        obs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in obs_now.items()}          

        # Extract pose and point cloud
        model_obs = {k: obs[k] for k in ("pose", "d3fields")}

        # Predict an action-horizon batch
        with torch.no_grad():
            actions = model.predict_action(model_obs)["action"][0].cpu().numpy()

        q_actions = actions[:action_horizon]          # shape (action_horizon, 10)

        # Print observation
        detailed_debug_summary(obs, q_actions, start_time)

        # Fill the queue
        for act in q_actions:
            action_queue.put(act)

        shared_obs["exec_done"] = False

        # Sleep to allow for execution latency
        time.sleep(0.75)

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
    model.num_inference_steps = 32  # Number of diffusion steps
    
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

def monitor_keys(policy_node):
    """Monitor keyboard input for robot control"""
    while True:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN: 
                if event.key == pygame.K_p:
                    policy_node.pause_policy()
                elif event.key == pygame.K_u:
                    policy_node.resume_policy()
                elif event.key == pygame.K_r:
                    policy_node.reset_xarm()
        time.sleep(0.01)

def detailed_debug_summary(obs, actions, start_time, *, max_channels=6):
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
            print(f"Pose {ind + 1}: Position = {pos.round(4)} | Gripper = {grip:.4f} | Timestamp = {pose_timestamp}")

    if "d3fields" in obs and isinstance(obs["d3fields"], torch.Tensor):
        print()
        for ind, curr_obs in enumerate(obs["d3fields"][0]):
            pcd_timestamp = obs["pcd_timestamps"][ind]
            print(f"Point Cloud {ind + 1}: Shape = {curr_obs.shape} | Timestamp = {pcd_timestamp}")

    if actions is not None:
        print(f"\nActions to be published\n{'-'*80}")
        for ind, curr_act in enumerate(actions):
            print(f"Action {ind + 1}: Position = {curr_act[:3].round(4)} | Gripper = {curr_act[9]:.4f} | Timestamp = {time.monotonic() - start_time}")

    print(bar + "\n")

def main(args=None):
    # Initialize multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    rclpy.init(args=args)
    
    # Initialize Pygame for keyboard control
    pygame.init()
    pygame.display.set_mode((300, 200))
    
    # Create shared memory structures
    manager = Manager()
    shared_obs = manager.dict(obs=None, paused=False, exec_done=True)
    action_queue = Queue()

    # Model path
    model_path = '/home/alex/Documents/3D-Diffusion-Policy/dt_ag/inference/models/2048_pts_more_fruit_pts.ckpt'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    start_time = time.monotonic()

    # Load GSAM configuration
    gsam_config = default_gsam_config()

    # Start inference process
    inference_process = Process(
        target=inference_loop, 
        args=(shared_obs, action_queue, model_path, 2, device, start_time)
    )
    inference_process.daemon = True
    inference_process.start()

    # Create ROS node
    node = PolicyNode3D(shared_obs, action_queue, gsam_config, start_time)

    # Start key monitoring thread
    key_thread = threading.Thread(target=monitor_keys, args=(node,), daemon=True)
    key_thread.start()

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