#!/usr/bin/env python3
"""
Real-time diffusion policy inference for xArm with 3 RealSense cameras.

Two-process architecture:
  Main process  — ROS2 PolicyNode: collects observations, executes actions via xArm SDK
  GPU process   — Loads model, runs predict_action(), schedules actions

Usage:
    python inference.py --model /path/to/checkpoint.ckpt
    python inference.py --model /path/to/checkpoint.ckpt --dt 0.05 --action-horizon 6
"""

import os
import sys
import pathlib
import argparse
import time
import math
import threading
import multiprocessing
from multiprocessing import Process, Manager, Queue

import cv2
import numpy as np
import torch
import dill
import hydra
import pygame

# ── Path setup ────────────────────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_UMI_ROOT = os.path.join(_THIS_DIR, "..", "dt_ag-main")
_UMI_DP = os.path.join(_UMI_ROOT, "universal_manipulation_interface")

for p in [_UMI_DP, _UMI_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

os.chdir(_UMI_DP)

from omegaconf import OmegaConf
OmegaConf.register_new_resolver("eval", eval, use_cache=True)

from diffusion_policy.workspace.base_workspace import BaseWorkspace

# ROS2 imports
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import message_filters
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Bool
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation as R

from dt_ag.rotation_transformer import RotationTransformer

# xArm SDK
from xarm.wrapper import XArmAPI

# ── Constants ─────────────────────────────────────────────────────────────

# ROS topics (observation)
POSE_TOPIC = "robot_obs/pose"
GRIPPER_TOPIC = "robot_obs/gripper"
RS_FRONT_TOPIC = "/rs_front/rs_front/color/image_raw"
RS_WRIST_TOPIC = "/rs_wrist/rs_wrist/color/image_raw"
RS_HEAD_TOPIC = "/rs_head/rs_head/color/image_raw"

# Robot
XARM_IP = "192.168.1.219"
TCP_OFFSET_Z = 172.0  # mm
HOME_POS = (259.1, 2.9, 86.1, -179.5, 0.3, 180.0)  # mm, degrees

# Workspace bounds (meters, with 10mm margin)
WS_X = (0.11, 0.69)
WS_Y = (-0.29, 0.29)
WS_Z = (0.02, 0.39)

# Image processing
IMG_SIZE = 224

# Camera keys matching shape_meta
CAMERA_KEYS = ["rs_front_rgb", "rs_wrist_rgb", "rs_head_rgb"]
CAMERA_TOPICS = {
    "rs_front_rgb": RS_FRONT_TOPIC,
    "rs_wrist_rgb": RS_WRIST_TOPIC,
    "rs_head_rgb": RS_HEAD_TOPIC,
}


# ── Model loading ────────────────────────────────────────────────────────

def load_policy(model_path: str, num_inference_steps: int = 16):
    """Load diffusion policy from checkpoint using workspace pattern."""
    path = pathlib.Path(model_path)
    payload = torch.load(path.open("rb"), pickle_module=dill, map_location="cpu")

    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    policy.num_inference_steps = num_inference_steps

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.eval().to(device)
    print(f"Policy loaded on {device} (EMA={cfg.training.use_ema}, "
          f"diffusion_steps={num_inference_steps})")
    return policy


def load_shape_meta(model_path: str) -> dict:
    """Extract shape_meta from checkpoint."""
    path = pathlib.Path(model_path)
    payload = torch.load(path.open("rb"), pickle_module=dill, map_location="cpu")
    return payload["cfg"].policy.shape_meta


# ── PolicyNode ────────────────────────────────────────────────────────────

class PolicyNode(Node):
    """ROS2 node: collects observations, executes actions via xArm SDK."""

    def __init__(self, shared_obs, action_queue, start_time, model_path: str):
        super().__init__("policy_inference_node")
        np.set_printoptions(suppress=True, precision=4)

        self.shared_obs = shared_obs
        self.action_queue = action_queue
        self.start_time = start_time

        # ── Load shape metadata ──────────────────────────────────────────
        self.shape_meta = load_shape_meta(model_path)
        obs_keys = list(self.shape_meta["obs"].keys())
        self.get_logger().info(f"Policy obs keys: {obs_keys}")

        # ── xArm setup ──────────────────────────────────────────────────
        self.get_logger().info(f"Connecting to xArm at {XARM_IP}...")
        self.arm = XArmAPI(XARM_IP)
        self._setup_xarm()

        # ── Rotation transformer ─────────────────────────────────────────
        # rot6d -> quaternion (for action output)
        self.rot_tf = RotationTransformer("rotation_6d", "quaternion")

        # ── QoS ──────────────────────────────────────────────────────────
        sensor_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
        )

        # ── Subscribers via ApproximateTimeSynchronizer ───────────────────
        self.pose_sub = message_filters.Subscriber(
            self, PoseStamped, POSE_TOPIC, qos_profile=sensor_qos)
        self.gripper_sub = message_filters.Subscriber(
            self, Float32, GRIPPER_TOPIC, qos_profile=sensor_qos)
        self.rs_front_sub = message_filters.Subscriber(
            self, Image, RS_FRONT_TOPIC, qos_profile=sensor_qos)
        self.rs_wrist_sub = message_filters.Subscriber(
            self, Image, RS_WRIST_TOPIC, qos_profile=sensor_qos)
        self.rs_head_sub = message_filters.Subscriber(
            self, Image, RS_HEAD_TOPIC, qos_profile=sensor_qos)

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.pose_sub, self.gripper_sub,
             self.rs_front_sub, self.rs_wrist_sub, self.rs_head_sub],
            queue_size=100,
            slop=0.05,
            allow_headerless=True,
        )
        self.sync.registerCallback(self.synced_obs_callback)

        # ── Observation buffers ──────────────────────────────────────────
        self._bridge = CvBridge()
        self.observation_horizon = 2
        self.gripper_state = 0.0

        self.pose_buffer = []
        self.cam_buffers = {k: [] for k in CAMERA_KEYS}

        # ── State ────────────────────────────────────────────────────────
        self.paused = False
        self.pending_actions = []

        # ── Timers ───────────────────────────────────────────────────────
        self.create_timer(1.0 / 30.0, self.update_observation)
        self.create_timer(1.0 / 100.0, self.timer_callback)

        # ── Initialize gripper ───────────────────────────────────────────
        self.arm.set_gripper_mode(0)
        self.arm.set_gripper_enable(True)
        self.arm.set_gripper_speed(5000)
        self.arm.clean_gripper_error()
        # Jiggle to activate
        self.arm.set_gripper_position(800, wait=False)
        time.sleep(0.3)
        self.arm.set_gripper_position(850, wait=False)

        # ── Reset to home ────────────────────────────────────────────────
        self.reset_xarm()

        self.get_logger().info("PolicyNode ready. Press 'u' to start, 'p' to pause, 'r' to reset.")

    # ── xArm setup ───────────────────────────────────────────────────────

    def _setup_xarm(self):
        """Initialize xArm matching data collection settings."""
        self.arm.clean_error()
        self.arm.clean_warn()
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)
        self.arm.set_state(state=0)
        time.sleep(1)

        self.arm.set_tcp_offset([0, 0, TCP_OFFSET_Z, 0, 0, 0])
        time.sleep(0.1)

        # Safety boundaries (mm)
        self.arm.set_reduced_tcp_boundary([
            WS_X[1] * 1000 + 10, WS_X[0] * 1000 - 10,
            WS_Y[1] * 1000 + 10, WS_Y[0] * 1000 - 10,
            WS_Z[1] * 1000 + 10, WS_Z[0] * 1000 - 10,
        ])
        self.arm.set_fense_mode(True)

        self.arm.set_tcp_maxacc(5000)
        self.arm.set_joint_maxacc(10)
        self.arm.set_reduced_max_tcp_speed(200)
        self.arm.set_reduced_max_joint_speed(60)
        self.arm.set_reduced_mode(True)
        self.servo_active = False

        self.get_logger().info("xArm initialized with safety limits")

    def _switch_to_online_trajectory(self):
        if not self.servo_active:
            self.arm.set_mode(7)
            self.arm.set_state(0)
            time.sleep(0.1)
            self.servo_active = True

    def _switch_to_position_mode(self):
        if self.servo_active:
            self.arm.set_mode(0)
            self.arm.set_state(0)
            time.sleep(0.1)
            self.servo_active = False

    def _recover_from_error(self):
        if self.arm.error_code == 0:
            return False
        self.get_logger().warn(f"xArm error {self.arm.error_code}, recovering...")
        self.arm.clean_error()
        self.arm.clean_warn()
        self.arm.motion_enable(enable=True)
        if self.servo_active:
            self.arm.set_mode(7)
        else:
            self.arm.set_mode(0)
        self.arm.set_state(0)
        time.sleep(0.1)
        return True

    # ── Observation callbacks ────────────────────────────────────────────

    def synced_obs_callback(self, pose_msg, gripper_msg,
                            rs_front_msg, rs_wrist_msg, rs_head_msg):
        """Process synchronized sensor data."""
        now = time.monotonic() - self.start_time
        self.gripper_state = gripper_msg.data

        # ── Pose → 10D ──────────────────────────────────────────────────
        pos = [pose_msg.pose.position.x,
               pose_msg.pose.position.y,
               pose_msg.pose.position.z]
        # ROS quaternion is xyzw, pytorch3d RotationTransformer expects wxyz
        quat_wxyz = [
            pose_msg.pose.orientation.w,
            pose_msg.pose.orientation.x,
            pose_msg.pose.orientation.y,
            pose_msg.pose.orientation.z,
        ]
        rot6d = self.rot_tf.inverse(torch.tensor(quat_wxyz, dtype=torch.float32))
        pose_10d = np.concatenate([pos, rot6d.numpy(), [self.gripper_state]],
                                  axis=None).astype(np.float32)

        self.pose_buffer.append((pose_10d, now))
        if len(self.pose_buffer) > self.observation_horizon:
            self.pose_buffer.pop(0)

        # ── Camera images ────────────────────────────────────────────────
        img_msgs = {
            "rs_front_rgb": rs_front_msg,
            "rs_wrist_rgb": rs_wrist_msg,
            "rs_head_rgb": rs_head_msg,
        }
        for cam_key, msg in img_msgs.items():
            img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))  # HWC → CHW

            buf = self.cam_buffers[cam_key]
            buf.append((img, now))
            if len(buf) > self.observation_horizon:
                buf.pop(0)

    def update_observation(self):
        """Pack observation buffers into shared memory dict."""
        h = self.observation_horizon

        # Check all buffers have enough data
        if len(self.pose_buffer) < h:
            return
        for cam_key in CAMERA_KEYS:
            if len(self.cam_buffers[cam_key]) < h:
                return

        # Pack pose
        pose_slice = self.pose_buffer[-h:]
        pose_np = np.stack([p[0] for p in pose_slice])
        pose_ts = np.array([p[1] for p in pose_slice])

        obs_dict = {
            "pose": torch.from_numpy(pose_np).unsqueeze(0),  # (1, 2, 10)
            "pose_timestamps": pose_ts,
        }

        # Pack cameras
        for cam_key in CAMERA_KEYS:
            cam_slice = self.cam_buffers[cam_key][-h:]
            cam_np = np.stack([c[0] for c in cam_slice])
            cam_ts = np.array([c[1] for c in cam_slice])
            obs_dict[cam_key] = torch.from_numpy(cam_np).unsqueeze(0)  # (1, 2, 3, 224, 224)
            obs_dict[f"{cam_key}_timestamps"] = cam_ts

        self.shared_obs["obs"] = obs_dict

    # ── Action execution ─────────────────────────────────────────────────

    def timer_callback(self):
        """Pop scheduled actions from queue and execute when time arrives."""
        current_time = time.monotonic()

        # Drain queue
        while not self.action_queue.empty():
            item = self.action_queue.get()
            if (isinstance(item, tuple) and len(item) == 2
                    and isinstance(item[0], str) and item[0] == "CLEAR_PENDING"):
                self.pending_actions.clear()
            else:
                try:
                    act, ts = item
                    self.pending_actions.append((act, ts))
                except (ValueError, TypeError):
                    continue

        if not self.pending_actions:
            self.shared_obs["exec_done"] = True
            return

        self.shared_obs["exec_done"] = False

        # Execute actions whose time has arrived
        remaining = []
        for act, ts in self.pending_actions:
            if current_time >= ts and not self.paused:
                self._execute_action(act)
            else:
                remaining.append((act, ts))
        self.pending_actions = remaining
        self.shared_obs["exec_done"] = len(self.pending_actions) == 0

    def _execute_action(self, action_10d: np.ndarray):
        """Send a 10D action to the xArm via SDK."""
        if self.arm.error_code != 0:
            self._recover_from_error()
            return

        self._switch_to_online_trajectory()

        pos = action_10d[:3].copy()  # meters
        rot6d = action_10d[3:9]
        grip = action_10d[9]

        # Clip to workspace bounds
        pos[0] = np.clip(pos[0], WS_X[0], WS_X[1])
        pos[1] = np.clip(pos[1], WS_Y[0], WS_Y[1])
        pos[2] = np.clip(pos[2], WS_Z[0], WS_Z[1])

        # Convert rot6d → quaternion → euler
        quat_wxyz = self.rot_tf.forward(torch.tensor(rot6d, dtype=torch.float32))
        quat_xyzw = [float(quat_wxyz[1]), float(quat_wxyz[2]),
                      float(quat_wxyz[3]), float(quat_wxyz[0])]
        roll, pitch, yaw = R.from_quat(quat_xyzw).as_euler("xyz", degrees=True)

        # Send to arm (convert m → mm)
        self.arm.set_position(
            x=pos[0] * 1000.0, y=pos[1] * 1000.0, z=pos[2] * 1000.0,
            roll=roll, pitch=pitch, yaw=yaw,
            speed=100, is_radian=False, wait=False,
        )

        # Gripper: normalized [0,1] → xArm [850,0]
        grasp = int(850 - 850 * np.clip(grip, 0.0, 1.0))
        self.arm.set_gripper_position(grasp, wait=False)

    # ── Keyboard controls ────────────────────────────────────────────────

    def pause_policy(self):
        self.get_logger().info("Paused.")
        self.paused = True
        self.shared_obs["paused"] = True
        self.pending_actions.clear()
        while not self.action_queue.empty():
            self.action_queue.get_nowait()

    def resume_policy(self):
        self.get_logger().info("Resumed.")
        self.paused = False
        self.shared_obs["paused"] = False

    def reset_xarm(self):
        self.get_logger().info("Resetting to home...")
        self.paused = True
        self.shared_obs["paused"] = True
        self.pending_actions.clear()

        self.arm.clean_error()
        self.arm.clean_warn()
        self.arm.motion_enable(enable=True)
        self._switch_to_position_mode()

        x, y, z, roll, pitch, yaw = HOME_POS
        self.arm.set_position(
            x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw,
            speed=100, is_radian=False, wait=True,
        )
        self.arm.set_gripper_position(850, wait=True)
        self.get_logger().info("Reset complete. Press 'u' to resume.")

    def cleanup(self):
        try:
            self.arm.disconnect()
        except Exception:
            pass


# ── Inference process ─────────────────────────────────────────────────────

def inference_loop(model_path, shared_obs, action_queue,
                   action_horizon=6, device="cuda", start_time=0,
                   dt=0.05, action_exec_latency=0.20,
                   num_inference_steps=16):
    """GPU process: load model and run inference in a loop."""
    policy = load_policy(model_path, num_inference_steps)

    # Get expected observation keys from checkpoint
    path = pathlib.Path(model_path)
    payload = torch.load(path.open("rb"), pickle_module=dill, map_location="cpu")
    model_obs_keys = list(payload["cfg"].policy.shape_meta["obs"].keys())
    print(f"Model expects obs keys: {model_obs_keys}")

    # Wait for first observation
    while shared_obs.get("obs") is None:
        time.sleep(0.05)
        print("Waiting for first observation...")

    # Initialize timestamps
    prev_timestamps = {}
    obs_now = shared_obs["obs"]
    if "pose_timestamps" in obs_now:
        prev_timestamps["pose"] = obs_now["pose_timestamps"][-1]
    for cam_key in CAMERA_KEYS:
        ts_key = f"{cam_key}_timestamps"
        if ts_key in obs_now:
            prev_timestamps[cam_key] = obs_now[ts_key][-1]

    print("Inference loop started.")

    while True:
        # Skip if paused
        if shared_obs.get("paused", False):
            time.sleep(0.05)
            continue

        loop_start = time.time()

        # Wait for new data from all sensors
        wait_start = time.time()
        while True:
            obs_now = shared_obs["obs"]
            all_new = True

            if "pose_timestamps" in obs_now:
                if np.min(obs_now["pose_timestamps"]) <= prev_timestamps.get("pose", -1):
                    all_new = False

            for cam_key in CAMERA_KEYS:
                ts_key = f"{cam_key}_timestamps"
                if ts_key in obs_now:
                    if np.min(obs_now[ts_key]) <= prev_timestamps.get(cam_key, -1):
                        all_new = False

            if all_new:
                break

            elapsed = time.time() - wait_start
            if elapsed > 1.0 and int(elapsed) != int(elapsed - 0.001):
                print(f"Waiting for new sensor data ({elapsed:.1f}s)...")

            time.sleep(0.001)

        wait_time = time.time() - wait_start

        # Update previous timestamps
        if "pose_timestamps" in obs_now:
            prev_timestamps["pose"] = obs_now["pose_timestamps"][-1]
        for cam_key in CAMERA_KEYS:
            ts_key = f"{cam_key}_timestamps"
            if ts_key in obs_now:
                prev_timestamps[cam_key] = obs_now[ts_key][-1]

        # Build model observation dict
        model_obs = {}
        for k in model_obs_keys:
            if k in obs_now:
                v = obs_now[k]
                if isinstance(v, torch.Tensor):
                    model_obs[k] = v.to(device)
                else:
                    model_obs[k] = v

        # Run inference
        inference_start = time.time()
        with torch.no_grad():
            actions = policy.predict_action(model_obs)["action"][0].detach().cpu().numpy()
        inference_time = time.time() - inference_start

        # Take first N actions
        q_actions = actions[:action_horizon]

        # Schedule actions
        t_start = time.monotonic()
        action_timestamps = np.array([t_start + (i + 1) * dt for i in range(len(q_actions))])
        current_time = time.monotonic()

        # Clear pending
        action_queue.put(("CLEAR_PENDING", current_time))

        # Filter by latency
        valid_idx = action_timestamps > (current_time + action_exec_latency)
        if np.sum(valid_idx) == 0:
            next_step = int(np.ceil((current_time - t_start) / dt))
            action_timestamps = np.array([t_start + next_step * dt])
            q_actions = q_actions[-1:]
        else:
            q_actions = q_actions[valid_idx]
            action_start = current_time
            action_timestamps = np.array(
                [action_start + i * dt for i in range(len(q_actions))]
            )

        for act, ts in zip(q_actions, action_timestamps):
            action_queue.put((act, ts))

        total_time = time.time() - loop_start
        print(f"Inference: {inference_time*1000:.0f}ms | "
              f"Wait: {wait_time*1000:.0f}ms | "
              f"Total: {total_time*1000:.0f}ms | "
              f"Actions: {len(q_actions)}")

        time.sleep(0.05)


# ── Pygame keyboard monitor ──────────────────────────────────────────────

def monitor_keys(policy_node, shared_obs):
    """Background thread: listen for keystrokes and display status."""
    try:
        pygame.init()
        screen = pygame.display.set_mode((320, 200))
        pygame.display.set_caption("xArm Policy Control")
        clock = pygame.time.Clock()
        font = pygame.font.SysFont("monospace", 18)
        font_small = pygame.font.SysFont("monospace", 14)

        COLOR_PAUSED = (50, 50, 60)
        COLOR_RUNNING = (20, 60, 30)

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("Shutting down...")
                    policy_node.pause_policy()
                    os._exit(0)
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_d:
                        policy_node.pause_policy()
                    elif event.key == pygame.K_s:
                        policy_node.resume_policy()
                    elif event.key == pygame.K_r:
                        policy_node.reset_xarm()
                    elif event.key == pygame.K_q:
                        print("Quit requested. Shutting down...")
                        policy_node.pause_policy()
                        time.sleep(0.2)
                        os._exit(0)

            # Draw status
            paused = shared_obs.get("paused", True)
            screen.fill(COLOR_PAUSED if paused else COLOR_RUNNING)

            status_text = "PAUSED" if paused else "RUNNING"
            status_color = (255, 200, 50) if paused else (50, 255, 80)
            text_surf = font.render(status_text, True, status_color)
            screen.blit(text_surf, (110, 20))

            keys_info = [
                ("S", "Start / Resume"),
                ("D", "Done / Pause"),
                ("R", "Reset home"),
                ("Q", "Quit"),
            ]
            for i, (key, desc) in enumerate(keys_info):
                line = font_small.render(f"  {key}  -  {desc}", True, (200, 200, 200))
                screen.blit(line, (30, 60 + i * 28))

            pygame.display.flip()
            clock.tick(10)
    except Exception as e:
        print(f"Pygame error: {e}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Diffusion Policy Inference on xArm")
    parser.add_argument("--model", type=str, required=True, help="Path to .ckpt file")
    parser.add_argument("--dt", type=float, default=0.05, help="Action period (seconds)")
    parser.add_argument("--action-horizon", type=int, default=6,
                        help="Number of actions to execute per inference cycle")
    parser.add_argument("--latency", type=float, default=0.20,
                        help="Action execution latency margin (seconds)")
    parser.add_argument("--diffusion-steps", type=int, default=16,
                        help="DDIM inference steps")
    parser.add_argument("--no-pygame", action="store_true",
                        help="Disable pygame keyboard controls")
    args = parser.parse_args()

    multiprocessing.set_start_method("spawn", force=True)
    rclpy.init()

    manager = Manager()
    shared_obs = manager.dict(obs=None, paused=True, exec_done=True)
    action_queue = Queue()
    start_time = time.monotonic()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Model: {args.model}")
    print(f"dt={args.dt}s ({1/args.dt:.0f}Hz) | "
          f"action_horizon={args.action_horizon} | "
          f"latency={args.latency}s | "
          f"diffusion_steps={args.diffusion_steps}")

    # Start inference process
    inference_proc = Process(
        target=inference_loop,
        args=(args.model, shared_obs, action_queue,
              args.action_horizon, device, start_time,
              args.dt, args.latency, args.diffusion_steps),
    )
    inference_proc.daemon = True
    inference_proc.start()

    # Create ROS2 node
    node = PolicyNode(shared_obs, action_queue, start_time, args.model)

    # Start pygame thread
    if not args.no_pygame:
        key_thread = threading.Thread(target=monitor_keys, args=(node, shared_obs), daemon=True)
        key_thread.start()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cleanup()
        node.destroy_node()
        rclpy.shutdown()
        inference_proc.terminate()


if __name__ == "__main__":
    main()
