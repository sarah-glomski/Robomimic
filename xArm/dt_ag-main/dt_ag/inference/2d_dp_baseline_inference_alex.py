#!/usr/bin/env python3
import sys, pathlib, os

# inference script location: ~/Documents/3D-Diffusion-Policy/dt_ag/inference
# go up two levels to ~/Documents/3D-Diffusion-Policy, then into the '3D-Diffusion-Policy' subfolder

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
DEBUG_DIR = SCRIPT_DIR / "2d_dp_debug_baseline_alex"
LOCAL_ROOT = pathlib.Path(__file__).resolve().parents[2] / "3D-Diffusion-Policy"
assert LOCAL_ROOT.is_dir(), f"Can't find code at {LOCAL_ROOT}"
sys.path.insert(0, str(LOCAL_ROOT))
os.chdir(str(LOCAL_ROOT))

# now imports will pick up your current repo
import diffusion_policy
print("Loading diffusion_policy from:", diffusion_policy.__file__)

import sys
import rclpy
from rclpy.node import Node
import threading
import pathlib
import numpy as np
import torch
import time

# Multiprocessing imports
import multiprocessing
from multiprocessing import Process, Manager, Queue

from utils.inference_utils import InferenceUtils, monitor_keys, load_policy, save_data, log_policy, log_inference_timing

# ----------------------------------------------------------------------
# Policy node
# ----------------------------------------------------------------------
class PolicyNode3D(InferenceUtils, Node):
    def __init__(self, shared_obs, action_queue, start_time, model_path: str):
        super().__init__('Policy_Node')
        self.init_policy_node(model_path, shared_obs, action_queue, start_time, sync_queue_size=100, sync_slop=0.05, inference_rate=30)

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

        while self.pending_actions and not self.paused:

            action = self.pending_actions.pop(0)
            ee_msg, grip_msg = self.generate_ee_action_msg(action)
            self.pub_robot_pose.publish(ee_msg)
            self.gripper_pub.publish(grip_msg)

            time.sleep(0.05)
        
        self.shared_obs["exec_done"] = True

# # ----------------------------------------------------------------------
# # Inference loop
# # ----------------------------------------------------------------------

def inference_loop(model_path, shared_obs, action_queue, action_horizon = 4, device = "cuda", start_time = 0):

    model = load_policy(model_path)
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
        for k in ["pose", "zed_rgb", "rs_side_rgb"]:
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


        # t_start = time.monotonic()
        # dt = 1/5.0
        # action_timestamps = np.array([t_start + (i+1) * dt for i in range(len(q_actions))])
        # current_time = time.monotonic()

        # action_exec_latency = 0.1 

        # env_actions = q_actions.copy()

        # # Filter out any actions that would be executed too soon.
        # valid_idx = action_timestamps > (current_time + action_exec_latency)
        # if np.sum(valid_idx) == 0:
        #     # If no actions remain valid, use the last action and schedule it for the next slot.
        #     next_step_idx = int(np.ceil((current_time - t_start) / dt))
        #     action_timestamps = np.array([t_start + next_step_idx * dt])
        #     env_actions = env_actions[-1:]
        #     print("No actions remain valid, using the last action and scheduling it for the next slot.")
        # else:
        #     env_actions = env_actions[valid_idx]
        #     action_timestamps = action_timestamps[valid_idx]

        # # let's first empty the action queue
        # while not action_queue.empty():
        #     print("Emptying action queue")
        #     action_queue.get()

        # for act, ts in zip(env_actions, action_timestamps):
        #     action_queue.put((act, ts))

        # queue_time = time.time() - t_start

        # # Fill the queue
        queue_start_time = time.time()
        for act in q_actions:
            action_queue.put((act, time.monotonic() - start_time))
        queue_time = time.time() - queue_start_time

        time.sleep(1.0)


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

def main(args=None):
    """Main function to initialize the policy node and start the inference loop."""

    # Initialize multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    rclpy.init(args=args)

    use_pygame = False

    if use_pygame:
        key_thread = threading.Thread(target=monitor_keys, args=(node,), daemon=True)
        key_thread.start()
    
    # Create shared memory structures
    manager = Manager()
    shared_obs = manager.dict(obs=None, paused=False, exec_done=True)
    action_queue = Queue()
    start_time = time.monotonic()

    # Model path
    model_path = '/home/alex/Documents/3D-Diffusion-Policy/dt_ag/inference/models/one_demo_1700.ckpt'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    inference_action_horizon = 2

    # Start inference process
    inference_process = Process(
        target=inference_loop, 
        args=(model_path, shared_obs, action_queue, inference_action_horizon, device, start_time)
    )
    inference_process.daemon = True
    inference_process.start()

    # Create ROS node
    node = PolicyNode3D(shared_obs, action_queue, start_time, model_path)

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