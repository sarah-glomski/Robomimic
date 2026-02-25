#!/usr/bin/env python3
"""
Launch file for diffusion policy inference on xArm.

Starts all required nodes:
  - 3 RealSense cameras (rs_front, rs_wrist, rs_head)
  - xarm_state_publisher (publishes robot state)
  - inference.py (runs policy and controls robot)

Usage:
    python launch_inference.py --model /path/to/checkpoint.ckpt
    python launch_inference.py --model /path/to/checkpoint.ckpt --dt 0.1 --action-horizon 4
"""

import sys
import os
import argparse
from launch import LaunchService, LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node


def generate_launch_description(model_path, dt=0.05, action_horizon=6,
                                latency=0.20, diffusion_steps=16,
                                no_pygame=False):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_collection_dir = os.path.join(script_dir, "..", "data_collection")

    state_publisher = os.path.join(data_collection_dir, "xarm_state_publisher.py")
    inference_script = os.path.join(script_dir, "inference.py")

    # Build inference command
    inference_cmd = [
        "python3", inference_script,
        "--model", model_path,
        "--dt", str(dt),
        "--action-horizon", str(action_horizon),
        "--latency", str(latency),
        "--diffusion-steps", str(diffusion_steps),
    ]
    if no_pygame:
        inference_cmd.append("--no-pygame")

    return LaunchDescription([
        # RealSense Camera: Front view
        Node(
            package="realsense2_camera",
            executable="realsense2_camera_node",
            name="rs_front",
            namespace="rs_front",
            output="screen",
            parameters=[{
                "serial_no": "244222071219",
                "camera_name": "rs_front",
                "enable_color": True,
                "enable_depth": False,
                "enable_infra1": False,
                "enable_infra2": False,
                "enable_gyro": False,
                "enable_accel": False,
                "rgb_camera.color_profile": "640x360x30",
            }],
        ),

        # RealSense Camera: Wrist-mounted
        Node(
            package="realsense2_camera",
            executable="realsense2_camera_node",
            name="rs_wrist",
            namespace="rs_wrist",
            output="screen",
            parameters=[{
                "serial_no": "317222072257",
                "camera_name": "rs_wrist",
                "enable_color": True,
                "enable_depth": False,
                "enable_infra1": False,
                "enable_infra2": False,
                "enable_gyro": False,
                "enable_accel": False,
                "rgb_camera.color_profile": "640x360x30",
            }],
        ),

        # RealSense Camera: Head-mounted
        Node(
            package="realsense2_camera",
            executable="realsense2_camera_node",
            name="rs_head",
            namespace="rs_head",
            output="screen",
            parameters=[{
                "serial_no": "845112071112",
                "camera_name": "rs_head",
                "enable_color": True,
                "enable_depth": False,
                "enable_infra1": False,
                "enable_infra2": False,
                "enable_gyro": False,
                "enable_accel": False,
                "rgb_camera.color_profile": "640x360x30",
            }],
        ),

        # XArm State Publisher
        ExecuteProcess(
            cmd=["python3", state_publisher],
            name="xarm_state_publisher",
            output="screen",
        ),

        # Policy Inference (controls robot directly via xArm SDK)
        ExecuteProcess(
            cmd=inference_cmd,
            name="policy_inference",
            output="screen",
        ),
    ])


def main(argv=sys.argv[1:]):
    parser = argparse.ArgumentParser(description="Launch diffusion policy inference")
    parser.add_argument("--model", type=str, required=True, help="Path to .ckpt file")
    parser.add_argument("--dt", type=float, default=0.05, help="Action period (seconds)")
    parser.add_argument("--action-horizon", type=int, default=6,
                        help="Actions to execute per inference cycle")
    parser.add_argument("--latency", type=float, default=0.20,
                        help="Action execution latency margin (seconds)")
    parser.add_argument("--diffusion-steps", type=int, default=16,
                        help="DDIM inference steps")
    parser.add_argument("--no-pygame", action="store_true",
                        help="Disable pygame keyboard controls")
    args, launch_argv = parser.parse_known_args(argv)

    print("=" * 60)
    print("Diffusion Policy Inference - xArm")
    print("=" * 60)
    print(f"  Model:            {args.model}")
    print(f"  dt:               {args.dt}s ({1/args.dt:.0f} Hz)")
    print(f"  Action horizon:   {args.action_horizon}")
    print(f"  Latency margin:   {args.latency}s")
    print(f"  Diffusion steps:  {args.diffusion_steps}")
    print()
    print("Keyboard controls (focus pygame window):")
    print("  p = Pause | u = Resume | r = Reset to home")
    print("=" * 60)

    ld = generate_launch_description(
        model_path=args.model,
        dt=args.dt,
        action_horizon=args.action_horizon,
        latency=args.latency,
        diffusion_steps=args.diffusion_steps,
        no_pygame=args.no_pygame,
    )
    ls = LaunchService(argv=launch_argv)
    ls.include_launch_description(ld)
    return ls.run()


if __name__ == "__main__":
    sys.exit(main())
