#!/usr/bin/env python3
"""
Launch file for Simulation Mode (no robot connection)

Launches cameras, hand tracker, and viser visualizer only.
No robot nodes or data collector — just visualize hand tracking
and trajectory goals in the viser 3D scene.

Usage:
    python3 launch_sim.py
"""

import sys
import os
from launch import LaunchService
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node


def generate_launch_description():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    hand_tracker = os.path.join(script_dir, 'mediapipe_hand_tracker.py')
    frame_visualizer = os.path.join(script_dir, 'viser_frame_visualizer.py')

    return LaunchDescription([
        # MediaPipe Hand Tracker
        ExecuteProcess(
            cmd=['python3', hand_tracker],
            name='mediapipe_hand_tracker',
            output='screen'
        ),

        # Viser 3D Frame Visualizer (web-based at http://localhost:8080)
        ExecuteProcess(
            cmd=['python3', frame_visualizer],
            name='viser_frame_visualizer',
            output='screen'
        ),

        # RealSense Camera 1: Front view
        Node(
            package='realsense2_camera',
            executable='realsense2_camera_node',
            name='rs_front',
            namespace='rs_front',
            output='screen',
            parameters=[{
                'serial_no': '317222072257',
                'camera_name': 'rs_front',
                'enable_color': True,
                'enable_depth': False,
                'enable_infra1': False,
                'enable_infra2': False,
                'enable_gyro': False,
                'enable_accel': False,
                'rgb_camera.color_profile': '640x360x30',
            }]
        ),

        # RealSense Camera 2: Head-mounted camera (used for hand tracking)
        Node(
            package='realsense2_camera',
            executable='realsense2_camera_node',
            name='rs_head',
            namespace='rs_head',
            output='screen',
            parameters=[{
                'serial_no': '845112071112',
                'camera_name': 'rs_head',
                'enable_color': True,
                'enable_depth': False,
                'enable_infra1': False,
                'enable_infra2': False,
                'enable_gyro': False,
                'enable_accel': False,
                'rgb_camera.color_profile': '640x360x30',
            }]
        ),
    ])


def main(argv=sys.argv[1:]):
    print("=" * 60)
    print("Simulation Mode - Hand Tracking Visualization")
    print("=" * 60)
    print()
    print("No robot connection required.")
    print("Cameras: front + head (wrist skipped)")
    print()
    print("3D Visualization:")
    print("  Open browser to: http://localhost:8080")
    print()
    print("=" * 60)

    ld = generate_launch_description()
    ls = LaunchService(argv=argv)
    ls.include_launch_description(ld)
    return ls.run()


if __name__ == '__main__':
    sys.exit(main())
