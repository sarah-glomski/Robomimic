#!/usr/bin/env python3
"""
Launch file for MediaPipe Hand Tracking Data Collection System

Launches:
- 2 RealSense cameras (rs_front for observation, rs_hand for hand tracking)
- xarm_state_publisher: Publishes robot state
- mediapipe_hand_tracker: Detects hand and publishes pose
- xarm_hand_controller: Controls robot based on hand tracking
- hdf5_data_collector: Collects synchronized data with pygame keyboard control
- viser_frame_visualizer: 3D web-based visualization (http://localhost:8080)

Usage:
    python3 launch_data_collection.py

Note: Update the RealSense serial numbers to match your cameras.
"""

import sys
import os
from launch import LaunchService
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node


def generate_launch_description():
    # Get the directory containing this launch file
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Python scripts to launch
    state_publisher = os.path.join(script_dir, 'xarm_state_publisher.py')
    hand_tracker = os.path.join(script_dir, 'mediapipe_hand_tracker.py')
    hand_controller = os.path.join(script_dir, 'xarm_hand_controller.py')
    data_collector = os.path.join(script_dir, 'hdf5_data_collector.py')
    frame_visualizer = os.path.join(script_dir, 'viser_frame_visualizer.py')

    return LaunchDescription([
        # XArm State Publisher
        ExecuteProcess(
            cmd=['python3', state_publisher],
            name='xarm_state_publisher',
            output='screen'
        ),

        # MediaPipe Hand Tracker
        ExecuteProcess(
            cmd=['python3', hand_tracker],
            name='mediapipe_hand_tracker',
            output='screen'
        ),

        # # XArm Hand Controller
        # ExecuteProcess(
        #     cmd=['python3', hand_controller],
        #     name='xarm_hand_controller',
        #     output='screen'
        # ),

        # HDF5 Data Collector (runs pygame in main thread)
        ExecuteProcess(
            cmd=['python3', data_collector],
            name='hdf5_data_collector',
            output='screen'
        ),

        # Viser 3D Frame Visualizer (web-based at http://localhost:8080)
        ExecuteProcess(
            cmd=['python3', frame_visualizer],
            name='viser_frame_visualizer',
            output='screen'
        ),

        # RealSense Camera 1: Front view (observation only)
        # UPDATE: Replace serial_no with your camera's serial number
        Node(
            package='realsense2_camera',
            executable='realsense2_camera_node',
            name='rs_front',
            namespace='rs_front',
            output='screen',
            parameters=[{
                'serial_no': '845112071112',  # Updated
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

        # RealSense Camera 2: Hand tracking camera
        # UPDATE: Replace serial_no with your camera's serial number
        # Node(
        #     package='realsense2_camera',
        #     executable='realsense2_camera_node',
        #     name='rs_hand',
        #     namespace='rs_hand',
        #     output='screen',
        #     parameters=[{
        #         'serial_no': 'YOUR_HAND_CAMERA_SERIAL',  # TODO: Update this
        #         'camera_name': 'rs_hand',
        #         'enable_color': True,
        #         'enable_depth': False,
        #         'enable_infra1': False,
        #         'enable_infra2': False,
        #         'enable_gyro': False,
        #         'enable_accel': False,
        #         'rgb_camera.color_profile': '640x360x30',
        #     }]
        # ),
    ])


def main(argv=sys.argv[1:]):
    """Entry point for the launch file."""
    print("=" * 60)
    print("MediaPipe Hand Tracking Data Collection System")
    print("=" * 60)
    print()
    print("IMPORTANT: Update RealSense serial numbers in this file!")
    print("Find serial numbers with: rs-enumerate-devices | grep Serial")
    print()
    print("Keyboard Controls (in pygame window):")
    print("  R - Reset robot to home position")
    print("  S - Start recording episode")
    print("  D - Done/end recording (saves HDF5)")
    print("  P - Pause recording and robot")
    print("  U - Unpause/resume")
    print("  Q - Quit")
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
