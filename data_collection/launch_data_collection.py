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

    # Append script_dir to existing PYTHONPATH (preserves ROS2 paths)
    current_pythonpath = os.environ.get('PYTHONPATH', '')
    new_pythonpath = f"{script_dir}:{current_pythonpath}" if current_pythonpath else script_dir

    # Python scripts to launch
    state_publisher = os.path.join(script_dir, 'xarm_state_publisher.py')
    hand_tracker = os.path.join(script_dir, 'mediapipe_hand_tracker.py')
    hand_controller = os.path.join(script_dir, 'xarm_hand_controller.py')
    xarm_simulator = os.path.join(script_dir, 'xarm_simulator.py')
    data_collector = os.path.join(script_dir, 'hdf5_data_collector.py')
    frame_visualizer = os.path.join(script_dir, 'viser_frame_visualizer.py')

    # Object-relative frame parameters (shared across nodes)
    # Edit these to match your physical object positions
    obj_params = [
        '--ros-args',
        '-p', 'human_object_x:=0.25',
        '-p', 'human_object_y:=-0.25',
        '-p', 'human_object_z:=0.10',
        '-p', 'robot_object_x:=0.25',
        '-p', 'robot_object_y:=0.00',
        '-p', 'robot_object_z:=0.10',
        '-p', 'use_object_relative:=true',
    ]

    # Camera position/orientation parameters
    # Edit these to match your camera mounting position relative to robot base
    # Default: camera 0.8m in front of robot, 0.5m high, looking towards robot
    camera_params = [
        '--ros-args',
        '-p', 'camera_x:=1.0',              # Forward from robot base (meters)
        '-p', 'camera_y:=0.0',              # Left/right (0 = centered)
        '-p', 'camera_z:=0.0',              # Height above robot base
        '-p', 'camera_orientation:=towards_robot',  # 'towards_robot', 'away_from_robot', or 'down'
    ]

    return LaunchDescription([
        # XArm State Publisher (requires robot to be powered on)
        # ExecuteProcess(
        #     cmd=['python3', state_publisher],
        #     name='xarm_state_publisher',
        #     output='screen',
        #     cwd=script_dir,
        #     additional_env={'PYTHONPATH': new_pythonpath},
        # ),

        # MediaPipe Hand Tracker (with depth-based 3D tracking)
        ExecuteProcess(
            cmd=['python3', hand_tracker] + camera_params,
            name='mediapipe_hand_tracker',
            output='screen',
            cwd=script_dir,
            additional_env={'PYTHONPATH': new_pythonpath},
        ),

        # xArm Simulator (for testing without real robot)
        # Uncomment for simulation mode
        # ExecuteProcess(
        #     cmd=['python3', xarm_simulator] + obj_params,
        #     name='xarm_simulator',
        #     output='screen',
        #     cwd=script_dir,
        #     additional_env={'PYTHONPATH': new_pythonpath},
        # ),

        # XArm Hand Controller (REAL ROBOT)
        # Mode 7: Cartesian online trajectory planning (smooth, speed-limited)
        ExecuteProcess(
            cmd=['python3', hand_controller] + obj_params,
            name='xarm_hand_controller',
            output='screen',
            cwd=script_dir,
            additional_env={'PYTHONPATH': new_pythonpath},
        ),

        # HDF5 Data Collector (runs pygame in main thread)
        ExecuteProcess(
            cmd=['python3', data_collector] + obj_params,
            name='hdf5_data_collector',
            output='screen',
            cwd=script_dir,
            additional_env={'PYTHONPATH': new_pythonpath},
        ),

        # Viser 3D Frame Visualizer (web-based at http://localhost:8080)
        ExecuteProcess(
            cmd=['python3', frame_visualizer] + obj_params + camera_params[1:],  # Skip '--ros-args' duplicate
            name='viser_frame_visualizer',
            output='screen',
            cwd=script_dir,
            additional_env={'PYTHONPATH': new_pythonpath},
        ),

        # RealSense Camera 1: Front view (color + depth for 3D hand tracking)
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
                'enable_depth': True,          # Enable depth for 3D tracking
                'align_depth.enable': True,    # Align depth to color frame
                'enable_infra1': False,
                'enable_infra2': False,
                'enable_gyro': False,
                'enable_accel': False,
                'rgb_camera.color_profile': '640x360x30',
                'depth_module.depth_profile': '640x360x30',
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
    print(">>> REAL ROBOT MODE <<<")
    print("Using Mode 7: Cartesian online trajectory planning")
    print("Speed limit: 400 mm/s | Keep emergency stop accessible!")
    print("To use simulator: edit launch file to enable xarm_simulator")
    print("and disable xarm_hand_controller.")
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
    print("Safety Features:")
    print("  - Workspace bounds enforced")
    print("  - Velocity limits: 150mm/s max")
    print("  - Position delta limits per step")
    print()
    print("Object-Relative Mode:")
    print("  Human object frame: (0.30, -0.20, 0.10)")
    print("  Robot object frame:  (0.25, 0.05, 0.10)")
    print("  Robot mirrors hand motion relative to its object")
    print()
    print("3D Hand Tracking (Depth-based):")
    print("  Camera position: (1.0, 0.0, 0.0) - 1.0m forward, 0.0m high")
    print("  Camera orientation: towards_robot")
    print("  Edit launch file camera_params to match your setup")
    print()
    print("=" * 60)

    ld = generate_launch_description()
    ls = LaunchService(argv=argv)
    ls.include_launch_description(ld)
    return ls.run()


if __name__ == '__main__':
    sys.exit(main())
