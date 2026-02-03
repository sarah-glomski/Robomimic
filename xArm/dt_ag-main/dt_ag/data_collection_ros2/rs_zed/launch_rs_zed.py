#!/usr/bin/env python3

import sys
from launch import LaunchService
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node

def generate_launch_description():
    script_1 = "/home/alex/Documents/3D-Diffusion-Policy/dt_ag/data_collection_ros2/publish_zed_data_collection.py"
    script_5 = "/home/alex/Documents/3D-Diffusion-Policy/dt_ag/data_collection_ros2/xarm_spacemouse_ros2.py"
    script_2 = "/home/alex/Documents/3D-Diffusion-Policy/dt_ag/data_collection_ros2/rs_zed/rs_zed_hdf5_collector.py"
    
    return LaunchDescription([
        ExecuteProcess(
            cmd=['python3', script_1],
            name='publish_zed',
            output='screen'
        ),
        ExecuteProcess(
            cmd=['python3', script_2],
            name='rs_zed_hdf5_collector',
            output='screen'
        ),
        ExecuteProcess(
            cmd=['python3', script_5],
            name='xarm_spacemouse_ros2',
            output='screen'
        ),

        # RealSense #1  → namespace/camera1
        Node(
            package='realsense2_camera',
            executable='realsense2_camera_node',
            name='rs_wrist',
            namespace='rs_wrist',
            output='screen',
            parameters=[{
                # change these to your desired resolution / FPS
                'serial_no': '317222074520', # unique to wrist camera
                'camera_name': 'rs_wrist',
                'enable_color': True,
                'enable_depth': False,
                'enable_infra1': False,
                'enable_infra2': False,
                'enable_gyro': False,
                'enable_accel': False,
                'rgb_camera.color_profile': '640x360x30',
            }]
        ),

        # RealSense #2  → namespace /camera2
        Node(
            package='realsense2_camera',
            executable='realsense2_camera_node',
            name='rs_front',
            namespace='rs_front',
            output='screen',
            parameters=[{
                'serial_no': '317222074068', # unique to front camera
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

        # RealSense #3  → namespace /camera3
        Node(
            package='realsense2_camera',
            executable='realsense2_camera_node',
            name='rs_side',
            namespace='rs_side',
            output='screen',
            parameters=[{
                'serial_no': '_040322073693', # unique to side camera
                'camera_name': 'rs_side',
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
    ld = generate_launch_description()
    ls = LaunchService(argv=argv)
    ls.include_launch_description(ld)
    return ls.run()

if __name__ == '__main__':
    sys.exit(main())
