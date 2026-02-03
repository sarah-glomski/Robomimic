#!/usr/bin/env python3

import sys
from launch import LaunchService
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node

def generate_launch_description():
    script_1 = "/home/alex/Documents/3D-Diffusion-Policy/dt_ag/utils/publish_zed.py"
    script_3 = "/home/alex/Documents/3D-Diffusion-Policy/dt_ag/utils/xarm_state_publisher.py"
    script_4 = "/home/alex/Documents/3D-Diffusion-Policy/dt_ag/utils/xarm_velocity_control.py"
    script_2 = "/home/alex/Documents/3D-Diffusion-Policy/dt_ag/utils/publish_dt.py"
    
    return LaunchDescription([
        ExecuteProcess(
            cmd=['python3', script_1],
            name='publish_zed',
            output='screen'
        ),
        ExecuteProcess(
            cmd=['python3', script_3],
            name='xarm_state_publisher',
            output='screen'
        ),
        ExecuteProcess(
            cmd=['python3', script_4],
            name='xarm_velocity_control',
            output='screen'
        ),
        ExecuteProcess(
            cmd=['python3', script_2],
            name='publish_dt',
            output='screen'
        ),

        # # RealSense #1  → namespace/camera1
        # Node(
        #     package='realsense2_camera',
        #     executable='realsense2_camera_node',
        #     name='rs_wrist',
        #     namespace='rs_wrist',
        #     output='screen',
        #     parameters=[{
        #         # change these to your desired resolution / FPS
        #         'serial_no': '317222074520', # unique to wrist camera
        #         'camera_name': 'rs_wrist',
        #         'enable_color': True,
        #         'enable_depth': False,
        #         'rgb_camera.color_profile': '640x360x30',
        #     }]
        # ),

        # RealSense #2  → namespace /camera2
        Node(
            package='realsense2_camera',
            executable='realsense2_camera_node',
            name='rs_side',
            namespace='rs_side',
            output='screen',
            parameters=[{
                'serial_no': '317222074068', # unique to side camera
                'camera_name': 'rs_side',
                'enable_color': True,
                'enable_depth': False,
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
