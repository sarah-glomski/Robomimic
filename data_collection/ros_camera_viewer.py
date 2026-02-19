#!/usr/bin/env python3
"""
ROS Camera Feed Viewer

Subscribes to RealSense compressed image topics published by the
realsense2_camera_node instances and displays them in OpenCV windows.

Launched alongside the camera nodes in launch_sim.py / launch_data_collection.py.
Press 'q' in any OpenCV window to quit.
"""

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


# Raw image topics matching the launch file camera namespaces
# Using raw Image (not compressed) for reliability with enable_sync mode
CAMERA_TOPICS = {
    "rs_front": "/rs_front/rs_front/color/image_raw",
    "rs_wrist": "/rs_wrist/rs_wrist/color/image_raw",
    "rs_head":  "/rs_head/rs_head/color/image_raw",
}


class ROSCameraViewer(Node):
    def __init__(self):
        super().__init__('ros_camera_viewer')

        self._bridge = CvBridge()
        self._frames = {}
        self._first_frame_received = set()

        sensor_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST
        )

        # Create named windows up front so they appear immediately
        for name in CAMERA_TOPICS:
            cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)

        for name, topic in CAMERA_TOPICS.items():
            self.create_subscription(
                Image, topic,
                lambda msg, n=name: self._image_callback(msg, n),
                sensor_qos
            )
            self.get_logger().info(f'Subscribed to {topic}')

        # Display timer at 30Hz
        self.create_timer(1.0 / 30.0, self._display)
        self.get_logger().info('Camera viewer ready — press q in any window to quit')

    def _image_callback(self, msg: Image, name: str):
        try:
            self._frames[name] = self._bridge.imgmsg_to_cv2(
                msg, desired_encoding='bgr8'
            )
            if name not in self._first_frame_received:
                self._first_frame_received.add(name)
                self.get_logger().info(f'First frame received from {name}')
        except Exception as e:
            self.get_logger().error(f'Error decoding {name}: {e}')

    def _display(self):
        for name, image in self._frames.items():
            cv2.imshow(name, image)

        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = ROSCameraViewer()

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
