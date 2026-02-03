#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
import pyzed.sl as sl
import cv2
from cv_bridge import CvBridge
import numpy as np
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class ZedImagePublisher(Node):
    def __init__(self):
        super().__init__('zed_image_publisher')

        sensor_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)

        # Create publishers for RGB, depth, and compressed RGB images
        self.rgb_publisher = self.create_publisher(Image, 'zed_image/rgb', sensor_qos)
        self.compressed_rgb_publisher = self.create_publisher(CompressedImage, 'zed_image/rgb/compressed', sensor_qos)
        self.compressed_depth_publisher = self.create_publisher(CompressedImage, 'zed_image/depth/compressed', sensor_qos)

        self.bridge = CvBridge()
        # Timer for grabbing and publishing images
        self.timer = self.create_timer(1/30.0, self.timer_callback)

        # JPEG compression parameters
        self.jpeg_quality = 80  # Adjustable quality (0-100)
        self.encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]

        # ==========================================================
        # (A) Initialize the ZED camera with provided settings
        # ==========================================================
        self.zed_cam = sl.Camera()
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD720
        init_params.depth_mode = sl.DEPTH_MODE.NEURAL
        init_params.coordinate_units = sl.UNIT.METER
        init_params.depth_stabilization = 1
        init_params.camera_image_flip = sl.FLIP_MODE.AUTO
        init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

        status = self.zed_cam.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            self.get_logger().error(f"ZED open failed: {status}")
            # Ensure the node is properly destroyed on failure
            self.destroy_node()
            raise RuntimeError(f"ZED open failed: {status}")

        self.get_logger().info("ZED camera initialized.")

        # # 1) Disable auto‐exposure/gain so manual settings take effect:
        # self.zed_cam.set_camera_settings(sl.VIDEO_SETTINGS.AEC_AGC, 0)

        # # 2) Increase brightness (range is 0–8; 8 is maximum):
        # self.zed_cam.set_camera_settings(sl.VIDEO_SETTINGS.BRIGHTNESS, 2)

        # ==========================================================
        # (B) Retrieve and log camera calibration information
        # ==========================================================
        camera_information = self.zed_cam.get_camera_information()
        calibration_params = camera_information.camera_configuration.calibration_parameters

        # Access intrinsic parameters for the left camera
        left_cam_intrinsics = calibration_params.left_cam
        focal_left_x = left_cam_intrinsics.fx
        focal_left_y = left_cam_intrinsics.fy
        principal_point_left_x = left_cam_intrinsics.cx
        principal_point_left_y = left_cam_intrinsics.cy
        distortion_coeffs_left = left_cam_intrinsics.disto # Radial and tangential distortion coefficients

        # Access stereo parameters
        baseline_translation_x = calibration_params.stereo_transform.get_translation().get()[0] # Translation between left and right eye on x-axis (baseline)
        # Note: ZED SDK provides baseline directly as tx

        # Log the calibration information
        self.get_logger().info("--- Camera Calibration Parameters (Left Camera) ---")
        self.get_logger().info(f"Focal Length (fx): {focal_left_x}")
        self.get_logger().info(f"Focal Length (fy): {focal_left_y}")
        self.get_logger().info(f"Principal Point (cx): {principal_point_left_x}")
        self.get_logger().info(f"Principal Point (cy): {principal_point_left_y}")
        self.get_logger().info(f"Distortion Coefficients (k1, k2, p1, p2, k3): {distortion_coeffs_left}")
        self.get_logger().info(f"Stereo Baseline (tx): {baseline_translation_x} meters")
        self.get_logger().info("-------------------------------------------------")


        self.zed_left_image = sl.Mat()
        self.zed_depth_map = sl.Mat()
        # Define the desired resolution for retrieval
        self.desired_res = sl.Resolution(640, 360) # 1280x720 resolution

    def timer_callback(self):
        # Grab a new frame from the ZED camera
        if self.zed_cam.grab() == sl.ERROR_CODE.SUCCESS:
            # Retrieve the left image in BGRA format
            self.zed_cam.retrieve_image(self.zed_left_image, sl.VIEW.LEFT, resolution=self.desired_res)
            frame_bgra = self.zed_left_image.get_data()
            # Convert from BGRA to BGR for compatibility with cv_bridge
            frame_bgr = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)

            # Retrieve the depth map
            self.zed_cam.retrieve_measure(self.zed_depth_map, sl.MEASURE.DEPTH, resolution=self.desired_res)
            depth_data = self.zed_depth_map.get_data()

            try:
                # Convert OpenCV images to ROS Image messages
                # Use bgr8 encoding for color image
                # rgb_msg = self.bridge.cv2_to_imgmsg(frame_bgr, encoding="bgr8")
                depth16 = (depth_data*1000).astype(np.uint16)

                # Encode the BGR image as JPEG
                success_rgb, encoded_image = cv2.imencode('.jpg', frame_bgr, self.encode_params)        
                success_depth, png = cv2.imencode(".png", depth16, [cv2.IMWRITE_PNG_COMPRESSION,3])

                # Create compressed RGB image
                compressed_rgb_msg = CompressedImage()
                compressed_depth_image = CompressedImage()
                
                if success_rgb:
                    compressed_rgb_msg.data = encoded_image.tobytes()
                    compressed_rgb_msg.format = "jpeg"
                else:
                    self.get_logger().error("Failed to encode image as JPEG")
                    return
                
                if success_depth:
                    compressed_depth_image.data = png.tobytes()
                    compressed_depth_image.format = "16UC1"
                else:
                    self.get_logger().error("Failed to encode depth image as 16UC1")
                    return

                # Set headers with current time and frame_id
                current_time = self.get_clock().now().to_msg()

                # rgb_msg.header.stamp = current_time
                # rgb_msg.header.frame_id = "zed_camera" # Consistent frame ID

                compressed_depth_image.header.stamp = current_time
                compressed_depth_image.header.frame_id = "zed_camera" # Consistent frame ID

                compressed_rgb_msg.header.stamp = current_time
                compressed_rgb_msg.header.frame_id = "zed_camera" # Consistent frame ID

                # Publish messages
                # self.rgb_publisher.publish(rgb_msg)
                self.compressed_depth_publisher.publish(compressed_depth_image)
                self.compressed_rgb_publisher.publish(compressed_rgb_msg)

                self.get_logger().debug("Published ZED RGB, depth, and compressed RGB images.")

            except Exception as e:
                self.get_logger().error(f"Image conversion or publishing error: {e}")
                # Continue to next timer callback even if one fails

    def destroy_node(self):
        # Close the ZED camera connection when the node is destroyed
        if self.zed_cam.is_opened():
            self.zed_cam.close()
            self.get_logger().info("ZED camera closed.")
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = ZedImagePublisher()
    try:
        # Keep the node alive
        rclpy.spin(node)
    except KeyboardInterrupt:
        # Handle Ctrl+C
        node.get_logger().info("Keyboard interrupt received, shutting down.")
    finally:
        # Ensure the node is destroyed and resources are released
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()