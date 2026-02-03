#!/usr/bin/env python3
"""
Dual-DenseTact publisher + amplified frame-to-frame difference stream
"""

import os, subprocess, cv2, numpy as np, rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge


class DualCameraPublisher(Node):
    # ──────────────────────────────────────────────────────────────────
    def __init__(self):
        super().__init__('dt_dual_camera_publisher')

        # parameter for amplification
        self.diff_gain = 8.0

        # camera IDs ---------------------------------------------------
        # self.left_id = 8
        # self.right_id = 10

        # Cam ids when 3 realsense cameras and zed are connected (dt's are plugged in last, left (tape) first then right)
        self.left_id = 20  
        self.right_id = 22  

        # manual cam settings
        self._wb_temp = 6500
        self._exposure = 50

        # publishers ---------------------------------------------------
        self._init_publishers()

        # OpenCV capture ----------------------------------------------
        self.left_cap = cv2.VideoCapture(self.left_id)
        self.right_cap = cv2.VideoCapture(self.right_id)
        for cap in (self.left_cap, self.right_cap):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not self.left_cap.isOpened():
            self.get_logger().error(f"Unable to open camera {self.left_id}")
        if not self.right_cap.isOpened():
            self.get_logger().error(f"Unable to open camera {self.right_id}")

        # helpers ------------------------------------------------------
        self.bridge = CvBridge()
        self.encode_params = [cv2.IMWRITE_JPEG_QUALITY, 80]    # jpeg Q-80
        self.count = 0
        self._settings_ok = False

        self.timer = self.create_timer(1/15.0, self.timer_cb)  # 15 Hz

    # ─── parameter callback ─────────────────────────────────────────
    def _param_cb(self, params):
        for p in params:
            if p.name == 'diff_gain' and p.type_ == rclpy.parameter.Parameter.Type.DOUBLE:
                self.diff_gain = max(1.0, p.value)
                self.get_logger().info(f"diff_gain set to {self.diff_gain}")
        return rclpy.parameter.SetParametersResult(successful=True)

    # ─── publisher initialisation ───────────────────────────────────
    def _init_publishers(self):
        self.left_comp_pub = self.create_publisher(CompressedImage, f'RunCamera/image_raw_{self.left_id}/compressed', 20)
        self.right_comp_pub = self.create_publisher(CompressedImage, f'RunCamera/image_raw_{self.right_id}/compressed', 20)

    # ─── camera control helpers (unchanged) ─────────────────────────
    @staticmethod
    def _v4l2_get(video_id, ctrl):
        cmd = f"v4l2-ctl --device /dev/video{video_id} -C {ctrl}"
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if res.returncode or not res.stdout:
            return None
        try:
            return int(res.stdout.strip().split(': ')[1])
        except ValueError:
            return res.stdout.strip()

    def _v4l2_set_seq(self, cmds):
        for c in cmds:
            if os.system(c):
                self.get_logger().warning(f"cmd failed: {c}")

    def _ensure_manual_settings(self):
        if self._settings_ok:
            return
        self.get_logger().info("Applying manual WB / exposure …")

        for vid in (self.left_id, self.right_id):
            if (self._v4l2_get(vid, "white_balance_automatic") != 0 or
                self._v4l2_get(vid, "white_balance_temperature") != self._wb_temp):
                self._v4l2_set_seq([
                    f"v4l2-ctl -d /dev/video{vid} -c white_balance_automatic=0",
                    f"v4l2-ctl -d /dev/video{vid} -c white_balance_temperature={self._wb_temp}"
                ])
            if (self._v4l2_get(vid, "auto_exposure") != 1 or
                self._v4l2_get(vid, "exposure_time_absolute") != self._exposure):
                self._v4l2_set_seq([
                    f"v4l2-ctl -d /dev/video{vid} -c auto_exposure=1",
                    f"v4l2-ctl -d /dev/video{vid} -c exposure_time_absolute={self._exposure}"
                ])
        self._settings_ok = True

    # ─── main timer ─────────────────────────────────────────────────
    def timer_cb(self):
        self.count += 1
        if self.count == 5:
            self._ensure_manual_settings()

        # left
        ok_l, frm_l = self.left_cap.read()
        if ok_l:
            frame_resized = cv2.resize(frm_l, (320, 180))
            ok, buf = cv2.imencode('.jpg', frame_resized, self.encode_params)
            if ok:
                comp_msg = CompressedImage()
                comp_msg.header.stamp = self.get_clock().now().to_msg()
                comp_msg.header.frame_id = f'camera_{self.left_id}_frame'
                comp_msg.format = 'jpeg'
                comp_msg.data = buf.tobytes()
                self.left_comp_pub.publish(comp_msg)
            else:
                self.get_logger().warning(f"Camera {self.left_id}: JPEG encode failed")
        else:
            self.get_logger().warning(f"Camera {self.left_id}: grab failed")

        # right
        ok_r, frm_r = self.right_cap.read()
        if ok_r:
            frame_resized = cv2.resize(frm_r, (320, 180))
            ok, buf = cv2.imencode('.jpg', frame_resized, self.encode_params)
            if ok:
                comp_msg = CompressedImage()
                comp_msg.header.stamp = self.get_clock().now().to_msg()
                comp_msg.header.frame_id = f'camera_{self.right_id}_frame'
                comp_msg.format = 'jpeg'
                comp_msg.data = buf.tobytes()
                self.right_comp_pub.publish(comp_msg)
            else:
                self.get_logger().warning(f"Camera {self.right_id}: JPEG encode failed")
        else:
            self.get_logger().warning(f"Camera {self.right_id}: grab failed")

    # ─── cleanup ────────────────────────────────────────────────────
    def destroy_node(self):
        if self.left_cap.isOpened():  self.left_cap.release()
        if self.right_cap.isOpened(): self.right_cap.release()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DualCameraPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()